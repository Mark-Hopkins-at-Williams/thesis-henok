"""Offline autocomplete compression of a parallel corpus, and a reader for the result.

Compressing on the fly costs an autocomplete-model forward pass per sentence and made
tokenization the bottleneck of training. Here a corpus is compressed once, in batches,
and stored as memory-mappable numpy arrays.

Layout of an output directory (one per split):

    shard-00000/ shard-00001/ ...     one directory per block of `shard_lines` input lines
        src_base.npy   uint16   ids (2..258) of the kept tokens of every kept pair, concatenated
        src_run.npy    uint8    number of removed bytes after each kept token (clipped at 255)
        src_offsets.npy int64   pair j owns [offsets[j], offsets[j+1]) of the two arrays above
        tgt_*.npy               the same for the target side
        line_ids.npy   int64    line number in the original files of each kept pair
        meta.json               counts, timing and the settings the shard was made with
    manifest.json                written by finalize(): shard list, totals, settings

The uncapped run lengths are stored, so one pass serves every cap. A token id is
`base + 256 * min(run, cap)`; the language token (1) is prepended when reading. Pairs
are kept in their original order (the training files are length-batched, so do not
shuffle). A pair is dropped if either side is empty, is longer than `max_bytes` bytes,
or is not valid UTF-8.

Shards are independent: several workers (one per GPU) can split them with
`worker`/`num_workers`, and a finished shard is skipped on restart.
"""

import hashlib
import itertools
import json
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

FORMAT_VERSION = 1
LANG_ID, EOS_ID = 1, 2
MANIFEST = "manifest.json"
ARRAYS = [
    "src_base",
    "src_run",
    "src_offsets",
    "tgt_base",
    "tgt_run",
    "tgt_offsets",
    "line_ids",
]


def shard_name(shard_id):
    return f"shard-{shard_id:05d}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def tokenizer_settings(tok, model_dir=None):
    """What identifies a tokenizer's output, recorded for provenance."""
    checkpoint = Path(model_dir) / "best_model.pt" if model_dir is not None else None
    return {
        "model_dir": str(model_dir) if model_dir is not None else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "keep_first": tok.keep_first,
        "keep_eos": tok.keep_eos,
        "prediction_threshold": tok.prediction_threshold,
        "prediction_mode": tok.prediction_mode,
    }


def _skip(handle, n):
    deque(itertools.islice(handle, n), maxlen=0)


def _read_lines(handle, n):
    return [line.rstrip(b"\n") for line in itertools.islice(handle, n)]


def filter_pairs(src_raw, tgt_raw, max_bytes):
    """Returns (kept indices, src texts, tgt texts, drop statistics)."""
    keep, src_txt, tgt_txt = [], [], []
    stats = {"dropped_empty": 0, "dropped_long": 0, "dropped_undecodable": 0}
    for j, (a, b) in enumerate(zip(src_raw, tgt_raw)):
        if not a or not b:
            stats["dropped_empty"] += 1
        elif len(a) > max_bytes or len(b) > max_bytes:
            stats["dropped_long"] += 1
        else:
            try:
                sa, sb = a.decode("utf-8"), b.decode("utf-8")
            except UnicodeDecodeError:
                stats["dropped_undecodable"] += 1
                continue
            keep.append(j)
            src_txt.append(sa)
            tgt_txt.append(sb)
    return keep, src_txt, tgt_txt, stats


def compress_side(tok, sentences, chunk=50000):
    bases, runs, lengths = [], [], []
    for i in range(0, len(sentences), chunk):
        for base, run in tok.compress_runs(sentences[i : i + chunk]):
            bases.append(base)
            runs.append(run)
            lengths.append(len(base))
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    return (
        np.concatenate(bases) if bases else np.zeros(0, dtype=np.uint16),
        np.concatenate(runs) if runs else np.zeros(0, dtype=np.uint8),
        offsets,
    )


def first_mismatch(arrays, side, tok, texts, n):
    """Recompress the first n texts one at a time and compare with the stored arrays.
    Returns the number of mismatching sentences among those checked."""
    saved = tok.max_length_encoding, tok.max_length
    tok.max_length_encoding, tok.max_length = 255, None
    bad = 0
    try:
        for j in range(min(n, len(texts))):
            a, b = arrays[f"{side}_offsets"][j], arrays[f"{side}_offsets"][j + 1]
            base = arrays[f"{side}_base"][a:b].astype(np.int64)
            run = arrays[f"{side}_run"][a:b].astype(np.int64)
            got = [LANG_ID] + (base + 256 * run).tolist()
            bad += got != tok(texts[j])
    finally:
        tok.max_length_encoding, tok.max_length = saved
    return bad


def process_shard(
    shard_id,
    first_line,
    src_raw,
    tgt_raw,
    src_tok,
    tgt_tok,
    out_dir,
    settings,
    verify=0,
    chunk=50000,
):
    """Filters, compresses and writes one shard; returns its metadata."""
    start = time.time()
    keep, src_txt, tgt_txt, stats = filter_pairs(src_raw, tgt_raw, settings["max_bytes"])
    arrays = {}
    for side, tok, texts in (("src", src_tok, src_txt), ("tgt", tgt_tok, tgt_txt)):
        base, run, offsets = compress_side(tok, texts, chunk)
        arrays[f"{side}_base"], arrays[f"{side}_run"] = base, run
        arrays[f"{side}_offsets"] = offsets
    arrays["line_ids"] = first_line + np.array(keep, dtype=np.int64)

    mismatches = None
    if verify:
        mismatches = first_mismatch(arrays, "src", src_tok, src_txt, verify)
        mismatches += first_mismatch(arrays, "tgt", tgt_tok, tgt_txt, verify)
        checked = 2 * min(verify, len(keep))
        if checked and mismatches > max(2, 0.02 * checked):
            raise RuntimeError(
                f"shard {shard_id}: {mismatches}/{checked} sentences differ between "
                "batched and per-sentence compression"
            )

    out_dir = Path(out_dir)
    tmp = out_dir / (shard_name(shard_id) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for name, array in arrays.items():
        np.save(tmp / f"{name}.npy", array)
    meta = {
        "shard_id": shard_id,
        "first_line": first_line,
        "lines_read": len(src_raw),
        "pairs": len(keep),
        **stats,
        "src_tokens": int(len(arrays["src_base"])),
        "tgt_tokens": int(len(arrays["tgt_base"])),
        "src_bytes": sum(len(s.encode()) + 1 for s in src_txt),
        "tgt_bytes": sum(len(s.encode()) + 1 for s in tgt_txt),
        "verify_mismatches": mismatches,
        "seconds": round(time.time() - start, 1),
        "settings": settings,
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    tmp.rename(out_dir / shard_name(shard_id))  # atomic: a complete shard has no .tmp
    return meta


def run_worker(
    src_file,
    tgt_file,
    out_dir,
    src_tok,
    tgt_tok,
    settings,
    worker=0,
    num_workers=1,
    max_shards=None,
    verify=0,
    log=print,
):
    """Processes shards worker, worker + num_workers, ... of the two parallel files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_lines = settings["shard_lines"]
    position, shard_id, handled = 0, worker, 0
    with open(src_file, "rb") as fs, open(tgt_file, "rb") as ft:
        while max_shards is None or handled < max_shards:
            first_line = shard_id * shard_lines
            _skip(fs, first_line - position)
            _skip(ft, first_line - position)
            src_raw, tgt_raw = _read_lines(fs, shard_lines), _read_lines(ft, shard_lines)
            if len(src_raw) != len(tgt_raw):
                raise ValueError(
                    f"files differ in length near line {first_line}: "
                    f"{len(src_raw)} vs {len(tgt_raw)} lines in shard {shard_id}"
                )
            if not src_raw:
                break
            position = first_line + len(src_raw)
            if (out_dir / shard_name(shard_id)).exists():
                log(f"shard {shard_id}: already done, skipping")
            else:
                meta = process_shard(
                    shard_id, first_line, src_raw, tgt_raw, src_tok, tgt_tok,
                    out_dir, settings, verify=verify if handled == 0 else 0,
                )
                log(
                    f"shard {shard_id}: {meta['pairs']}/{meta['lines_read']} pairs kept "
                    f"(long {meta['dropped_long']}, empty {meta['dropped_empty']}), "
                    f"{meta['seconds']}s, verify mismatches {meta['verify_mismatches']}"
                )
            handled += 1
            shard_id += num_workers
    return handled


def finalize(out_dir):
    """Checks that all shards exist and agree on settings, and writes the manifest."""
    out_dir = Path(out_dir)
    leftovers = sorted(p.name for p in out_dir.glob("shard-*.tmp"))
    if leftovers:
        raise ValueError(f"unfinished shards present: {leftovers}")
    dirs = sorted(p for p in out_dir.glob("shard-?????") if p.is_dir())
    ids = [int(p.name.split("-")[1]) for p in dirs]
    if not dirs or ids != list(range(len(ids))):
        missing = sorted(set(range(max(ids, default=-1) + 1)) - set(ids))
        raise ValueError(f"shards are not contiguous from 0; missing {missing}")
    metas = [json.loads((p / "meta.json").read_text()) for p in dirs]
    settings = metas[0]["settings"]
    for m in metas:
        if m["settings"] != settings:
            raise ValueError(f"shard {m['shard_id']} was made with different settings")
    for m in metas[:-1]:
        if m["lines_read"] != settings["shard_lines"]:
            raise ValueError(f"shard {m['shard_id']} is partial but not the last one")
    totals = {
        k: sum(m[k] for m in metas)
        for k in (
            "lines_read", "pairs", "dropped_empty", "dropped_long",
            "dropped_undecodable", "src_tokens", "tgt_tokens", "src_bytes", "tgt_bytes",
        )
    }
    manifest = {
        "format_version": FORMAT_VERSION,
        "settings": settings,
        "totals": totals,
        # tokens (language token included) over bytes + EOS + language token, as elsewhere
        "src_compression_ratio": (totals["src_tokens"] + totals["pairs"])
        / (totals["src_bytes"] + totals["pairs"]),
        "tgt_compression_ratio": (totals["tgt_tokens"] + totals["pairs"])
        / (totals["tgt_bytes"] + totals["pairs"]),
        "shards": [{"id": m["shard_id"], "pairs": m["pairs"], "first_line": m["first_line"]} for m in metas],
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


class UncompressedBytes:
    """Stand-in for the autocomplete tokenizer that removes nothing.

    Used for the uncompressed byte-target control: every byte and the EOS are kept, so
    the stored runs are all zero. Offers just what precomputation and its verification
    need from a tokenizer.
    """

    keep_first = keep_eos = True
    prediction_threshold = None
    prediction_mode = "none"
    max_length_encoding = 0
    max_length = None

    def compress_runs(self, sents, max_tokens_per_forward=None):
        out = []
        for s in sents:
            data = s.encode()
            base = np.empty(len(data) + 1, dtype=np.uint16)
            base[:-1] = np.frombuffer(data, dtype=np.uint8).astype(np.uint16) + 3
            base[-1] = EOS_ID
            out.append((base, np.zeros(len(base), dtype=np.uint8)))
        return out

    def __call__(self, sent):
        return [LANG_ID] + [b + 3 for b in sent.encode()] + [EOS_ID]


class PrecomputedPairs:
    """Random access to a finalized directory as (src_tokens, tgt_tokens) lists.

    Token ids are identical to what the AutocompletingTokenizer would produce with the
    given caps and maximum lengths (language token first, truncation ending in EOS).
    """

    def __init__(
        self, split_dir, src_cap, tgt_cap, src_max_length=None, tgt_max_length=None
    ):
        if not (0 <= src_cap <= 255 and 0 <= tgt_cap <= 255):
            raise ValueError("caps must be in 0..255 (runs are stored clipped at 255)")
        self.dir = Path(split_dir)
        manifest = json.loads((self.dir / MANIFEST).read_text())
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported precomputed format")
        self.manifest = manifest
        counts = [s["pairs"] for s in manifest["shards"]]
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.caps = {"src": src_cap, "tgt": tgt_cap}
        self.max_lengths = {"src": src_max_length, "tgt": tgt_max_length}
        self._open = {}

    def __len__(self):
        return int(self.starts[-1])

    def _shard(self, k):
        if k not in self._open:
            path = self.dir / shard_name(k)
            self._open[k] = {
                name: np.load(path / f"{name}.npy", mmap_mode="r") for name in ARRAYS
            }
        return self._open[k]

    def _locate(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        k = int(np.searchsorted(self.starts, i, side="right") - 1)
        return k, i - int(self.starts[k])

    def _tokens(self, arrays, side, j):
        offsets = arrays[f"{side}_offsets"]
        a, b = int(offsets[j]), int(offsets[j + 1])
        base = arrays[f"{side}_base"][a:b].astype(np.int64)
        run = arrays[f"{side}_run"][a:b].astype(np.int64)
        tokens = [LANG_ID] + (base + 256 * np.minimum(run, self.caps[side])).tolist()
        limit = self.max_lengths[side]
        if limit is not None and len(tokens) > limit - 1:
            tokens = tokens[: limit - 1] + [EOS_ID]
        return tokens

    def __getitem__(self, i):
        k, j = self._locate(i)
        arrays = self._shard(k)
        return self._tokens(arrays, "src", j), self._tokens(arrays, "tgt", j)

    def line_id(self, i):
        k, j = self._locate(i)
        return int(self._shard(k)["line_ids"][j])

    def iter_from(self, start=0):
        """Yields pairs in order from pair index `start` (used to resume training)."""
        for i in range(start, len(self)):
            yield self[i]


def collate_pairs(batch, src_pad=0, tgt_pad=-100):
    """Pads a list of (src_tokens, tgt_tokens) into the input/label dicts of the model."""
    src, tgt = zip(*batch)
    src_padded = pad_sequence(
        [torch.tensor(x) for x in src], batch_first=True, padding_value=src_pad
    )
    tgt_padded = pad_sequence(
        [torch.tensor(x) for x in tgt], batch_first=True, padding_value=tgt_pad
    )
    return (
        {"input_ids": src_padded, "attention_mask": (src_padded != src_pad).int()},
        {"input_ids": tgt_padded, "attention_mask": (tgt_padded != tgt_pad).int()},
    )


class PrecomputedBatches:
    """Batches of a PrecomputedPairs in the form finetune.py and validate.py consume.

    Iterating yields (src, tgt, metadata) like MixtureOfBitexts. Pairs are read in
    order with no shuffling. A training instance (only_once_thru=False) wraps around
    at the end of the data; an evaluation instance stops. `position` is the index of
    the next pair and can be saved and restored with seek() to resume exactly.
    """

    def __init__(
        self,
        pairs,
        batch_size,
        src_pad=0,
        tgt_pad=-100,
        metadata=None,
        only_once_thru=False,
    ):
        if len(pairs) == 0:
            raise ValueError("no pairs to iterate over")
        self.pairs = pairs
        self.batch_size = batch_size
        self.src_pad, self.tgt_pad = src_pad, tgt_pad
        self.metadata = metadata or {}
        self.only_once_thru = only_once_thru
        self.position = 0

    def restart(self):
        self.position = 0

    def seek(self, pair_index):
        self.position = pair_index % len(self.pairs)

    def __iter__(self):
        while True:
            if self.position >= len(self.pairs):
                if self.only_once_thru:
                    return
                self.position = 0
            end = min(self.position + self.batch_size, len(self.pairs))
            batch = [self.pairs[i] for i in range(self.position, end)]
            self.position = end
            src, tgt = collate_pairs(batch, self.src_pad, self.tgt_pad)
            yield src, tgt, self.metadata
