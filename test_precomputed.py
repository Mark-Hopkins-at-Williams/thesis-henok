import json
import random
import tempfile
import unittest
from pathlib import Path

from precomputed import (
    PrecomputedPairs,
    finalize,
    run_worker,
    shard_name,
    tokenizer_settings,
)
from test_target_side import make_tokenizer, random_text

MAX_BYTES = 30
SHARD_LINES = 10


def make_corpus(directory, n=53, seed=7):
    """Writes parallel files with some empty and some too-long lines; returns the
    (src, tgt) texts of the pairs that should survive, with their line numbers."""
    rng = random.Random(seed)

    def text(n, surprises=False):  # random bytes must not contain line breaks
        return random_text(rng, n, surprises).replace("\n", "x").replace("\r", "x")

    src, tgt = [], []
    for i in range(n):
        src.append(text(rng.randrange(1, 25), surprises=bool(i % 2)))
        tgt.append(text(rng.randrange(1, 25), surprises=True))
    src[3] = ""  # empty source
    tgt[17] = ""  # empty target
    src[20] = text(60)  # too long
    tgt[41] = text(60)  # too long
    (directory / "src.txt").write_text("\n".join(src) + "\n")
    (directory / "tgt.txt").write_text("\n".join(tgt) + "\n")
    kept = [
        i
        for i in range(n)
        if src[i] and tgt[i] and len(src[i]) <= MAX_BYTES and len(tgt[i]) <= MAX_BYTES
    ]
    return src, tgt, kept


def process(directory, out, worker=0, num_workers=1, **kwargs):
    src_tok = make_tokenizer(cap=8)
    tgt_tok = make_tokenizer(cap=8, keep_first=True, keep_eos=True)
    settings = {
        "max_bytes": MAX_BYTES,
        "shard_lines": SHARD_LINES,
        "src": tokenizer_settings(src_tok),
        "tgt": tokenizer_settings(tgt_tok),
    }
    return run_worker(
        directory / "src.txt",
        directory / "tgt.txt",
        out,
        src_tok,
        tgt_tok,
        settings,
        worker=worker,
        num_workers=num_workers,
        log=lambda s: None,
        **kwargs,
    )


class TestPrecompute(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.out = self.dir / "out"
        self.src, self.tgt, self.kept = make_corpus(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self):
        process(self.dir, self.out, worker=0, num_workers=2, verify=5)
        process(self.dir, self.out, worker=1, num_workers=2, verify=5)
        return finalize(self.out)

    def test_filtering_stats_and_order(self):
        manifest = self.build()
        totals = manifest["totals"]
        self.assertEqual(totals["lines_read"], 53)
        self.assertEqual(totals["pairs"], len(self.kept))
        self.assertEqual(totals["dropped_empty"], 2)
        self.assertEqual(totals["dropped_long"], 2)
        self.assertEqual(len(manifest["shards"]), 6)  # 53 lines / 10 per shard
        reader = PrecomputedPairs(self.out, 8, 8)
        self.assertEqual(len(reader), len(self.kept))
        self.assertEqual([reader.line_id(i) for i in range(len(reader))], self.kept)

    def test_reader_matches_per_sentence_tokenizer(self):
        self.build()
        for src_cap, tgt_cap in ((8, 8), (1, 3), (16, 16)):
            for src_len, tgt_len in ((None, None), (12, 9)):
                reader = PrecomputedPairs(self.out, src_cap, tgt_cap, src_len, tgt_len)
                src_tok = make_tokenizer(cap=src_cap, max_length=src_len)
                tgt_tok = make_tokenizer(
                    cap=tgt_cap, keep_first=True, keep_eos=True, max_length=tgt_len
                )
                for i, line in enumerate(self.kept):
                    self.assertEqual(
                        reader[i],
                        (src_tok(self.src[line]), tgt_tok(self.tgt[line])),
                        f"caps={src_cap},{tgt_cap} lengths={src_len},{tgt_len} pair {i}",
                    )

    def test_source_and_target_use_their_own_boundary_settings(self):
        self.build()
        reader = PrecomputedPairs(self.out, 8, 8)
        for i in range(len(reader)):
            src, tgt = reader[i]
            self.assertEqual((tgt[0], tgt[-1]), (1, 2))  # target boundaries kept

    def test_iter_from_resumes_in_place(self):
        self.build()
        reader = PrecomputedPairs(self.out, 8, 8)
        everything = list(reader.iter_from(0))
        self.assertEqual(list(reader.iter_from(17)), everything[17:])
        self.assertEqual(list(reader.iter_from(len(reader))), [])

    def test_finished_shards_are_skipped_on_restart(self):
        self.build()
        marker = self.out / shard_name(2) / "meta.json"
        before = marker.stat().st_mtime_ns
        process(self.dir, self.out, worker=0, num_workers=2)  # shard 2 belongs to worker 0
        self.assertEqual(marker.stat().st_mtime_ns, before)

    def test_finalize_detects_missing_shards(self):
        process(self.dir, self.out, worker=0, num_workers=2)  # only the even shards
        with self.assertRaises(ValueError) as caught:
            finalize(self.out)
        self.assertIn("missing", str(caught.exception))

    def test_finalize_detects_mixed_settings(self):
        self.build()
        meta_path = self.out / shard_name(1) / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["settings"]["max_bytes"] = 999
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            finalize(self.out)

    def test_misaligned_files_are_rejected(self):
        with open(self.dir / "tgt.txt", "a") as f:
            f.write("extra line\n")
        with self.assertRaises(ValueError):
            process(self.dir, self.out)

    def test_cap_out_of_range(self):
        self.build()
        with self.assertRaises(ValueError):
            PrecomputedPairs(self.out, 256, 8)


if __name__ == "__main__":
    unittest.main()
