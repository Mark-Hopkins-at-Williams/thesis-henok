"""Compress a parallel corpus once, offline, with the autocomplete models.

Run from the repository root. One process per GPU, each taking every
num_workers-th shard, e.g. for a training split on GPUs 0-3:

    D=/mnt/storage/hopkins/data/wmt19/ru-en
    for w in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$w nohup python scripts/precompute_compressed.py \\
        --src_file $D/organized/train.en --tgt_file $D/organized/train.ru \\
        --src_model experiments/autocomplete-v1 --tgt_model experiments/ru-autocomplete-v1 \\
        --tgt_keep_boundaries --out_dir $D/compressed/train \\
        --worker $w --num_workers 4 > logs/precompute_$w.log 2>&1 &
    done
    # when all workers have finished:
    python scripts/precompute_compressed.py --finalize --out_dir $D/compressed/train

Interrupted workers can simply be restarted; finished shards are skipped.
--tgt_keep_boundaries forces the first byte and the EOS of every target sentence to
stay explicit (needed when the decoder has to generate the sequence).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compressor import load_autocompleting_tokenizer
from precomputed import (
    FORMAT_VERSION,
    UncompressedBytes,
    finalize,
    run_worker,
    tokenizer_settings,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--finalize", action="store_true", help="write manifest.json and exit")
    parser.add_argument("--src_file")
    parser.add_argument("--tgt_file")
    parser.add_argument("--src_model", help="directory holding the source autocomplete best_model.pt")
    parser.add_argument("--tgt_model")
    parser.add_argument("--src_keep_boundaries", action="store_true")
    parser.add_argument("--tgt_keep_boundaries", action="store_true")
    parser.add_argument(
        "--tgt_no_compression",
        action="store_true",
        help="keep every target byte (uncompressed byte-target control); --tgt_model is then not needed",
    )
    parser.add_argument("--max_bytes", type=int, default=512, help="drop pairs with a longer side")
    parser.add_argument("--shard_lines", type=int, default=500_000)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_shards", type=int, default=None, help="stop after this many shards (for testing)")
    parser.add_argument("--verify", type=int, default=200, help="recheck this many sentences per side of a worker's first shard with the per-sentence tokenizer")
    args = parser.parse_args()

    if args.finalize:
        manifest = finalize(args.out_dir)
        totals = manifest["totals"]
        print(f"{len(manifest['shards'])} shards, {totals['pairs']}/{totals['lines_read']} pairs kept "
              f"(long {totals['dropped_long']}, empty {totals['dropped_empty']}, "
              f"undecodable {totals['dropped_undecodable']})")
        print(f"compression ratio: src {manifest['src_compression_ratio']:.3f}, "
              f"tgt {manifest['tgt_compression_ratio']:.3f}")
        return

    required = ["src_file", "tgt_file", "src_model"]
    if not args.tgt_no_compression:
        required.append("tgt_model")
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name} is required")
    src_tok = load_autocompleting_tokenizer(
        args.src_model, None, 8,
        keep_first=args.src_keep_boundaries, keep_eos=args.src_keep_boundaries,
    )
    if args.tgt_no_compression:
        tgt_tok = UncompressedBytes()
        args.tgt_model = None
    else:
        tgt_tok = load_autocompleting_tokenizer(
            args.tgt_model, None, 8,
            keep_first=args.tgt_keep_boundaries, keep_eos=args.tgt_keep_boundaries,
        )
    settings = {
        "format_version": FORMAT_VERSION,
        "max_bytes": args.max_bytes,
        "shard_lines": args.shard_lines,
        "src_file": str(Path(args.src_file).resolve()),
        "tgt_file": str(Path(args.tgt_file).resolve()),
        "src": tokenizer_settings(src_tok, args.src_model),
        "tgt": tokenizer_settings(tgt_tok, args.tgt_model),
    }
    n = run_worker(
        args.src_file, args.tgt_file, args.out_dir, src_tok, tgt_tok, settings,
        worker=args.worker, num_workers=args.num_workers,
        max_shards=args.max_shards, verify=args.verify,
        log=lambda s: print(s, flush=True),
    )
    print(f"worker {args.worker}/{args.num_workers}: handled {n} shards", flush=True)


if __name__ == "__main__":
    main()
