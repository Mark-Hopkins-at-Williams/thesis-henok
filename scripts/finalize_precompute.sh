#!/bin/sh
# Finalize both precomputed train sets and check that they cover the whole input.
# Run from the repository root with the hopkins-py environment active, after every
# scripts/precompute_compressed.py worker has finished. Exits non-zero on any problem.
set -e
D=/mnt/storage/hopkins/data/wmt19/ru-en

if pgrep -f "precompute_compressed.py.*--worker" > /dev/null; then
  echo "precompute workers are still running"
  exit 1
fi

EXPECTED=$(wc -l < "$D/organized/train.en")
for dir in compressed compressed_byte; do
  python scripts/precompute_compressed.py --finalize --out_dir "$D/$dir/train"
  # a worker that died at the tail would leave contiguous but incomplete shards
  python - "$D/$dir/train/manifest.json" "$EXPECTED" <<'EOF'
import json, sys
totals = json.load(open(sys.argv[1]))["totals"]
if totals["lines_read"] != int(sys.argv[2]):
    sys.exit(f"{sys.argv[1]}: shards cover {totals['lines_read']} lines, input has {sys.argv[2]}")
print(f"{sys.argv[1]}: covers all {totals['lines_read']} lines, {totals['pairs']} pairs kept")
EOF
done
