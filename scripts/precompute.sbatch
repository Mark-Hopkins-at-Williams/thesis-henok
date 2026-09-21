#!/bin/sh
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -p dl
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
# One precompute worker for the train split:
#   sbatch -J pre-ac-0 scripts/precompute.sbatch KIND WORKER NUM_WORKERS
# KIND is "ac" (autocomplete-compressed Russian target) or "byte" (uncompressed byte
# target, the control). Workers split the 500k-line shards; a finished shard is skipped,
# so a worker can simply be resubmitted after a failure.
. /mnt/storage/general/anaconda3/etc/profile.d/conda.sh
conda activate hopkins-py
cd "$SLURM_SUBMIT_DIR"
KIND=$1
WORKER=$2
NUM_WORKERS=$3
D=/mnt/storage/hopkins/data/wmt19/ru-en
if [ "$KIND" = ac ]; then
  TARGET="--tgt_model experiments/ru-autocomplete-v1 --tgt_keep_boundaries --out_dir $D/compressed/train"
else
  TARGET="--tgt_no_compression --out_dir $D/compressed_byte/train"
fi
python scripts/precompute_compressed.py \
  --src_file "$D/organized/train.en" --tgt_file "$D/organized/train.ru" \
  --src_model experiments/autocomplete-v1 $TARGET \
  --worker "$WORKER" --num_workers "$NUM_WORKERS"
