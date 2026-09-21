#!/bin/sh
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -p dl
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
# CPU-only gate, submitted with --dependency=afterany on the precompute workers: it runs
# once they have all ended, finalizes both train sets and checks they cover the whole
# input. The training jobs depend on it (afterok), so they start only if the data is
# complete, and none of this needs anyone to be logged in.
. /mnt/storage/general/anaconda3/etc/profile.d/conda.sh
conda activate hopkins-py
cd "$SLURM_SUBMIT_DIR"
echo "precompute workers have ended $(date)"
./scripts/finalize_precompute.sh
echo "data complete $(date)"
