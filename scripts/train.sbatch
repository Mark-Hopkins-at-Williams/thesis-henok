#!/bin/sh
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 60-00:00
#SBATCH -p dl
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --gres=gpu:1
# Train, or resume, one run. Submit from the repository root:
#   sbatch -J en-ru-ac16 scripts/train.sbatch configs/en-ru-ac16.json
#   sbatch -J en-ru-ac16 scripts/train.sbatch configs/en-ru-ac16.json experiments/en-ru-ac16-v0
# The second form resumes from the newest checkpoint.pt in that experiment directory.
#
# Slurm here does not tell the two RTX 6000 Ada cards (GPUs 0,1) from the four A6000s
# (GPUs 2-5); it hands out the lowest free index first. So submit the slowest run (the
# uncompressed byte target) first if you want it on an Ada.
. /mnt/storage/general/anaconda3/etc/profile.d/conda.sh
conda activate hopkins-py
cd "$SLURM_SUBMIT_DIR"
CONFIG=$1
RESUME=$2
echo "job $SLURM_JOB_ID on $(hostname); GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=name --format=csv,noheader)"
if [ -n "$RESUME" ]; then
  python finetune.py --resume "$RESUME"
else
  python finetune.py --config "$CONFIG"
fi
