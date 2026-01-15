#!/bin/sh
#SBATCH -c 1
#SBATCH -t 3-12:00
#SBATCH -p dl
#SBATCH -o logs/log_%j.out
#SBATCH -e logs/log_%j.err
#SBATCH --gres=gpu:1

python est_v3.mark.py --config configs/est2.json
#python corrupt_gold_comparisons_w_attn.py

