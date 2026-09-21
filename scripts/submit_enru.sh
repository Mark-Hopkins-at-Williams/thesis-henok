#!/bin/sh
# Submit the whole en-ru pipeline as one Slurm dependency chain, so it completes by
# itself even if nobody is logged in. Run from the repository root:
#
#   precompute workers (4 for the compressed target, 2 for the byte control)
#     -> gate: finalize both train sets and check they cover the whole input
#       -> the five training runs (cancelled if the gate fails)
#
# Safe to re-run after a failure: finished shards are skipped. The byte control is
# submitted first among the runs: Slurm hands out the lowest free GPU index, so the
# slowest run should land on an Ada card (GPUs 0 and 1).
set -e
PRE=""
for w in 0 1 2 3; do
  PRE="$PRE:$(sbatch --parsable -J pre-ac-$w scripts/precompute.sbatch ac $w 4)"
done
for w in 0 1; do
  PRE="$PRE:$(sbatch --parsable -J pre-byte-$w scripts/precompute.sbatch byte $w 2)"
done
GATE=$(sbatch --parsable --dependency=afterany"$PRE" -J precompute-gate scripts/precompute_gate.sbatch)
echo "precompute jobs$PRE, gate job $GATE"
for run in en-ru-byte en-ru-ac16 en-ru-ac8 en-ru-ac32 en-ru-ac16-seed2; do
  sbatch --dependency=afterok:"$GATE" --kill-on-invalid-dep=yes -J "$run" \
    scripts/train.sbatch "configs/$run.json"
done
squeue -u "$USER" -o "%i %j %T %r"
