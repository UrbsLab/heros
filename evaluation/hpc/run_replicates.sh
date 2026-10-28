#!/bin/bash
# Replicate HEROS runs on the Cedars Slurm cluster (esplhpccompbio): one array task = one config x one CV fold x one random seed.
# Array index i -> fold = i % CV, seed = i / CV, so 10 folds x 20 seeds = indexes 0-199.
# For a full experiment (all configs, combined results in one folder), use submit_replicates.sh instead.
# To submit by hand, from the repo root, once per config:
#   mkdir -p logs
#   sbatch --array=0-199 evaluation/hpc/run_replicates.sh baseline
#   sbatch --array=0-199 evaluation/hpc/run_replicates.sh local_quant_opt_tree
#   sbatch --array=0-199 evaluation/hpc/run_replicates.sh local_quant_opt_gini
# Optional 2nd/3rd args: dataset (default MUX6_QF_500) and results folder. Set CV=<k> to change the fold count,
# ITERATIONS=<n> to change the HEROS iterations (default 50000) and TOLERANCE=<t> to set vacuous_tolerance (default 0).
#SBATCH -p defq
#SBATCH --job-name=heros_rep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

CONFIG=${1:?usage: sbatch --array=0-199 run_replicates.sh <baseline|local_quant_opt_tree|> [dataset] [results_dir]}
DATASET=${2:-MUX6_QF_500}
RESULTS_DIR=${3:-results/replicates/$DATASET}
CV=${CV:-10}
ITERATIONS=${ITERATIONS:-50000}
TOLERANCE=${TOLERANCE:-0}
TASK=${SLURM_ARRAY_TASK_ID:?submit with --array}
FOLD=$((TASK % CV))
SEED=$((TASK / CV))
CONDA_ENV=${CONDA_ENV:-heros}

# Keep numpy/sklearn to the cores Slurm gave us
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source ~/common/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

cd "$SLURM_SUBMIT_DIR"
python evaluation/run_notebook_comparison.py "$DATASET" --configs "$CONFIG" --cv "$CV" --fold "$FOLD" --seed "$SEED" --iterations "$ITERATIONS" \
    --vacuous_tolerance "$TOLERANCE" --out "$RESULTS_DIR/${CONFIG}_tol${TOLERANCE}_fold${FOLD}_seed${SEED}.csv"
