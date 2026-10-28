#!/bin/bash
# Submit a full replicate experiment on the Cedars Slurm cluster (esplhpccompbio) and combine its results.
# Edit the settings below, then run from the repo root (on the submit node; this only submits jobs):
#   bash evaluation/hpc/submit_replicates.sh
# Each config runs as one Slurm array (one task = one CV fold x one seed, see run_replicates.sh). A final summary job waits
# for every task, writes results/<RUN_NAME>/all_replicates.csv and summary.csv, and deletes the per-task csvs once all are present.

# ---- settings -----------------------------------------------------------------------------------------------------
RUN_NAME=tree_tolerance                                       # results go to results/$RUN_NAME, logs to logs/$RUN_NAME
CONFIGS="local_quant_opt_tree"                                # any configs from run_notebook_comparison.py
TOLERANCES="0 0.1 0.2"                                        # vacuous_tolerance values; each config runs once per value
DATASET=MUX6_QF_500                                           # any dataset name from run_notebook_comparison.py
ITERATIONS=50000                                              # HEROS rule-learning iterations
CV=10                                                         # folds per seed
SEEDS=20                                                      # seeds per config
TIME=4:00:00                                                  # Slurm time limit per task (raise for big datasets / more iterations)
# -------------------------------------------------------------------------------------------------------------------

RUN_DIR=results/$RUN_NAME
LOG_DIR=logs/$RUN_NAME
if [ -e "$RUN_DIR" ]; then
    echo "$RUN_DIR already exists; set a new RUN_NAME so earlier results are not mixed in" >&2
    exit 1
fi
mkdir -p "$RUN_DIR/runs" "$LOG_DIR"

N_TASKS=$((CV * SEEDS))
JOBS=""
N_CONFIGS=0
for CONFIG in $CONFIGS; do
    for TOLERANCE in $TOLERANCES; do
        JOB=$(CV=$CV ITERATIONS=$ITERATIONS TOLERANCE=$TOLERANCE sbatch --parsable --array=0-$((N_TASKS - 1)) -t "$TIME" \
              -o "$LOG_DIR/%x_%A_%a.out" -e "$LOG_DIR/%x_%A_%a.err" \
              evaluation/hpc/run_replicates.sh "$CONFIG" "$DATASET" "$RUN_DIR/runs") || exit 1
        echo "$CONFIG (tolerance $TOLERANCE) on $DATASET, $ITERATIONS iterations: job $JOB ($N_TASKS tasks)"
        JOBS="$JOBS:$JOB"
        N_CONFIGS=$((N_CONFIGS + 1))
    done
done

# Runs after every array task has ended (afterany: also when some fail, so you still get a summary and a warning)
JOB=$(sbatch --parsable -p defq --job-name=heros_summary --mem=4G -t 0:30:00 --dependency=afterany$JOBS \
      -o "$LOG_DIR/summary.out" -e "$LOG_DIR/summary.err" \
      --wrap="source ~/common/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV:-heros} && \
python evaluation/summarize_replicates.py $RUN_DIR/runs --out-dir $RUN_DIR --expected $((N_TASKS * N_CONFIGS)) --remove-runs") || exit 1
echo "summary: job $JOB (starts when all tasks have finished; output in $LOG_DIR/summary.out)"
