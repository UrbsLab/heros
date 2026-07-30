#!/usr/bin/env python3
# run_heros_alt_table2.0.py
#
# Submits a single cluster job that generates a combined HEROS table
# (with per-family significance) restricted to specific experiment groups.
#
# Example:
# python run_heros_alt_table2.0.py \
#   --w /project/kamoun_shared/gabe/I2C2-Documentation \
#   --o output \
#   --groups mux_cv_default,mux_cv_default_tree_init,gametes_cv_default,gametes_cv_default_tree_init \
#   --baseline-map multiplexer:mux_cv_default,gametes:gametes_cv_default \
#   --alpha 0.05 \
#   --rc LSF --rm 8 --q i2c2_normal \
#   --outcsv HEROS_Combined_Table.csv

import sys
import os
import time
import argparse
from pathlib import Path


def parse_args(argv):
    p = argparse.ArgumentParser(description="Submit HEROS combined-table job to cluster (with significance).")

    # Paths
    p.add_argument("--w", dest="writepath", type=str, required=True,
                   help="Path containing this script AND the job file (e.g., /project/.../I2C2-Documentation).")
    p.add_argument("--o", dest="outputfolder", type=str, required=True,
                   help="Folder under --w that holds experiment outputs. Usually: output")

    # Which experiment groups to include (WITHOUT 'HEROS_' prefix)
    p.add_argument("--groups", dest="groups", type=str, required=True,
                   help="Comma-separated group names under output/ (e.g., mux_cv_default,gametes_cv_default_tree_init)")

    # Output CSV name (written into output root)
    p.add_argument("--outcsv", dest="outcsv", type=str, default="HEROS_Combined_Table.csv",
                   help="Name of the combined output CSV file.")

    # Significance controls
    p.add_argument("--baseline-map", dest="baseline_map", type=str,
                   default="multiplexer:mux_cv_default,gametes:gametes_cv_default",
                   help="Per-family baselines for significance. Format: 'multiplexer:<group>,gametes:<group>'.")
    p.add_argument("--alpha", dest="alpha", type=float, default=0.05,
                   help="Significance threshold p-value (default 0.05).")

    # HPC params
    p.add_argument("--rc", dest="run_cluster", type=str, default="SLURM", choices=["LSF", "SLURM"])
    p.add_argument("--rm", dest="reserved_memory", type=int, default=8, help="GB")
    p.add_argument("--q", dest="queue", type=str, default="defq")

    return p.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    writepath = Path(args.writepath).resolve()
    output_root = (writepath / args.outputfolder).resolve()
    scratchPath = (writepath / "scratch").resolve()
    logPath = (writepath / "logs").resolve()

    scratchPath.mkdir(parents=True, exist_ok=True)
    logPath.mkdir(parents=True, exist_ok=True)

    # Job python file (must exist in writepath)
    job_py = "job_heros_table.py"
    #if not job_py.exists():
    #    print(f"[ERR] job file not found: {job_py}")
    #    return 2

    groups_csv = ",".join([g.strip() for g in args.groups.split(",") if g.strip()])
    if not groups_csv:
        print("[ERR] --groups must contain at least one group name.")
        return 2

    job_cmd = (
        f"python {job_py} "
        f"--o {output_root} "
        f"--groups {groups_csv} "
        f"--outcsv {args.outcsv} "
        f"--baseline-map {args.baseline_map} "
        f"--alpha {args.alpha} "
    )

    job_ref = str(int(time.time()))
    job_name = f"HEROS_table_{output_root.name}_{job_ref}"
    job_sh = scratchPath / f"{job_name}_run.sh"

    with open(job_sh, "w") as f:
        f.write("#!/bin/bash\n")
        if args.run_cluster.upper() == "LSF":
            f.write(f"#BSUB -q {args.queue}\n")
            f.write(f"#BSUB -J {job_name}\n")
            f.write(f"#BSUB -R \"rusage[mem={args.reserved_memory}G]\"\n")
            f.write(f"#BSUB -M {args.reserved_memory}GB\n")
            f.write(f"#BSUB -o {logPath}/{job_name}.o\n")
            f.write(f"#BSUB -e {logPath}/{job_name}.e\n")
            f.write(job_cmd + "\n")
        else:
            f.write(f"#SBATCH -p {args.queue}\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write(f"#SBATCH --mem={args.reserved_memory}G\n")
            f.write(f"#SBATCH -o {logPath}/{job_name}.o\n")
            f.write(f"#SBATCH -e {logPath}/{job_name}.e\n")
            f.write(f"srun {job_cmd}\n")

    os.chmod(job_sh, 0o755)

    if args.run_cluster.upper() == "LSF":
        os.system(f"bsub < {job_sh}")
        print(f"[SUBMITTED][LSF] {job_sh}")
    else:
        os.system(f"sbatch {job_sh}")
        print(f"[SUBMITTED][SLURM] {job_sh}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
