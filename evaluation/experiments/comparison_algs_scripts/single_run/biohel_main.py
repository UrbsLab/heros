#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional


# -----------------------------
# Dataset configs
# -----------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MUX6": {
        "train_path": "../datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 6,
    },
    "MUX11": {
        "train_path": "../datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 11,
    },
    "MUX20": {
        "train_path": "../datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 20,
    },
    "GAM_A": {
        "train_path": "../datasets/gametes/A_uni_4add_CV_Train_1.txt",
        "test_path": "../datasets/gametes/A_uni_4add_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_C": {
        "train_path": "../datasets/gametes/C_2way_epistasis_CV_Train_1.txt",
        "test_path": "../datasets/gametes/C_2way_epistasis_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_E": {
        "train_path": "../datasets/gametes/E_uni_4het_CV_Train_1.txt",
        "test_path": "../datasets/gametes/E_uni_4het_CV_Test_1.txt",
        "excluded_columns": ["Model", "InstanceID", "Class"],
        "category": "GAMETES",
        "features": 100,
    },
}
DATASETS: List[str] = list(DATASET_CONFIGS.keys())


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs(base: Path) -> None:
    for d in (base / "jobs", base / "logs", base / "results", base / "figures"):
        d.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s)


def lsf_header(
    job_name: str,
    queue: str,
    time_hhmm: str,
    mem_gb: int,
    cpus: int,
    out_log: Path,
    err_log: Path,
) -> str:
    # LSF uses -W H:MM and rusage[mem=MB]
    mem_mb = mem_gb * 1024
    return f"""#!/bin/bash
#BSUB -J {job_name}
#BSUB -q {queue}
#BSUB -W {time_hhmm}
#BSUB -R "rusage[mem={mem_mb}]"
#BSUB -n {cpus}
#BSUB -o {out_log}
#BSUB -e {err_log}

set -euo pipefail
"""


def slurm_header(
    job_name: str,
    partition: str,
    time_hhmm: str,
    mem_gb: int,
    cpus: int,
    out_log: Path,
    err_log: Path,
) -> str:
    # Slurm uses --time and --mem (MB by default if integer; allow "G" suffix too)
    # Use mem in GB explicitly for clarity.
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time_hhmm}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem_gb}G
#SBATCH --output={out_log}
#SBATCH --error={err_log}
set -euo pipefail
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description="BioHELMain: submit one dataset per job using LSF or Slurm."
    )
    p.add_argument("--project", required=True, help="Project folder name under out_path")
    p.add_argument("--out-path", required=True, help="Base output directory (e.g., /scratch/$USER)")
    p.add_argument(
        "--hpc",
        choices=["lsf", "slurm", "local"],
        default="lsf",
        help="Scheduler type: lsf, slurm, or local (no submission)",
    )

    # Common resources
    p.add_argument("--time", default="20:00", help="Walltime H:MM (LSF -W / Slurm --time)")
    p.add_argument("--mem", type=int, default=4, help="Memory in GB")
    p.add_argument("--cpus", type=int, default=1, help="CPUs (-n for LSF, --cpus-per-task for Slurm)")
    p.add_argument("--python", default="python", help="Python executable on compute nodes")
    p.add_argument("--biohel-bin", default="./biohel", help="BioHEL binary path or command")

    # LSF-specific
    p.add_argument("--queue", default="i2c2_normal", help="LSF queue (-q) [used when --hpc=lsf], SLURM eqivalent to partition")

    # Selection
    p.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS,
        help="Optional subset of datasets to run (default: all)",
    )

    # Environment bootstrap on compute node
    p.add_argument(
        "--env-setup",
        default="",
        help="Optional shell lines for env setup (module load / conda activate), inserted into job script.",
    )

    args = p.parse_args()

    base = Path(args.out_path) / args.project
    ensure_dirs(base)

    job_script_name = "biohel_job.py"

    for ds in args.datasets:
        if ds not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {ds}. Known: {sorted(DATASET_CONFIGS.keys())}")

        uid = f"{args.project}_{ds}_{uuid.uuid4().hex[:6]}"

        job_file = base / "jobs" / f"{uid}.sh"
        out_log = base / "logs" / f"{uid}.out"
        err_log = base / "logs" / f"{uid}.err"

        # Header depends on scheduler
        if args.hpc == "lsf":
            header = lsf_header(
                job_name=uid,
                queue=args.queue,
                time_hhmm=args.time,
                mem_gb=args.mem,
                cpus=args.cpus,
                out_log=out_log,
                err_log=err_log,
            )
            submit_cmd = f"bsub < {job_file}"

        elif args.hpc == "slurm":
            header = slurm_header(
                job_name=uid,
                partition=args.queue,
                time_hhmm=args.time,
                mem_gb=args.mem,
                cpus=args.cpus,
                out_log=out_log,
                err_log=err_log,
            )
            submit_cmd = f"sbatch {job_file}"

        else:
            # local run: no header needed, but keep script for provenance
            header = "#!/bin/bash\nset -euo pipefail\n"
            submit_cmd = ""  # not used

        env_setup = args.env_setup.strip()
        env_block = (env_setup + "\n") if env_setup else ""

        body = f"""{env_block}
{args.python} {job_script_name} \
  --project {args.project} \
  --out-path {args.out_path} \
  --dataset {ds} \
  --biohel-bin {args.biohel_bin}
"""

        write_text(job_file, header + body)
        os.chmod(job_file, 0o750)

        if args.hpc == "local":
            print(f"[LOCAL] {uid}: {ds}")
            # Execute the job script body directly in current environment
            rc = os.system(body)
            if rc != 0:
                print(f"[LOCAL] Job failed (exit code {rc}): {uid}")
        else:
            print(f"[SUBMIT {args.hpc.upper()}] {uid}: {ds}")
            rc = os.system(submit_cmd)
            if rc != 0:
                print(f"[WARN] Submit command failed (exit code {rc}): {submit_cmd}")

    print("\nDone submitting jobs.")


if __name__ == "__main__":
    main()
