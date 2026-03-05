#!/usr/bin/env python3
import sys
import os
import time
import argparse
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="Submit RIPPER summary job (HEROS-like outputs)")

    parser.add_argument("--w", dest="writepath", type=str, required=True)
    parser.add_argument("--o", dest="outputfolder", type=str, required=True)

    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)

    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF", choices=["LSF", "SLURM"])
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=1)
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal")

    parser.add_argument("--plots", dest="plots", action="store_true",
                        help="Generate boxplots (testing accuracy, rule count, coverage).")

    opts = parser.parse_args(argv[1:])

    writepath = Path(opts.writepath)
    outputfolder = opts.outputfolder

    cv_partitions = int(opts.cv_partitions)
    random_seeds = int(opts.random_seeds)

    run_cluster = opts.run_cluster
    reserved_memory = int(opts.reserved_memory)
    queue = opts.queue
    plots = bool(opts.plots)

    algorithm = "RIPPER"

    base_output_path_0 = writepath / "output" / f"{algorithm}_{outputfolder}"
    if not base_output_path_0.exists():
        raise FileNotFoundError(f"Expected output folder not found: {base_output_path_0}")

    scratchPath = writepath / "scratch"
    scratchPath.mkdir(parents=True, exist_ok=True)

    logPath = writepath / "logs"
    logPath.mkdir(parents=True, exist_ok=True)

    job_ref = str(time.time())
    job_name = f"RIPPER_summary_{outputfolder}_{job_ref}"
    job_path = scratchPath / f"{job_name}_run.sh"

    cmd = (
        f"python job_ripper_sum_hpc.py"
        f" --o {base_output_path_0}"
        f" --cv {cv_partitions}"
        f" --r {random_seeds}"
    )
    if plots:
        cmd += " --plots"

    with open(job_path, "w") as sh:
        sh.write("#!/bin/bash\n")
        if run_cluster == "LSF":
            sh.write(f"#BSUB -q {queue}\n")
            sh.write(f"#BSUB -J {job_name}\n")
            sh.write(f'#BSUB -R "rusage[mem={reserved_memory}G]"\n')
            sh.write(f"#BSUB -M {reserved_memory}GB\n")
            sh.write(f"#BSUB -o {logPath}/{job_name}.o\n")
            sh.write(f"#BSUB -e {logPath}/{job_name}.e\n")
            sh.write(cmd + "\n")
        else:
            sh.write(f"#SBATCH -p {queue}\n")
            sh.write(f"#SBATCH --job-name={job_name}\n")
            sh.write(f"#SBATCH --mem={reserved_memory}G\n")
            sh.write(f"#SBATCH -o {logPath}/{job_name}.o\n")
            sh.write(f"#SBATCH -e {logPath}/{job_name}.e\n")
            sh.write("srun " + cmd + "\n")
            
    if run_cluster == "LSF":
        os.system(f"bsub < {job_path}")
    else:
        os.system(f"sbatch {job_path}")

    print("1 job submitted successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
