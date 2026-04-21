#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
import argparse


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL summary submission (LSF/Slurm/Local)")

    # Script parameters
    parser.add_argument("--w", dest="writepath", type=str, required=True,
                        help="Path where outputs/logs/scratch live")
    parser.add_argument("--o", dest="outputfolder", type=str, required=True,
                        help="Unique folder name for this analysis")

    # Experiment parameters
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)

    # HPC parameters
    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF",
                        help="Cluster type: LSF, SLURM, or LOCAL")
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=4,
                        help="Reserved memory (GB)")
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal",
                        help="LSF queue or Slurm partition")
    parser.add_argument("--python", dest="python_cmd", type=str, default="python",
                        help="Python executable")
    parser.add_argument("--job", dest="job_script", type=str, default="job_biohel_sum_hpc.py",
                        help="Summary worker script")

    parser.add_argument("--plots", dest="plots", action="store_true",
                        help="Generate plots in summary")

    options = parser.parse_args(argv[1:])

    writepath = options.writepath
    outputfolder = options.outputfolder
    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds

    run_cluster = options.run_cluster.upper()
    reserved_memory = options.reserved_memory
    queue = options.queue
    python_cmd = options.python_cmd
    job_script = options.job_script
    plots = options.plots

    algorithm = "BioHEL"

    # Folder management (mirror HEROS)
    if not os.path.exists(writepath):
        os.mkdir(writepath)

    base_output_path = os.path.join(writepath, "output")
    if not os.path.exists(base_output_path):
        os.mkdir(base_output_path)

    scratchPath = os.path.join(writepath, "scratch")
    if not os.path.exists(scratchPath):
        os.mkdir(scratchPath)

    logPath = os.path.join(writepath, "logs")
    if not os.path.exists(logPath):
        os.mkdir(logPath)

    base_output_path_0 = os.path.join(base_output_path, f"{algorithm}_{outputfolder}")
    if not os.path.exists(base_output_path_0):
        os.mkdir(base_output_path_0)

    if run_cluster == "LSF":
        submit_lsf_cluster_job(
            scratchPath, logPath, reserved_memory, queue,
            base_output_path_0, cv_partitions, random_seeds,
            outputfolder, python_cmd, job_script, plots
        )
    elif run_cluster == "SLURM":
        submit_slurm_cluster_job(
            scratchPath, logPath, reserved_memory, queue,
            base_output_path_0, cv_partitions, random_seeds,
            outputfolder, python_cmd, job_script, plots
        )
    elif run_cluster == "LOCAL":
        cmd = (
            f"{python_cmd} {job_script}"
            f" --o {base_output_path_0}"
            f" --cv {cv_partitions}"
            f" --r {random_seeds}"
            + (" --plots" if plots else "")
        )
        rc = os.system(cmd)
        if rc != 0:
            print("LOCAL summary failed.")
            return 1
    else:
        print("ERROR: Cluster type not found")
        return 1

    print("1 job submitted successfully")
    return 0


def submit_lsf_cluster_job(
    scratchPath, logPath, reserved_memory, queue,
    base_output_path_0, cv_partitions, random_seeds,
    outputfolder, python_cmd, job_script, plots
):
    job_ref = str(time.time())
    job_name = f"BIOHEL_summary_{outputfolder}_{job_ref}"
    job_path = os.path.join(scratchPath, job_name + "_run.sh")

    with open(job_path, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write("#BSUB -q " + queue + "\n")
        sh.write("#BSUB -J " + job_name + "\n")
        sh.write("#BSUB -n 1\n")
        sh.write('#BSUB -R "rusage[mem=' + str(reserved_memory) + 'G]"\n')
        sh.write("#BSUB -M " + str(reserved_memory) + "GB\n")
        sh.write("#BSUB -o " + os.path.join(logPath, job_name) + ".o\n")
        sh.write("#BSUB -e " + os.path.join(logPath, job_name) + ".e\n")

        cmd = (
            f"{python_cmd} {job_script}"
            f" --o {base_output_path_0}"
            f" --cv {cv_partitions}"
            f" --r {random_seeds}"
            + (" --plots" if plots else "")
        )
        sh.write(cmd + "\n")

    os.system("bsub < " + job_path)


def submit_slurm_cluster_job(
    scratchPath, logPath, reserved_memory, queue,
    base_output_path_0, cv_partitions, random_seeds,
    outputfolder, python_cmd, job_script, plots
):
    job_ref = str(time.time())
    job_name = f"BIOHEL_summary_{outputfolder}_{job_ref}"
    job_path = os.path.join(scratchPath, job_name + "_run.sh")

    with open(job_path, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write("#SBATCH -p " + queue + "\n")
        sh.write("#SBATCH --job-name=" + job_name + "\n")
        sh.write("#SBATCH --mem=" + str(reserved_memory) + "G\n")
        sh.write("#SBATCH -o " + os.path.join(logPath, job_name) + ".o\n")
        sh.write("#SBATCH -e " + os.path.join(logPath, job_name) + ".e\n")

        cmd = (
            f"srun {python_cmd} {job_script}"
            f" --o {base_output_path_0}"
            f" --cv {cv_partitions}"
            f" --r {random_seeds}"
            + (" --plots" if plots else "")
        )
        sh.write(cmd + "\n")

    os.system("sbatch " + job_path)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
