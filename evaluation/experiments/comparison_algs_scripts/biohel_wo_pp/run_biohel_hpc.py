#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path


def submit_lsf_cluster_job(
    scratchPath: str,
    logPath: str,
    reserved_memory_gb: int,
    queue: str,
    job_name: str,
    python_cmd: str,
    job_py: str,
    full_data_path: str,
    outputPath: str,
    biohel_bin: str,
    outcome_label: str,
    instanceID_label: str,
    excluded_column: str,
    seed: int,
    verbose: bool,
):
    job_ref = str(time.time())
    full_job_name = f"BIOHEL_{job_name}_seed_{seed}_{job_ref}"
    job_path = os.path.join(scratchPath, full_job_name + "_run.sh")

    with open(job_path, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write(f"#BSUB -q {queue}\n")
        sh.write(f"#BSUB -J {full_job_name}\n")
        sh.write(f'#BSUB -R "rusage[mem={reserved_memory_gb}G]"\n')
        sh.write(f"#BSUB -M {reserved_memory_gb}GB\n")
        sh.write(f"#BSUB -o {os.path.join(logPath, full_job_name)}.o\n")
        sh.write(f"#BSUB -e {os.path.join(logPath, full_job_name)}.e\n")
        sh.write(
            f"{python_cmd} {job_py}"
            f" --d {full_data_path}"
            f" --o {outputPath}"
            f" --biohel {biohel_bin}"
            f" --ol {outcome_label}"
            f" --il {instanceID_label}"
            f" --el {excluded_column}"
            f" --rs {seed}"
            + (" --v" if verbose else "")
            + "\n"
        )

    os.system("bsub < " + job_path)


def submit_slurm_cluster_job(
    scratchPath: str,
    logPath: str,
    reserved_memory_gb: int,
    queue: str,
    job_name: str,
    python_cmd: str,
    job_py: str,
    full_data_path: str,
    outputPath: str,
    biohel_bin: str,
    outcome_label: str,
    instanceID_label: str,
    excluded_column: str,
    seed: int,
    verbose: bool,
):
    job_ref = str(time.time())
    full_job_name = f"BIOHEL_{job_name}_seed_{seed}_{job_ref}"
    job_path = os.path.join(scratchPath, full_job_name + "_run.sh")

    with open(job_path, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write(f"#SBATCH -p {queue}\n")
        sh.write(f"#SBATCH --job-name={full_job_name}\n")
        sh.write(f"#SBATCH --mem={reserved_memory_gb}G\n")
        sh.write(f"#SBATCH -o {os.path.join(logPath, full_job_name)}.o\n")
        sh.write(f"#SBATCH -e {os.path.join(logPath, full_job_name)}.e\n")
        sh.write(
            f"srun {python_cmd} {job_py}"
            f" --d {full_data_path}"
            f" --o {outputPath}"
            f" --biohel {biohel_bin}"
            f" --ol {outcome_label}"
            f" --il {instanceID_label}"
            f" --el {excluded_column}"
            f" --rs {seed}"
            + (" --v" if verbose else "")
            + "\n"
        )

    os.system("sbatch " + job_path)


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL CV + random seeds job submission (LSF/Slurm/Local)")

    # Script parameters
    parser.add_argument("--d", dest="datafolder", type=str, required=True,
                        help="Path to data folder containing dataset subfolders.")
    parser.add_argument("--w", dest="writepath", type=str, required=True,
                        help="Path where outputs/logs/scratch will be stored.")
    parser.add_argument("--o", dest="outputfolder", type=str, default="myAnalysis",
                        help="Unique folder name for this analysis.")

    # Dataset parameters (match HEROS style)
    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    # Experiment parameters
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    parser.add_argument("--rs", dest="random_state", type=int, default=42,
                        help="Used only when random_seeds==1, otherwise seeds are 0..r-1")

    # BioHEL parameters
    parser.add_argument("--biohel", dest="biohel_bin", type=str, required=True,
                        help="Path to BioHEL binary (or command on PATH).")

    # HPC parameters
    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF",
                        help="Cluster type: LSF, SLURM, or LOCAL")
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=4,
                        help="Reserved memory for job in GB")
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal",
                        help="LSF queue or Slurm partition")
    parser.add_argument("--check", dest="check", action="store_true",
                        help="Check and report incomplete jobs")
    parser.add_argument("--resub", dest="resubmit", action="store_true",
                        help="Resubmit incomplete jobs (only relevant with --check)")
    parser.add_argument("--python", dest="python_cmd", type=str, default="python",
                        help="Python executable to use on nodes.")
    parser.add_argument("--job", dest="job_script", type=str, default="job_biohel_hpc.py",
                        help="Worker job script filename.")

    parser.add_argument("--v", dest="verbose", action="store_true",
                        help="Verbose flag")

    options = parser.parse_args(argv[1:])

    datafolder = options.datafolder
    writepath = options.writepath
    outputfolder = options.outputfolder

    outcome_label = options.outcome_label
    instanceID_label = options.instanceID_label
    excluded_column = options.excluded_column

    cv_partitions = options.cv_partitions
    random_seeds = options.random_seeds
    random_state = options.random_state

    biohel_bin = options.biohel_bin

    run_cluster = options.run_cluster.upper()
    reserved_memory = options.reserved_memory
    queue = options.queue
    check = options.check
    resubmit = options.resubmit
    python_cmd = options.python_cmd
    job_script = options.job_script
    verbose = options.verbose

    algorithm = "BioHEL"

    # Folder management (mirror HEROS)
    Path(writepath).mkdir(parents=True, exist_ok=True)
    base_output_root = Path(writepath) / "output"
    base_output_root.mkdir(exist_ok=True)

    scratchPath = Path(writepath) / "scratch"
    scratchPath.mkdir(exist_ok=True)

    logPath = Path(writepath) / "logs"
    logPath.mkdir(exist_ok=True)

    base_output_path_0 = base_output_root / f"{algorithm}_{outputfolder}"
    base_output_path_0.mkdir(exist_ok=True)

    jobCount = 0
    missing_count = 0

    # For each dataset subfolder (entry)
    for entry in os.listdir(datafolder):
        entry_path = os.path.join(datafolder, entry)
        if not os.path.isdir(entry_path):
            continue

        datapath = entry_path
        base_output_path_1 = base_output_path_0 / entry
        base_output_path_1.mkdir(exist_ok=True)

        for i in range(0, random_seeds):
            if random_seeds > 1:
                target_seed = i
            else:
                target_seed = random_state

            base_output_path_2 = base_output_path_1 / f"seed_{i}"
            base_output_path_2.mkdir(exist_ok=True)

            for j in range(1, cv_partitions + 1):
                full_data_path = os.path.join(datapath, f"{entry}_CV_Train_{j}.txt")
                full_data_name = f"{entry}_CV_Train_{j}"
                outputPath = base_output_path_2 / f"cv_{j}"
                outputPath.mkdir(exist_ok=True)

                # Sentinel: consider job complete if result_row.csv exists
                sentinel = outputPath / "result_row.csv"

                if check:
                    if not sentinel.exists():
                        print("Missing: " + str(outputPath))
                        missing_count += 1
                        if resubmit:
                            # fall through to submission
                            pass
                        else:
                            continue
                    else:
                        continue

                # Submit or run locally
                if run_cluster == "LSF":
                    submit_lsf_cluster_job(
                        str(scratchPath),
                        str(logPath),
                        reserved_memory,
                        queue,
                        full_data_name,
                        python_cmd,
                        job_script,
                        full_data_path,
                        str(outputPath),
                        biohel_bin,
                        outcome_label,
                        instanceID_label,
                        excluded_column,
                        target_seed,
                        verbose,
                    )
                    jobCount += 1

                elif run_cluster == "SLURM":
                    submit_slurm_cluster_job(
                        str(scratchPath),
                        str(logPath),
                        reserved_memory,
                        queue,
                        full_data_name,
                        python_cmd,
                        job_script,
                        full_data_path,
                        str(outputPath),
                        biohel_bin,
                        outcome_label,
                        instanceID_label,
                        excluded_column,
                        target_seed,
                        verbose,
                    )
                    jobCount += 1

                elif run_cluster == "LOCAL":
                    cmd = (
                        f"{python_cmd} {job_script}"
                        f" --d {full_data_path}"
                        f" --o {outputPath}"
                        f" --biohel {biohel_bin}"
                        f" --ol {outcome_label}"
                        f" --il {instanceID_label}"
                        f" --el {excluded_column}"
                        f" --rs {target_seed}"
                        + (" --v" if verbose else "")
                    )
                    rc = os.system(cmd)
                    if rc != 0:
                        print(f"LOCAL run failed: {cmd}")
                    jobCount += 1

                else:
                    print("ERROR: Cluster type not found. Use LSF, SLURM, or LOCAL.")
                    return 1

    print(str(jobCount) + " jobs submitted successfully")
    if check:
        print(str(missing_count) + " jobs incomplete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
