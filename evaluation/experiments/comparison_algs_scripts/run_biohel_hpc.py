#!/usr/bin/env python3
import sys
import os
import time
import argparse
from pathlib import Path


def main(argv):
    parser = argparse.ArgumentParser(description="Submit BioHEL jobs across datasets × seeds × CV folds")

    # Paths
    parser.add_argument("--d", dest="datafolder", type=str, required=True,
                        help="Path to data folder containing dataset subfolders.")
    parser.add_argument("--w", dest="writepath", type=str, required=True,
                        help="Base folder where output/, scratch/, logs/ will be written.")
    parser.add_argument("--o", dest="outputfolder", type=str, required=True,
                        help="Unique analysis name (used as BioHEL_<outputfolder>).")

    # Dataset columns
    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    # Experiment loop
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    parser.add_argument("--rs0", dest="seed_start", type=int, default=0,
                        help="Seed index start (default 0). Useful for partial runs.")

    # Binaries
    parser.add_argument("--biohel_bin", dest="biohel_bin", type=str, default="./biohel",
                        help="BioHEL binary name or full path.")
    parser.add_argument("--enable_rpe", dest="enable_rpe", action="store_true",
                        help="Enable BioHEL-RPE postprocessing stage.")
    parser.add_argument("--postprocess_bin", dest="postprocess_bin", type=str, default="./postprocess",
                        help="RPE binary name (postprocess or postprocessing) or full path.")
    parser.add_argument("--postprocess_conf", dest="postprocess_conf", type=str, default="postprocess.conf",
                        help="Path to RPE config file (e.g., test-pp.conf).")

    # HPC
    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF", choices=["LSF", "SLURM"])
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=4)
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal")

    # Check/resubmit
    parser.add_argument("--check", dest="check", action="store_true",
                        help="If set: do not submit new jobs, only report missing folds (result_row.csv).")
    parser.add_argument("--resub", dest="resubmit", action="store_true",
                        help="If set with --check: resubmit missing folds.")

    parser.add_argument("--v", dest="verbose", action="store_true")

    opts = parser.parse_args(argv[1:])

    datafolder = Path(opts.datafolder)
    writepath = Path(opts.writepath)
    outputfolder = opts.outputfolder

    outcome_label = opts.outcome_label
    instanceID_label = opts.instanceID_label
    excluded_column = opts.excluded_column

    cv_partitions = int(opts.cv_partitions)
    random_seeds = int(opts.random_seeds)
    seed_start = int(opts.seed_start)

    biohel_bin = opts.biohel_bin
    enable_rpe = bool(opts.enable_rpe)
    postprocess_bin = opts.postprocess_bin
    postprocess_conf = opts.postprocess_conf

    run_cluster = opts.run_cluster
    reserved_memory = int(opts.reserved_memory)
    queue = opts.queue

    check = bool(opts.check)
    resubmit = bool(opts.resubmit)
    verbose = bool(opts.verbose)

    algorithm = "BioHEL"

    # Folder management
    writepath.mkdir(parents=True, exist_ok=True)

    base_output_root = writepath / "output"
    base_output_root.mkdir(exist_ok=True)

    scratchPath = writepath / "scratch"
    scratchPath.mkdir(exist_ok=True)

    logPath = writepath / "logs"
    logPath.mkdir(exist_ok=True)

    base_output_path_0 = base_output_root / f"{algorithm}_{outputfolder}"
    base_output_path_0.mkdir(exist_ok=True)

    # Loop and submit
    jobCount = 0
    missing_count = 0

    for entry in sorted(os.listdir(datafolder)):
        dataset_dir = datafolder / entry
        if not dataset_dir.is_dir():
            continue

        dataset_out_dir = base_output_path_0 / entry
        dataset_out_dir.mkdir(exist_ok=True)

        # Submit for each seed and CV fold
        for seed_idx in range(seed_start, random_seeds):
            seed_out_dir = dataset_out_dir / f"seed_{seed_idx}"
            seed_out_dir.mkdir(exist_ok=True)

            for cv_idx in range(1, cv_partitions + 1):
                train_path = dataset_dir / f"{entry}_CV_Train_{cv_idx}.txt"
                if not train_path.exists():
                    # Skip silently if not a CV dataset folder
                    continue

                cv_out_dir = seed_out_dir / f"cv_{cv_idx}"
                cv_out_dir.mkdir(exist_ok=True)

                sentinel = cv_out_dir / "result_row.csv"

                if check:
                    if not sentinel.exists():
                        missing_count += 1
                        print(f"Missing: {cv_out_dir}")
                        if resubmit:
                            _submit_one(
                                run_cluster=run_cluster,
                                scratchPath=scratchPath,
                                logPath=logPath,
                                reserved_memory=reserved_memory,
                                queue=queue,
                                job_label=f"{entry}_seed_{seed_idx}_cv_{cv_idx}",
                                train_path=str(train_path),
                                output_dir=str(cv_out_dir),
                                outcome_label=outcome_label,
                                instanceID_label=instanceID_label,
                                excluded_column=excluded_column,
                                seed=seed_idx,
                                biohel_bin=biohel_bin,
                                enable_rpe=enable_rpe,
                                postprocess_bin=postprocess_bin,
                                postprocess_conf=postprocess_conf,
                                verbose=verbose,
                            )
                            jobCount += 1
                    continue

                # Normal run: submit if not complete
                if not sentinel.exists():
                    _submit_one(
                        run_cluster=run_cluster,
                        scratchPath=scratchPath,
                        logPath=logPath,
                        reserved_memory=reserved_memory,
                        queue=queue,
                        job_label=f"{entry}_seed_{seed_idx}_cv_{cv_idx}",
                        train_path=str(train_path),
                        output_dir=str(cv_out_dir),
                        outcome_label=outcome_label,
                        instanceID_label=instanceID_label,
                        excluded_column=excluded_column,
                        seed=seed_idx,
                        biohel_bin=biohel_bin,
                        enable_rpe=enable_rpe,
                        postprocess_bin=postprocess_bin,
                        postprocess_conf=postprocess_conf,
                        verbose=verbose,
                    )
                    jobCount += 1
                else:
                    if verbose:
                        print(f"Exists: {sentinel}")

    if check:
        print(f"{missing_count} folds incomplete.")
        if resubmit:
            print(f"{jobCount} jobs resubmitted.")
    else:
        print(f"{jobCount} jobs submitted.")


def _submit_one(
    run_cluster: str,
    scratchPath: Path,
    logPath: Path,
    reserved_memory: int,
    queue: str,
    job_label: str,
    train_path: str,
    output_dir: str,
    outcome_label: str,
    instanceID_label: str,
    excluded_column: str,
    seed: int,
    biohel_bin: str,
    enable_rpe: bool,
    postprocess_bin: str,
    postprocess_conf: str,
    verbose: bool,
):
    job_ref = str(time.time())
    job_name = f"BIOHEL_{job_label}_{job_ref}"
    job_path = scratchPath / f"{job_name}_run.sh"

    cmd = (
        f"python job_biohel_hpc.py"
        f" --d {train_path}"
        f" --o {output_dir}"
        f" --biohel {biohel_bin}"
        f" --ol {outcome_label}"
        f" --il {instanceID_label}"
        f" --el {excluded_column}"
        f" --rs {seed}"
    )

    if enable_rpe:
        cmd += (
            f" --enable_rpe"
            f" --postprocess_bin {postprocess_bin}"
            f" --postprocess_conf {postprocess_conf}"
        )

    if verbose:
        cmd += " --v"

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


if __name__ == "__main__":
    sys.exit(main(sys.argv))
