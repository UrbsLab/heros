# I2C2-Documentation/run_heros_alt_sum.py
#
# Submits the summary aggregation as a single cluster job (LSF or SLURM).
# - Minimal args: --w (writepath), --o (outputfolder), --cv, --r
# - Writes logs into writepath/logs
# - Calls: python I2C2-Documentation/job_heros_alt_sum.py --o <output_root> --cv ... --r ...

import os
import sys
import time
import argparse

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def submit_lsf(scratch_path, log_path, reserved_memory, queue, output_root, cv_partitions, random_seeds, outputfolder,
               outcome_label, instanceID_label, excluded_column):
    job_ref = str(time.time())
    job_name = f"HEROS_summary_{outputfolder}_{job_ref}"
    job_script = os.path.join(scratch_path, f"{job_name}_run.sh")

    with open(job_script, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write(f"#BSUB -q {queue}\n")
        sh.write(f"#BSUB -J {job_name}\n")
        sh.write(f"#BSUB -R \"rusage[mem={reserved_memory}G]\"\n")
        sh.write(f"#BSUB -M {reserved_memory}GB\n")
        sh.write(f"#BSUB -o {os.path.join(log_path, job_name)}.o\n")
        sh.write(f"#BSUB -e {os.path.join(log_path, job_name)}.e\n")
        sh.write(
            "python -u job_heros_sum.py"
            f" --o {output_root}"
            f" --ol {outcome_label} --il {instanceID_label} --el {excluded_column}"
            f" --cv {cv_partitions} --r {random_seeds}\n"
        )

    os.system(f"bsub < {job_script}")

def submit_slurm(scratch_path, log_path, reserved_memory, queue, output_root, cv_partitions, random_seeds, outputfolder,
                 outcome_label, instanceID_label, excluded_column):
    job_ref = str(time.time())
    job_name = f"HEROS_summary_{outputfolder}_{job_ref}"
    job_script = os.path.join(scratch_path, f"{job_name}_run.sh")

    with open(job_script, "w") as sh:
        sh.write("#!/bin/bash\n")
        sh.write(f"#SBATCH -p {queue}\n")
        sh.write(f"#SBATCH --job-name={job_name}\n")
        sh.write(f"#SBATCH --mem={reserved_memory}G\n")
        sh.write(f"#SBATCH -o {os.path.join(log_path, job_name)}.o\n")
        sh.write(f"#SBATCH -e {os.path.join(log_path, job_name)}.e\n")
        sh.write(
            "srun python -u job_heros_sum.py"
            f" --o {output_root}"
            f" --ol {outcome_label} --il {instanceID_label} --el {excluded_column}"
            f" --cv {cv_partitions} --r {random_seeds}\n"
        )

    os.system(f"sbatch {job_script}")

def main(argv):
    parser = argparse.ArgumentParser(description="Submit HEROS summary aggregation job.")
    parser.add_argument("--w", dest="writepath", type=str, required=True, help="Base write path (contains output/, logs/, scratch/).")
    parser.add_argument("--o", dest="outputfolder", type=str, required=True, help="Experiment output folder name (suffix after HEROS_).")
    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    parser.add_argument("--rc", dest="run_cluster", type=str, default="LSF", choices=["LSF", "SLURM"])
    parser.add_argument("--rm", dest="reserved_memory", type=int, default=4)
    parser.add_argument("--q", dest="queue", type=str, default="i2c2_normal")

    args = parser.parse_args(argv[1:])

    writepath = args.writepath
    outputfolder = args.outputfolder
    cv_partitions = args.cv_partitions
    random_seeds = args.random_seeds

    algorithm = "HEROS"
    output_root = os.path.join(writepath, "output", f"{algorithm}_{outputfolder}")

    # Ensure dirs exist
    ensure_dir(writepath)
    ensure_dir(os.path.join(writepath, "output"))
    ensure_dir(os.path.join(writepath, "scratch"))
    ensure_dir(os.path.join(writepath, "logs"))
    ensure_dir(output_root)

    if args.run_cluster == "LSF":
        submit_lsf(
            scratch_path=os.path.join(writepath, "scratch"),
            log_path=os.path.join(writepath, "logs"),
            reserved_memory=args.reserved_memory,
            queue=args.queue,
            output_root=output_root,
            cv_partitions=cv_partitions,
            random_seeds=random_seeds,
            outputfolder=outputfolder,
            outcome_label=args.outcome_label,
            instanceID_label=args.instanceID_label,
            excluded_column=args.excluded_column,
        )
    else:
        submit_slurm(
            scratch_path=os.path.join(writepath, "scratch"),
            log_path=os.path.join(writepath, "logs"),
            reserved_memory=args.reserved_memory,
            queue=args.queue,
            output_root=output_root,
            cv_partitions=cv_partitions,
            random_seeds=random_seeds,
            outputfolder=outputfolder,
            outcome_label=args.outcome_label,
            instanceID_label=args.instanceID_label,
            excluded_column=args.excluded_column,
        )

    print("1 summary job submitted successfully")

if __name__ == "__main__":
    sys.exit(main(sys.argv))
