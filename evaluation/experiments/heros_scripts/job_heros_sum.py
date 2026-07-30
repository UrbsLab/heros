# I2C2-Documentation/job_heros_alt_sum.py
#
# Robust summary/aggregation script for HEROS CV + seed experiments.
# - No HEROS import required
# - Skips missing CV files (prints what’s missing)
# - Derives evaluation points dynamically (no hard-coded row_index)
# - Produces:
#   seed_i/mean_CV_evaluation_summary.csv, seed_i/sd_CV_evaluation_summary.csv
#   dataset/mean_seed_evaluation_summary.csv, dataset/sd_seed_evaluation_summary.csv
#   dataset/all_<row>_evaluations.csv  (all seed×cv rows)
#   dataset/cv_ave_<row>_evaluations.csv (seed-level CV means)
#   dataset/boxplot_testing_accuracy_all.png, dataset/boxplot_rule_count_all.png
#   outputPath/phase1_mux_ideal_success*.csv (if multiplexer outputs)

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return None

def list_datasets(output_path: str) -> list[str]:
    datasets = []
    for entry in sorted(os.listdir(output_path)):
        p = os.path.join(output_path, entry)
        if os.path.isdir(p):
            datasets.append(entry)
    return datasets

def get_row_indexes_from_any_eval(output_path: str, dataset: str, random_seeds: int, cv_partitions: int) -> list[str]:
    # Find first existing evaluation_summary.csv and return its Row Indexes
    for i in range(random_seeds):
        fp = os.path.join(output_path, dataset, f"seed_{i}", "run_1/evaluation_summary.csv")
        df = safe_read_csv(fp)
        if df is not None and "Row Indexes" in df.columns:
            rows = df["Row Indexes"].tolist()
            return rows
    return []

def aggregate_seed_level(output_path: str, dataset: str, seed: int, cv_partitions: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns (mean_df, sd_df) for this seed across CVs.
    Each df includes 'Row Indexes' column.
    """
    seed_path = os.path.join(output_path, dataset, f"seed_{seed}")
    dfs = []
    row_names = None

    missing = 0
    
    fp = os.path.join(seed_path, "run_1/evaluation_summary.csv")
    df = safe_read_csv(fp)
    if df is None:
        missing += 1
        print(f"[MISSING] {fp}")
        
    if "Row Indexes" not in df.columns:
        print(f"[WARN] No 'Row Indexes' in {fp}; skipping")
        missing += 1
        

    row_names = df["Row Indexes"]
    df_x = df.drop(columns=["Row Indexes"])
    dfs.append(df_x)

    if not dfs:
        print(f"[WARN] No CV eval files found for dataset={dataset}, seed={seed}")
        return None, None

    # Concatenate vertically and group by row position (level=0) to average CVs
    mean_x = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
    sd_x   = pd.concat(dfs).groupby(level=0).std(numeric_only=True)

    mean_df = pd.concat([row_names, mean_x], axis=1)
    sd_df   = pd.concat([row_names, sd_x], axis=1)

    # Save
    mean_df.to_csv(os.path.join(seed_path, "mean_CV_evaluation_summary.csv"), index=False)
    sd_df.to_csv(os.path.join(seed_path, "sd_CV_evaluation_summary.csv"), index=False)

    if missing > 0:
        print(f"[INFO] dataset={dataset} seed={seed}: skipped {missing}/{cv_partitions} missing CV eval files")

    return mean_df, sd_df

def aggregate_dataset_level(output_path: str, dataset: str, random_seeds: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Uses seed_i/mean_CV_evaluation_summary.csv to compute mean/sd across seeds.
    """
    dataset_path = os.path.join(output_path, dataset)
    dfs = []
    row_names = None

    missing = 0
    for i in range(random_seeds):
        fp = os.path.join(dataset_path, f"seed_{i}", "mean_CV_evaluation_summary.csv")
        df = safe_read_csv(fp)
        if df is None:
            missing += 1
            print(f"[MISSING] {fp}")
            continue
        if "Row Indexes" not in df.columns:
            missing += 1
            print(f"[WARN] No 'Row Indexes' in {fp}; skipping")
            continue

        row_names = df["Row Indexes"]
        df_x = df.drop(columns=["Row Indexes"])
        dfs.append(df_x)

    if not dfs:
        print(f"[WARN] No seed mean CV files found for dataset={dataset}")
        return None, None

    mean_x = pd.concat(dfs).groupby(level=0).mean(numeric_only=True)
    sd_x   = pd.concat(dfs).groupby(level=0).std(numeric_only=True)

    mean_df = pd.concat([row_names, mean_x], axis=1)
    sd_df   = pd.concat([row_names, sd_x], axis=1)

    mean_df.to_csv(os.path.join(dataset_path, "mean_seed_evaluation_summary.csv"), index=False)
    sd_df.to_csv(os.path.join(dataset_path, "sd_seed_evaluation_summary.csv"), index=False)

    if missing > 0:
        print(f"[INFO] dataset={dataset}: skipped {missing}/{random_seeds} missing seed mean CV files")

    return mean_df, sd_df

def write_global_eval_lists(output_path: str, dataset: str, random_seeds: int, cv_partitions: int) -> None:
    """
    Creates dataset-level:
      all_<row_index>_evaluations.csv      (every seed×cv result row)
      cv_ave_<row_index>_evaluations.csv   (every seed’s CV-mean result row)
    """
    dataset_path = os.path.join(output_path, dataset)

    # Determine row indexes dynamically (from any existing cv eval)
    row_indexes = get_row_indexes_from_any_eval(output_path, dataset, random_seeds, cv_partitions)
    if not row_indexes:
        print(f"[WARN] Could not determine Row Indexes for dataset={dataset}; skipping global lists")
        return

    # Build "all runs" dict row_index -> list of rows (without Row Indexes)
    header = None
    eval_point_all: dict[str, list[list]] = {ri: [] for ri in row_indexes}

    missing = 0
    for i in range(random_seeds):
        fp = os.path.join(dataset_path, f"seed_{i}", "run_1/evaluation_summary.csv")
        df = safe_read_csv(fp)
        if df is None or "Row Indexes" not in df.columns:
            missing += 1
            

        if header is None:
            header = [c for c in df.columns if c != "Row Indexes"]

        for _, row in df.iterrows():
            ri = row["Row Indexes"]
            if ri not in eval_point_all:
                # In case different runs have extra rows, include them too
                eval_point_all[ri] = []
            values = [row[c] for c in header]
            eval_point_all[ri].append(values)

    if header is None:
        print(f"[WARN] No evaluation_summary.csv readable for dataset={dataset}; skipping")
        return

    for ri, rows in eval_point_all.items():
        if not rows:
            continue
        pd.DataFrame(rows, columns=header).to_csv(
            os.path.join(dataset_path, f"all_{ri}_evaluations.csv"), index=False
        )

    if missing > 0:
        print(f"[INFO] dataset={dataset}: skipped {missing} missing/bad cv eval files while building all_* lists")

    # Now build CV-average lists from each seed's mean_CV_evaluation_summary.csv
    eval_point_cv_ave: dict[str, list[list]] = {}
    header2 = None

    missing2 = 0
    for i in range(random_seeds):
        fp = os.path.join(dataset_path, f"seed_{i}", "mean_CV_evaluation_summary.csv")
        df = safe_read_csv(fp)
        if df is None or "Row Indexes" not in df.columns:
            missing2 += 1
            continue

        if header2 is None:
            header2 = [c for c in df.columns if c != "Row Indexes"]

        for _, row in df.iterrows():
            ri = row["Row Indexes"]
            if ri not in eval_point_cv_ave:
                eval_point_cv_ave[ri] = []
            values = [row[c] for c in header2]
            eval_point_cv_ave[ri].append(values)

    if header2 is not None:
        for ri, rows in eval_point_cv_ave.items():
            if not rows:
                continue
            pd.DataFrame(rows, columns=header2).to_csv(
                os.path.join(dataset_path, f"cv_ave_{ri}_evaluations.csv"), index=False
            )

    if missing2 > 0:
        print(f"[INFO] dataset={dataset}: skipped {missing2} missing seed mean CV files while building cv_ave_* lists")

def make_boxplots(output_path: str, dataset: str, random_seeds: int, cv_partitions: int) -> None:
    """
    Uses the dynamically discovered row indexes and all_<ri>_evaluations.csv files.
    Produces:
      boxplot_testing_accuracy_all.png
      boxplot_rule_count_all.png
    """
    dataset_path = os.path.join(output_path, dataset)
    row_indexes = get_row_indexes_from_any_eval(output_path, dataset, random_seeds, cv_partitions)
    if not row_indexes:
        print(f"[WARN] No row indexes found for dataset={dataset}; skipping plots")
        return

    # Accuracy plot
    acc_series = []
    acc_labels = []
    for ri in row_indexes:
        fp = os.path.join(dataset_path, f"all_{ri}_evaluations.csv")
        df = safe_read_csv(fp)
        if df is None:
            continue
        if "test_balanced_accuracy" not in df.columns:
            continue
        acc_series.append(df["test_balanced_accuracy"])
        acc_labels.append(ri)

    if acc_series:
        acc_df = pd.concat(acc_series, axis=1)
        acc_df.columns = acc_labels
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=acc_df)
        plt.xticks(ticks=range(len(acc_labels)), labels=acc_labels, rotation=90)
        plt.xlabel("Evaluation Points")
        plt.ylabel("Balanced Testing Accuracy")
        plt.savefig(os.path.join(dataset_path, "boxplot_testing_accuracy_all.png"), bbox_inches="tight")
        plt.close()

    # Rule count plot
    cnt_series = []
    cnt_labels = []
    for ri in row_indexes:
        fp = os.path.join(dataset_path, f"all_{ri}_evaluations.csv")
        df = safe_read_csv(fp)
        if df is None:
            continue
        if "rule_count" not in df.columns:
            continue
        cnt_series.append(df["rule_count"])
        cnt_labels.append(ri)

    if cnt_series:
        cnt_df = pd.concat(cnt_series, axis=1)
        cnt_df.columns = cnt_labels
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=cnt_df)
        plt.xticks(ticks=range(len(cnt_labels)), labels=cnt_labels, rotation=90)
        plt.xlabel("Evaluation Points")
        plt.ylabel("Rule Count")
        plt.savefig(os.path.join(dataset_path, "boxplot_rule_count_all.png"), bbox_inches="tight")
        plt.close()

# ---- Multiplexer ideal-rule utilities (unchanged logic, but robust file checks) ----

def int_to_binary_list(num: int, n: int) -> list[int]:
    return [int(bit) for bit in format(num, f"0{n}b")]

def gen_ideal_rules(mux: int) -> list[list]:
    address_bits = {6:2, 11:3, 20:4, 37:5, 70:6, 135:7}
    ideal_list = []
    reg_bits = mux - address_bits[mux]
    for i in range(reg_bits):
        idx = list(range(address_bits[mux])) + [i + address_bits[mux]]

        vals0 = int_to_binary_list(i, address_bits[mux]) + [0]
        ideal_list.append([str(idx), str(vals0), 0])

        vals1 = int_to_binary_list(i, address_bits[mux]) + [1]
        ideal_list.append([str(idx), str(vals1), 1])

    return ideal_list

def mux_ideal_rule_scan(output_path: str, cv_partitions: int, random_seeds: int) -> None:
    # Only do if "multiplexer" experiment folder name suggests mux
    if "multiplexer" not in output_path.lower():
        return

    # You can extend this mapping as needed
    mux_output_folders = {
        "A_multiplexer_6_bit_500_inst": 6,
        "B_multiplexer_11_bit_5000_inst": 11,
        "C_multiplexer_20_bit_10000_inst": 20,
        "D_multiplexer_37_bit_10000_inst": 37,
        "E_multiplexer_70_bit_20000_inst": 70,
    }

    header = ["Dataset", "Seed", "CV", "Ideal Count", "Ideal Proportion"]
    results = []

    for dataset in list_datasets(output_path):
        if dataset not in mux_output_folders:
            continue

        ideal_rules = gen_ideal_rules(mux_output_folders[dataset])
        ideal_count = len(ideal_rules)
        dataset_path = os.path.join(output_path, dataset)

        for i in range(random_seeds):
            for j in range(1, cv_partitions + 1):
                rule_fp = os.path.join(dataset_path, f"seed_{i}", f"cv_{j}", "rule_pop.csv")
                df = safe_read_csv(rule_fp)
                if df is None:
                    print(f"[MISSING] {rule_fp}")
                    continue

                if not {"Condition Indexes", "Condition Values", "Action"}.issubset(df.columns):
                    print(f"[WARN] rule_pop.csv missing expected columns in {rule_fp}")
                    continue

                found = 0
                for _, row in df.iterrows():
                    combined = [row["Condition Indexes"], row["Condition Values"], int(row["Action"])]
                    if combined in ideal_rules:
                        found += 1

                results.append([dataset, i, j, found, (found / float(ideal_count)) if ideal_count else 0.0])

    if not results:
        return

    out_csv = os.path.join(output_path, "phase1_mux_ideal_success.csv")
    df_res = pd.DataFrame(results, columns=header)
    df_res.to_csv(out_csv, index=False)

    # Average summary per dataset
    avg_rows = []
    for dataset in df_res["Dataset"].unique():
        avg_count = df_res.loc[df_res["Dataset"] == dataset, "Ideal Count"].mean()
        avg_prop  = df_res.loc[df_res["Dataset"] == dataset, "Ideal Proportion"].mean()
        avg_rows.append([dataset, avg_count, avg_prop])

    df_avg = pd.DataFrame(avg_rows, columns=["Dataset", "Ideal Count", "Ideal Proportion"])
    df_avg.to_csv(os.path.join(output_path, "phase1_mux_ideal_success_average.csv"), index=False)

# ---- main ----

def main(argv):
    parser = argparse.ArgumentParser(description="Aggregate HEROS CV/seed results.")
    parser.add_argument("--o", dest="outputPath", type=str, required=True, help="Path to HEROS output folder (algorithm_output root).")
    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    args = parser.parse_args(argv[1:])

    output_path = args.outputPath
    cv_partitions = args.cv_partitions
    random_seeds = args.random_seeds

    print(f"[INFO] Summarizing: {output_path}")
    datasets = list_datasets(output_path)
    print(f"[INFO] Found datasets: {datasets}")

    # 1) seed-level mean/sd across CVs
    for dataset in datasets:
        for seed in range(random_seeds):
            aggregate_seed_level(output_path, dataset, seed, cv_partitions)

    # 2) dataset-level mean/sd across seeds
    for dataset in datasets:
        aggregate_dataset_level(output_path, dataset, random_seeds)

    # 3) global lists + plots
    for dataset in datasets:
        write_global_eval_lists(output_path, dataset, random_seeds, cv_partitions)
        make_boxplots(output_path, dataset, random_seeds, cv_partitions)

    # 4) mux ideal rule scan (optional)
    mux_ideal_rule_scan(output_path, cv_partitions, random_seeds)

    print("[INFO] Done.")

if __name__ == "__main__":
    sys.exit(main(sys.argv))
