#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_csv(path)


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _make_eval_df_from_result_row(result_row_path: Path) -> pd.DataFrame:
    """
    Convert BioHEL's per-fold result_row.csv into a HEROS-like evaluation_summary.csv
    structure with a single evaluation point: Row Indexes == 'final'.

    HEROS expects:
      - a column 'Row Indexes'
      - other metric columns numeric
    """
    rr = _safe_read_csv(result_row_path)

    # Normalize column names we want to carry forward
    # Keep a stable set across versions; missing ones become NaN.
    wanted_cols = [
        "train_accuracy",
        "test_accuracy",
        "train_coverage",
        "test_coverage",
        "train_default_rate",
        "test_default_rate",
        "num_rules",
        "runtime",
        "wall_time",
    ]
    row = {}
    for c in wanted_cols:
        row[c] = rr[c].iloc[0] if c in rr.columns and len(rr) else np.nan

    out = pd.DataFrame([row])
    out.insert(0, "Row Indexes", "final")
    return out


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL summary job (HEROS-compatible outputs)")

    # Script Parameters
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Path to BioHEL_<analysis> output folder (contains dataset subfolders)")

    # Keep these args for interface parity with HEROS (not strictly needed for BioHEL summary)
    # parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    # parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    # parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    # Experiment Parameters
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)

    parser.add_argument("--plots", dest="plots", action="store_true",
                        help="If set, generate boxplots like HEROS")

    options = parser.parse_args(argv[1:])

    outputPath = Path(options.outputPath)
    cv_partitions = int(options.cv_partitions)
    random_seeds = int(options.random_seeds)
    make_plots = bool(options.plots)

    if not outputPath.exists():
        raise FileNotFoundError(f"Output path does not exist: {outputPath}")

    # Metrics we aggregate (superset; missing values allowed)
    metric_cols = [
        "train_accuracy",
        "test_accuracy",
        "train_coverage",
        "test_coverage",
        "train_default_rate",
        "test_default_rate",
        "num_rules",
        "runtime",
        "wall_time",
    ]

    # ----------------------------
    # 1) Create CV-level evaluation_summary.csv files (HEROS compatibility)
    #    (Optional but strongly recommended for consistency)
    # ----------------------------
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.exists():
                continue

            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                if not cv_level_path.exists():
                    continue

                eval_path = cv_level_path / "evaluation_summary.csv"
                if eval_path.exists():
                    # already there; do not overwrite
                    continue

                rr_path = cv_level_path / "result_row.csv"
                if rr_path.exists():
                    eval_df = _make_eval_df_from_result_row(rr_path)
                    eval_df.to_csv(eval_path, index=False)

    # ----------------------------
    # 2) Seed-level CV summaries (mean/sd across CV folds)
    #    Output filenames match HEROS:
    #      seed_i/mean_CV_evaluation_summary.csv
    #      seed_i/sd_CV_evaluation_summary.csv
    # ----------------------------
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.is_dir():
                continue

            dfs = []
            row_names = None

            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                eval_path = cv_level_path / "evaluation_summary.csv"
                if not eval_path.exists():
                    continue

                df = _safe_read_csv(eval_path)
                if "Row Indexes" not in df.columns:
                    continue

                row_names = df["Row Indexes"]
                df_x = df.drop(columns=["Row Indexes"])
                df_x = _ensure_numeric(df_x, list(df_x.columns))
                dfs.append(df_x)

            if not dfs:
                continue

            # Mean across CV folds
            ave_df_x = pd.concat(dfs).groupby(level=0).mean()
            mean_df = pd.concat([row_names, ave_df_x], axis=1)
            mean_df.to_csv(seed_level_path / "mean_CV_evaluation_summary.csv", index=False)

            # SD across CV folds
            sd_df_x = pd.concat(dfs).groupby(level=0).std()
            sd_df = pd.concat([row_names, sd_df_x], axis=1)
            sd_df.to_csv(seed_level_path / "sd_CV_evaluation_summary.csv", index=False)

    # ----------------------------
    # 3) Dataset-level seed summaries (mean/sd across seeds)
    #    Output filenames match HEROS:
    #      mean_seed_evaluation_summary.csv
    #      sd_seed_evaluation_summary.csv
    # ----------------------------
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        dfs = []
        row_names = None

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            mean_path = seed_level_path / "mean_CV_evaluation_summary.csv"
            if not mean_path.exists():
                continue

            df = _safe_read_csv(mean_path)
            if "Row Indexes" not in df.columns:
                continue

            row_names = df["Row Indexes"]
            df_x = df.drop(columns=["Row Indexes"])
            df_x = _ensure_numeric(df_x, list(df_x.columns))
            dfs.append(df_x)

        if not dfs:
            continue

        ave_df_x = pd.concat(dfs).groupby(level=0).mean()
        mean_df = pd.concat([row_names, ave_df_x], axis=1)
        mean_df.to_csv(data_level_path / "mean_seed_evaluation_summary.csv", index=False)

        sd_df_x = pd.concat(dfs).groupby(level=0).std()
        sd_df = pd.concat([row_names, sd_df_x], axis=1)
        sd_df.to_csv(data_level_path / "sd_seed_evaluation_summary.csv", index=False)

    # ----------------------------
    # 4) Global results lists (HEROS style)
    #    Since BioHEL only has one evaluation point, we use:
    #      Row Indexes == 'final'
    #
    # Output filenames match HEROS pattern:
    #   all_final_evaluations.csv         (all seed×cv)
    #   cv_ave_final_evaluations.csv      (CV-averaged per seed)
    # ----------------------------
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        # all runs (seed×cv)
        rows_all = []

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.is_dir():
                continue

            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                rr_path = cv_level_path / "result_row.csv"
                if not rr_path.exists():
                    continue

                rr = _safe_read_csv(rr_path)
                rr_row = {c: (rr[c].iloc[0] if c in rr.columns and len(rr) else np.nan) for c in metric_cols}
                rr_row["Seed"] = i
                rr_row["CV"] = j
                rows_all.append(rr_row)

        if rows_all:
            df_all = pd.DataFrame(rows_all)
            df_all = _ensure_numeric(df_all, metric_cols)
            df_all.to_csv(data_level_path / "all_final_evaluations.csv", index=False)

        # CV-averaged per seed (seed-level mean across cv)
        rows_cv_ave = []
        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            mean_path = seed_level_path / "mean_CV_evaluation_summary.csv"
            if not mean_path.exists():
                continue

            df = _safe_read_csv(mean_path)
            df = _ensure_numeric(df, metric_cols)

            # pick the 'final' row
            if "Row Indexes" not in df.columns:
                continue
            df_final = df[df["Row Indexes"] == "final"].copy()
            if df_final.empty:
                continue

            row = {c: (df_final[c].iloc[0] if c in df_final.columns else np.nan) for c in metric_cols}
            row["Seed"] = i
            rows_cv_ave.append(row)

        if rows_cv_ave:
            df_cv_ave = pd.DataFrame(rows_cv_ave)
            df_cv_ave = _ensure_numeric(df_cv_ave, metric_cols)
            df_cv_ave.to_csv(data_level_path / "cv_ave_final_evaluations.csv", index=False)

    # ----------------------------
    # 5) Plots (HEROS-consistent filenames)
    # ----------------------------
    if make_plots:
        # These replicate HEROS output filenames:
        #   boxplot_testing_accuracy_all.png
        #   boxplot_rule_count_all.png
        # And add coverage plots with consistent naming:
        #   boxplot_testing_coverage_all.png
        #   boxplot_train_coverage_all.png
        for entry in os.listdir(outputPath):
            data_level_path = outputPath / entry
            if not data_level_path.is_dir():
                continue

            all_path = data_level_path / "all_final_evaluations.csv"
            if not all_path.exists():
                continue

            df_all = _safe_read_csv(all_path)
            df_all = _ensure_numeric(df_all, metric_cols + ["Seed", "CV"])

            # Test accuracy boxplot (all runs)
            if "test_accuracy" in df_all.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_all, y="test_accuracy")
                plt.xlabel("")
                plt.ylabel("Test Accuracy")
                plt.title("Balanced Testing Accuracy (All Runs)")  # keep generic title
                plt.savefig(data_level_path / "boxplot_testing_accuracy_all.png", bbox_inches="tight")
                plt.close()

            # Rule count boxplot (all runs) -> BioHEL uses num_rules; keep filename consistent
            if "num_rules" in df_all.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_all, y="num_rules")
                plt.xlabel("")
                plt.ylabel("Rule Count")
                plt.title("Rule Count (All Runs)")
                plt.savefig(data_level_path / "boxplot_rule_count_all.png", bbox_inches="tight")
                plt.close()

            # Coverage plots (new but consistent naming)
            if "test_coverage" in df_all.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_all, y="test_coverage")
                plt.xlabel("")
                plt.ylabel("Test Coverage")
                plt.title("Test Coverage (All Runs)")
                plt.savefig(data_level_path / "boxplot_testing_coverage_all.png", bbox_inches="tight")
                plt.close()

            if "train_coverage" in df_all.columns:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=df_all, y="train_coverage")
                plt.xlabel("")
                plt.ylabel("Train Coverage")
                plt.title("Train Coverage (All Runs)")
                plt.savefig(data_level_path / "boxplot_train_coverage_all.png", bbox_inches="tight")
                plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
