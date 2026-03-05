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


def _get_first_row_value(rr: pd.DataFrame, col: str) -> float:
    if rr is None or rr.empty or col not in rr.columns:
        return np.nan
    return rr[col].iloc[0]


def _make_eval_df_from_result_rows(rr_raw_path: Path | None, rr_final_path: Path | None) -> pd.DataFrame:
    """
    Build HEROS-compatible evaluation_summary.csv.

    Output policy (per your requirement):
      - raw row (non-RPE): use result_row_raw.csv; keep num_rules, runtime, wall_time as-is
      - final row (RPE):  use result_row.csv but normalize:
            runtime   = wall_time + postprocess_wall_time
            wall_time = wall_time + postprocess_wall_time
            num_rules = postprocess_num_rules
        and DROP postprocess_* columns from the eval tables (they won't be written).

    Backward-compat:
      - If only result_row.csv exists (legacy non-RPE), treat it as raw.
    """

    # Columns that will exist in evaluation_summary.csv (NO postprocess_* columns)
    eval_cols = [
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

    rows = []

    # ---------- RAW (non-RPE) ----------
    if rr_raw_path is not None and rr_raw_path.exists():
        rr_raw = _safe_read_csv(rr_raw_path)
        row = {c: _get_first_row_value(rr_raw, c) for c in eval_cols}
        row["Row Indexes"] = "raw"
        rows.append(row)

    # ---------- FINAL (RPE) ----------
    if rr_final_path is not None and rr_final_path.exists():
        rr_final = _safe_read_csv(rr_final_path)

        # Base values
        train_acc = _get_first_row_value(rr_final, "train_accuracy")
        test_acc = _get_first_row_value(rr_final, "test_accuracy")
        train_cov = _get_first_row_value(rr_final, "train_coverage")
        test_cov = _get_first_row_value(rr_final, "test_coverage")
        train_def = _get_first_row_value(rr_final, "train_default_rate")
        test_def = _get_first_row_value(rr_final, "test_default_rate")

        wall = _get_first_row_value(rr_final, "wall_time")
        post = _get_first_row_value(rr_final, "postprocess_wall_time")
        # normalize missing postprocess to 0
        if np.isnan(post):
            post = 0.0

        # RPE normalization rules:
        wall_total = (wall + post) if not np.isnan(wall) else np.nan
        runtime_total = wall_total  # per your requirement
        # num_rules from postprocess_num_rules (fallback to num_rules if missing)
        prules = _get_first_row_value(rr_final, "postprocess_num_rules")
        if np.isnan(prules):
            prules = _get_first_row_value(rr_final, "num_rules")

        row = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_coverage": train_cov,
            "test_coverage": test_cov,
            "train_default_rate": train_def,
            "test_default_rate": test_def,
            "num_rules": prules,
            "runtime": runtime_total,
            "wall_time": wall_total,
            "Row Indexes": "final",
        }
        rows.append(row)

    # ---------- LEGACY: only result_row.csv present; treat as raw ----------
    if not rows and rr_final_path is not None and rr_final_path.exists():
        rr = _safe_read_csv(rr_final_path)
        row = {c: _get_first_row_value(rr, c) for c in eval_cols}
        row["Row Indexes"] = "raw"
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Row Indexes"] + eval_cols)

    out = pd.DataFrame(rows)
    out = out[["Row Indexes"] + eval_cols]
    return out


def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL summary job (HEROS-compatible outputs)")

    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Path to BioHEL_<analysis> output folder (contains dataset subfolders)")
    parser.add_argument("--cv", dest="cv_partitions", type=int, default=10)
    parser.add_argument("--r", dest="random_seeds", type=int, default=30)
    parser.add_argument("--plots", dest="plots", action="store_true")

    options = parser.parse_args(argv[1:])

    outputPath = Path(options.outputPath)
    cv_partitions = int(options.cv_partitions)
    random_seeds = int(options.random_seeds)
    make_plots = bool(options.plots)

    if not outputPath.exists():
        raise FileNotFoundError(f"Output path does not exist: {outputPath}")

    # Metrics that exist in evaluation_summary.csv (postprocess_* dropped)
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

    # 1) Ensure evaluation_summary.csv exists per cv folder (HEROS-style interface)
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
                    continue

                rr_raw_path = cv_level_path / "result_row_raw.csv"
                rr_final_path = cv_level_path / "result_row.csv"

                if rr_raw_path.exists() or rr_final_path.exists():
                    eval_df = _make_eval_df_from_result_rows(
                        rr_raw_path if rr_raw_path.exists() else None,
                        rr_final_path if rr_final_path.exists() else None,
                    )
                    eval_df.to_csv(eval_path, index=False)

    # 2) Seed-level CV summaries: mean/sd across CV folds
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.is_dir():
                continue

            dfs = []

            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                eval_path = cv_level_path / "evaluation_summary.csv"
                if not eval_path.exists():
                    continue

                df = _safe_read_csv(eval_path)
                if "Row Indexes" not in df.columns or df.empty:
                    continue

                df = _ensure_numeric(df, metric_cols)
                df_x = df.drop(columns=["Row Indexes"])
                df_x.index = df["Row Indexes"].astype(str).values  # groupable labels
                dfs.append(df_x)

            if not dfs:
                continue

            all_x = pd.concat(dfs, axis=0)
            ave_df_x = all_x.groupby(level=0).mean()
            sd_df_x = all_x.groupby(level=0).std()

            mean_df = ave_df_x.reset_index().rename(columns={"index": "Row Indexes"})
            sd_df = sd_df_x.reset_index().rename(columns={"index": "Row Indexes"})

            mean_df.to_csv(seed_level_path / "mean_CV_evaluation_summary.csv", index=False)
            sd_df.to_csv(seed_level_path / "sd_CV_evaluation_summary.csv", index=False)

    # 3) Dataset-level seed summaries: mean/sd across seeds
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        dfs = []

        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            mean_path = seed_level_path / "mean_CV_evaluation_summary.csv"
            if not mean_path.exists():
                continue

            df = _safe_read_csv(mean_path)
            if "Row Indexes" not in df.columns or df.empty:
                continue

            df = _ensure_numeric(df, metric_cols)
            df_x = df.drop(columns=["Row Indexes"])
            df_x.index = df["Row Indexes"].astype(str).values
            dfs.append(df_x)

        if not dfs:
            continue

        all_x = pd.concat(dfs, axis=0)
        ave_df_x = all_x.groupby(level=0).mean()
        sd_df_x = all_x.groupby(level=0).std()

        mean_df = ave_df_x.reset_index().rename(columns={"index": "Row Indexes"})
        sd_df = sd_df_x.reset_index().rename(columns={"index": "Row Indexes"})

        mean_df.to_csv(data_level_path / "mean_seed_evaluation_summary.csv", index=False)
        sd_df.to_csv(data_level_path / "sd_seed_evaluation_summary.csv", index=False)

    # 4) Global results lists (HEROS-style patterns)
    for entry in os.listdir(outputPath):
        data_level_path = outputPath / entry
        if not data_level_path.is_dir():
            continue

        # (a) all_evaluations.csv and per-rowidx splits
        rows_all = []
        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            if not seed_level_path.is_dir():
                continue

            for j in range(1, cv_partitions + 1):
                cv_level_path = seed_level_path / f"cv_{j}"
                eval_path = cv_level_path / "evaluation_summary.csv"
                if not eval_path.exists():
                    continue

                df_eval = _safe_read_csv(eval_path)
                if df_eval.empty or "Row Indexes" not in df_eval.columns:
                    continue

                df_eval = _ensure_numeric(df_eval, metric_cols)

                for _, r in df_eval.iterrows():
                    rowidx = str(r["Row Indexes"])
                    rr_row = {c: (r[c] if c in df_eval.columns else np.nan) for c in metric_cols}
                    rr_row["Row Indexes"] = rowidx
                    rr_row["Seed"] = i
                    rr_row["CV"] = j
                    rows_all.append(rr_row)

        if rows_all:
            df_all = pd.DataFrame(rows_all)
            df_all = _ensure_numeric(df_all, metric_cols + ["Seed", "CV"])
            df_all.to_csv(data_level_path / "all_evaluations.csv", index=False)

            for rowidx in sorted(df_all["Row Indexes"].dropna().unique().tolist()):
                sub = df_all[df_all["Row Indexes"] == rowidx].copy()
                sub.to_csv(data_level_path / f"all_{rowidx}_evaluations.csv", index=False)

        # (b) cv_ave_evaluations.csv and per-rowidx splits
        rows_cv_ave = []
        for i in range(0, random_seeds):
            seed_level_path = data_level_path / f"seed_{i}"
            mean_path = seed_level_path / "mean_CV_evaluation_summary.csv"
            if not mean_path.exists():
                continue

            df = _safe_read_csv(mean_path)
            if df.empty or "Row Indexes" not in df.columns:
                continue

            df = _ensure_numeric(df, metric_cols)

            for _, r in df.iterrows():
                rowidx = str(r["Row Indexes"])
                row = {c: (r[c] if c in df.columns else np.nan) for c in metric_cols}
                row["Row Indexes"] = rowidx
                row["Seed"] = i
                rows_cv_ave.append(row)

        if rows_cv_ave:
            df_cv_ave = pd.DataFrame(rows_cv_ave)
            df_cv_ave = _ensure_numeric(df_cv_ave, metric_cols + ["Seed"])
            df_cv_ave.to_csv(data_level_path / "cv_ave_evaluations.csv", index=False)

            for rowidx in sorted(df_cv_ave["Row Indexes"].dropna().unique().tolist()):
                sub = df_cv_ave[df_cv_ave["Row Indexes"] == rowidx].copy()
                sub.to_csv(data_level_path / f"cv_ave_{rowidx}_evaluations.csv", index=False)

    # 5) Plots (per row type)
    if make_plots:
        for entry in os.listdir(outputPath):
            data_level_path = outputPath / entry
            if not data_level_path.is_dir():
                continue

            all_path = data_level_path / "all_evaluations.csv"
            if not all_path.exists():
                continue

            df_all = _safe_read_csv(all_path)
            if df_all.empty or "Row Indexes" not in df_all.columns:
                continue

            df_all = _ensure_numeric(df_all, metric_cols + ["Seed", "CV"])

            for rowidx in sorted(df_all["Row Indexes"].dropna().unique().tolist()):
                sub = df_all[df_all["Row Indexes"] == rowidx].copy()
                suffix = f"_{rowidx}"

                if "test_accuracy" in sub.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=sub, y="test_accuracy")
                    plt.xlabel("")
                    plt.ylabel("Test Accuracy")
                    plt.title(f"Balanced Testing Accuracy (All Runs) [{rowidx}]")
                    plt.savefig(data_level_path / f"boxplot_testing_accuracy_all{suffix}.png", bbox_inches="tight")
                    plt.close()

                if "num_rules" in sub.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=sub, y="num_rules")
                    plt.xlabel("")
                    plt.ylabel("Rule Count")
                    plt.title(f"Rule Count (All Runs) [{rowidx}]")
                    plt.savefig(data_level_path / f"boxplot_rule_count_all{suffix}.png", bbox_inches="tight")
                    plt.close()

                if "test_coverage" in sub.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=sub, y="test_coverage")
                    plt.xlabel("")
                    plt.ylabel("Test Coverage")
                    plt.title(f"Test Coverage (All Runs) [{rowidx}]")
                    plt.savefig(data_level_path / f"boxplot_testing_coverage_all{suffix}.png", bbox_inches="tight")
                    plt.close()

                if "train_coverage" in sub.columns:
                    plt.figure(figsize=(10, 6))
                    sns.boxplot(data=sub, y="train_coverage")
                    plt.xlabel("")
                    plt.ylabel("Train Coverage")
                    plt.title(f"Train Coverage (All Runs) [{rowidx}]")
                    plt.savefig(data_level_path / f"boxplot_train_coverage_all{suffix}.png", bbox_inches="tight")
                    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
