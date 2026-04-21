#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


DEFAULT_HEROS_CSV = "/project/kamoun_shared/output_shared/paper_tables/combined_heros_cv_default_runs_long.csv"
DEFAULT_OTHER_CSV = "/project/kamoun_shared/output_shared/paper_tables/combined_runs_long.csv"
DEFAULT_OUTDIR = "/project/kamoun_shared/output_shared/paper_tables"
DEFAULT_PREFIX = "heros_baseline_wilcoxon"

# Only test these metrics (exact list requested)
METRICS = ["test_balanced_accuracy", "test_coverage", "rule_count", "run_time"]

# Comparators (AlgorithmFamily, Scenario) in combined_runs_long.csv
COMPARATORS = [
    ("RIPPER", "RIPPER"),
    ("BioHEL", "BioHEL_noRPE"),
    ("BioHEL", "BioHEL_RPE"),
]


def mann_whitney_two_sided(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Wilcoxon rank-sum equivalent for independent samples (Mann–Whitney U), two-sided."""
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def extract_numeric(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def choose_dataset_col(heros: pd.DataFrame, other: pd.DataFrame) -> str:
    if "Dataset" in heros.columns and "Dataset" in other.columns:
        return "Dataset"
    if "dataset" in heros.columns and "dataset" in other.columns:
        return "dataset"
    raise ValueError("Expected a shared dataset identifier column ('Dataset' or 'dataset').")


def metric_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Map requested metrics to available columns.
    We still ONLY TEST the requested METRICS.
    """
    m: Dict[str, Optional[str]] = {}

    # Balanced accuracy
    if "test_balanced_accuracy" in df.columns:
        m["test_balanced_accuracy"] = "test_balanced_accuracy"
    elif "test_accuracy" in df.columns:
        # common in combined_runs_long
        m["test_balanced_accuracy"] = "test_accuracy"
    else:
        m["test_balanced_accuracy"] = None

    m["test_coverage"] = "test_coverage" if "test_coverage" in df.columns else None
    m["rule_count"] = "rule_count" if "rule_count" in df.columns else None

    # Runtime
    if "run_time" in df.columns:
        m["run_time"] = "run_time"
    elif "runtime" in df.columns:
        m["run_time"] = "runtime"
    elif "final_runtime" in df.columns:
        m["run_time"] = "final_runtime"
    else:
        m["run_time"] = None

    return m


def dataset_type(ds: str) -> str:
    s = str(ds).upper()
    if "MULTIPLEXER" in s:
        return "MUX"
    return "GAMETES"


def dataset_rep(ds: str) -> str:
    """
    Extract A/B/C replicate for sorting.
    Handles: MUX_A_..., GAMETES_B_...
    """
    s = str(ds).upper()
    s = s.split("_")[0]
    return s


def rep_sort_key(rep: str) -> int:
    if isinstance(rep, str) and rep and rep[0].isalpha():
        return ord(rep[0].upper())
    return 999


def run_tests_and_build_wide(
    heros_csv: Path,
    other_csv: Path,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      long_df: one row per dataset × comparator × metric with p, fdr, direction
      wide_df: one row per dataset × comparator, metric results spread into columns
    """
    heros = pd.read_csv(heros_csv)
    other = pd.read_csv(other_csv)

    dscol = choose_dataset_col(heros, other)

    # Baseline HEROS rows
    if "AlgorithmFamily" in heros.columns:
        heros_base = heros[heros["AlgorithmFamily"].astype(str).str.upper() == "HEROS"].copy()
    else:
        heros_base = heros.copy()

    map_heros = metric_column_mapping(heros_base)
    map_other = metric_column_mapping(other)

    metrics_to_test = [m for m in METRICS if map_heros.get(m) and map_other.get(m)]
    if not metrics_to_test:
        raise RuntimeError(
            "None of the requested metrics could be mapped in both inputs.\n"
            f"Requested: {METRICS}\n"
            f"HEROS mapping: {map_heros}\n"
            f"OTHER mapping: {map_other}"
        )

    # Dataset overlap
    ds_common = sorted(set(heros_base[dscol].dropna().unique()) & set(other[dscol].dropna().unique()))
    if not ds_common:
        raise RuntimeError(f"No overlapping datasets found between files using column '{dscol}'.")

    rows = []
    for ds in ds_common:
        base_ds = heros_base[heros_base[dscol] == ds]
        if base_ds.empty:
            continue

        for alg, scen in COMPARATORS:
            comp_ds = other[
                (other["AlgorithmFamily"].astype(str) == alg)
                & (other["Scenario"].astype(str) == scen)
                & (other[dscol] == ds)
            ]
            if comp_ds.empty:
                continue

            for metric in metrics_to_test:
                bcol = map_heros[metric]
                ccol = map_other[metric]
                assert bcol and ccol

                base_vals = extract_numeric(base_ds[bcol])
                comp_vals = extract_numeric(comp_ds[ccol])

                if base_vals.size < 2 or comp_vals.size < 2:
                    continue

                _, p_val = mann_whitney_two_sided(base_vals, comp_vals)

                base_med = float(np.median(base_vals))
                comp_med = float(np.median(comp_vals))
                diff = comp_med - base_med
                direction = (
                    "Comparator higher" if diff > 0 else
                    "Comparator lower" if diff < 0 else
                    "Tie (median)"
                )

                rows.append({
                    "Dataset": ds,
                    "DatasetType": dataset_type(ds),
                    "Rep": dataset_rep(ds),
                    "BaselineAlgorithm": "HEROS",
                    "BaselineScenario": "default_model",
                    "ComparatorAlgorithm": alg,
                    "ComparatorScenario": scen,
                    "Metric": metric,
                    "p_value": float(p_val),
                    "direction": direction,
                })

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No tests were run; check filters, dataset overlap, and metric mappings.")

    # FDR across ALL tests (dataset×comparator×metric)
    rej, p_adj, _, _ = multipletests(long_df["p_value"].to_numpy(), alpha=alpha, method="fdr_bh")
    long_df["p_value_fdr_bh"] = p_adj
    long_df["significant"] = rej

    # Build wide by columns per metric (p, fdr, dir), with Algorithm+Scenario as columns (per row)
    wide = long_df.pivot_table(
        index=[
            "Dataset", "DatasetType", "Rep",
            "ComparatorAlgorithm", "ComparatorScenario",
        ],
        columns="Metric",
        values=["p_value", "p_value_fdr_bh", "direction", "significant"],
        aggfunc="first",
    )

    # Flatten columns: (stat, metric) -> f"{metric}__{stat}"
    wide.columns = [f"{metric}__{stat}" for stat, metric in wide.columns]
    wide = wide.reset_index()

    return long_df, wide


def keep_only_significant_metric_columns(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only metric column-groups (for each metric: p, fdr, dir) that are ever significant
    in ANY row, based on the corresponding metric__significant column.
    Always keep identifier columns.
    """
    id_cols = ["Dataset", "DatasetType", "Rep", "ComparatorAlgorithm", "ComparatorScenario"]

    # Determine which metrics are ever significant
    keep_metrics: List[str] = []
    for m in METRICS:
        sig_col = f"{m}__significant"
        if sig_col in wide.columns and bool(wide[sig_col].fillna(False).any()):
            keep_metrics.append(m)

    # Build final column list: per metric keep only p, fdr, dir (not the boolean)
    keep_cols = id_cols[:]
    for m in keep_metrics:
        for suffix in ["p_value", "p_value_fdr_bh", "direction"]:
            c = f"{m}__{suffix}"
            if c in wide.columns:
                keep_cols.append(c)

    out = wide[keep_cols].copy()
    return out


def sort_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by dataset type (MUX then GAMETES), then A/B/C, then algorithm (and scenario), then dataset string.
    """
    type_order = {"MUX": 0, "GAMETES": 1, "OTHER": 99}
    df = df.copy()
    df["_type_sort"] = df["DatasetType"].map(lambda x: type_order.get(str(x), 99))
    df["_rep_sort"] = df["Rep"].map(rep_sort_key)
    df = df.sort_values(
        by=["_type_sort", "_rep_sort", "ComparatorAlgorithm", "ComparatorScenario", "Dataset"],
        kind="mergesort",
    ).drop(columns=["_type_sort", "_rep_sort"])
    return df


def write_outputs(
    df_all: pd.DataFrame,
    outdir: Path,
    prefix: str,
) -> Tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    all_path = outdir / f"{prefix}_significant_only_all.csv"
    mux_path = outdir / f"{prefix}_significant_only_mux.csv"
    gam_path = outdir / f"{prefix}_significant_only_gametes.csv"

    df_all.to_csv(all_path, index=False)
    df_all[df_all["DatasetType"] == "MUX"].to_csv(mux_path, index=False)
    df_all[df_all["DatasetType"] == "GAMETES"].to_csv(gam_path, index=False)

    return all_path, mux_path, gam_path


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Wilcoxon rank-sum (Mann–Whitney U) per-dataset tests using HEROS as baseline.\n"
            "Outputs WIDE tables with Algorithm+Scenario as key columns, and metrics as column groups.\n"
            "Keeps ONLY metric column groups that are significant (FDR-BH) anywhere.\n"
            "Also writes MUX-only and GAMETES-only subsets."
        )
    )
    p.add_argument("--heros_csv", type=Path, default=Path(DEFAULT_HEROS_CSV), help="HEROS combined CSV")
    p.add_argument("--other_csv", type=Path, default=Path(DEFAULT_OTHER_CSV), help="Other algorithms combined CSV")
    p.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR), help=f"Output directory (default: {DEFAULT_OUTDIR})")
    p.add_argument("--prefix", type=str, default=DEFAULT_PREFIX, help="Filename prefix for outputs")
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR-BH significance (default: 0.05)")
    return p


def main(argv: List[str]) -> int:
    args = build_argparser().parse_args(argv)

    _, wide = run_tests_and_build_wide(
        heros_csv=args.heros_csv,
        other_csv=args.other_csv,
        alpha=args.alpha,
    )

    wide_sig_only = keep_only_significant_metric_columns(wide)
    wide_sig_only = sort_output(wide_sig_only)

    all_path, mux_path, gam_path = write_outputs(wide_sig_only, args.outdir, args.prefix)

    print(f"Wrote:\n  {all_path}\n  {mux_path}\n  {gam_path}")
    print(f"Rows (all): {wide_sig_only.shape[0]} | Cols: {wide_sig_only.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
