#!/usr/bin/env python3
from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


DEFAULT_HEROS_CSV = "/project/kamoun_shared/output_shared/paper_tables/combined_heros_cv_default_runs_long.csv"
DEFAULT_OTHER_CSV = "/project/kamoun_shared/output_shared/paper_tables/combined_runs_long.csv"
DEFAULT_OUT = "/project/kamoun_shared/output_shared/paper_tables/heros_baseline_wilcoxon_ranksum_by_dataset.csv"

# Only test these metrics (as requested)
METRICS = ["test_balanced_accuracy", "test_coverage", "rule_count", "run_time"]


def cliffs_delta(baseline: np.ndarray, comparator: np.ndarray) -> float:
    """
    Cliff's delta for comparator vs baseline.
    Positive => comparator tends to be larger than baseline.
    """
    baseline = baseline.astype(float)
    comparator = comparator.astype(float)

    # O(n*m) exact computation; fine for typical eval sizes (e.g., 200 vs 200).
    diffs = comparator[:, None] - baseline[None, :]
    gt = np.sum(diffs > 0)
    lt = np.sum(diffs < 0)
    return float((gt - lt) / (len(baseline) * len(comparator)))


def mann_whitney_two_sided(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Mann-Whitney U test, two-sided.
    SciPy calls this MannWhitneyU; it's equivalent to Wilcoxon rank-sum for independent samples.
    Returns (U, p).
    """
    res = stats.mannwhitneyu(x, y, alternative="two-sided", method="auto")
    return float(res.statistic), float(res.pvalue)


def choose_dataset_key(heros: pd.DataFrame, other: pd.DataFrame) -> str:
    """
    Prefer "Dataset" if it intersects; otherwise fall back to "dataset" if present.
    """
    if "Dataset" in heros.columns and "Dataset" in other.columns:
        return "Dataset"
    if "dataset" in heros.columns and "dataset" in other.columns:
        return "dataset"
    raise ValueError("Could not find a shared dataset identifier column (expected 'Dataset' or 'dataset').")


def metric_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Map requested metric names to actual columns present.
    We *only* compute tests for metrics we can locate.

    - test_balanced_accuracy:
        - if present, use directly
        - else fall back to test_accuracy if present (common in combined_runs_long)
    - test_coverage: must exist
    - rule_count: must exist
    - run_time:
        - prefer run_time
        - else fall back to runtime / final_runtime
    """
    mapping: Dict[str, Optional[str]] = {}

    # balanced accuracy
    if "test_balanced_accuracy" in df.columns:
        mapping["test_balanced_accuracy"] = "test_balanced_accuracy"
    elif "test_accuracy" in df.columns:
        mapping["test_balanced_accuracy"] = "test_accuracy"
    else:
        mapping["test_balanced_accuracy"] = None

    mapping["test_coverage"] = "test_coverage" if "test_coverage" in df.columns else None
    mapping["rule_count"] = "rule_count" if "rule_count" in df.columns else None

    # runtime naming
    if "run_time" in df.columns:
        mapping["run_time"] = "run_time"
    elif "runtime" in df.columns:
        mapping["run_time"] = "runtime"
    elif "final_runtime" in df.columns:
        mapping["run_time"] = "final_runtime"
    else:
        mapping["run_time"] = None

    return mapping


def extract_numeric(series: pd.Series) -> np.ndarray:
    """
    Extract numeric values safely, dropping NaNs and non-finite values.
    """
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    return x


def run_tests(
    heros_csv: Path,
    other_csv: Path,
    out_csv: Path,
    alpha: float = 0.05,
) -> pd.DataFrame:
    heros = pd.read_csv(heros_csv)
    other = pd.read_csv(other_csv)

    dataset_col = choose_dataset_key(heros, other)

    # Baseline filter: HEROS in the HEROS file (it’s already standardized in your combined_heros... output)
    if "AlgorithmFamily" in heros.columns:
        heros_base = heros[heros["AlgorithmFamily"].astype(str).str.upper() == "HEROS"].copy()
    else:
        heros_base = heros.copy()

    # Comparators in combined_runs_long.csv
    # Adjust these if your "other" file uses different labels.
    comparators = [
        ("RIPPER", "RIPPER"),
        ("BioHEL", "BioHEL_noRPE"),
        ("BioHEL", "BioHEL_RPE"),
    ]

    # Determine which actual columns to use for each requested metric in each df
    map_heros = metric_column_mapping(heros_base)
    map_other = metric_column_mapping(other)

    # We will only test metrics resolvable in BOTH dataframes
    metrics_to_test: List[str] = []
    for m in METRICS:
        if map_heros.get(m) is not None and map_other.get(m) is not None:
            metrics_to_test.append(m)

    if not metrics_to_test:
        raise RuntimeError(
            "None of the requested metrics could be mapped in both files.\n"
            f"Requested: {METRICS}\n"
            f"HEROS mapping: {map_heros}\n"
            f"OTHER mapping: {map_other}"
        )

    # Use datasets present in both files (intersection)
    ds_common = sorted(set(heros_base[dataset_col].dropna().unique()) & set(other[dataset_col].dropna().unique()))
    if not ds_common:
        raise RuntimeError(f"No overlapping datasets found between files using column '{dataset_col}'.")

    rows = []
    for ds in ds_common:
        base_ds = heros_base[heros_base[dataset_col] == ds]

        for alg, scen in comparators:
            comp_ds = other[
                (other["AlgorithmFamily"].astype(str) == alg)
                & (other["Scenario"].astype(str) == scen)
                & (other[dataset_col] == ds)
            ]

            if comp_ds.empty or base_ds.empty:
                continue

            for metric in metrics_to_test:
                bcol = map_heros[metric]
                ccol = map_other[metric]
                assert bcol is not None and ccol is not None

                base_vals = extract_numeric(base_ds[bcol])
                comp_vals = extract_numeric(comp_ds[ccol])

                # Need at least 2 observations each for a meaningful rank-sum test
                if base_vals.size < 2 or comp_vals.size < 2:
                    continue

                u_stat, p_val = mann_whitney_two_sided(base_vals, comp_vals)

                base_med = float(np.median(base_vals))
                comp_med = float(np.median(comp_vals))
                med_diff = comp_med - base_med

                # Note: “better” depends on the metric. We only record direction; interpret accordingly.
                rows.append(
                    {
                        "Dataset": ds,
                        "Metric": metric,
                        "Baseline": "HEROS_default_model",
                        "ComparatorAlgorithm": alg,
                        "ComparatorScenario": scen,
                        # "n_baseline": int(base_vals.size),
                        # "n_comp": int(comp_vals.size),
                        "baseline_median": base_med,
                        "comp_median": comp_med,
                        "median_diff_comp_minus_base": float(med_diff),
                        # "cliffs_delta_comp_vs_base": float(cliffs_delta(base_vals, comp_vals)),
                        "u_statistic": u_stat,
                        "p_value": p_val,
                    }
                )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No tests were run (empty results). Check filters, dataset overlap, and metric mappings.")

    # Multiple-testing correction across ALL dataset×comparator×metric tests
    rej, p_adj, _, _ = multipletests(results["p_value"].to_numpy(), alpha=alpha, method="fdr_bh")
    results["p_value_fdr_bh"] = p_adj
    results[f"significant_fdr_bh_alpha_{alpha:g}"] = rej

    results["direction_by_median"] = np.where(
        results["median_diff_comp_minus_base"] > 0,
        "Comparator higher",
        np.where(results["median_diff_comp_minus_base"] < 0, "Comparator lower", "Tie (median)"),
    )

    # Stable sorting for readability
    results = results.sort_values(
        ["Metric", "ComparatorAlgorithm", "ComparatorScenario", "Dataset"],
        kind="mergesort",
    ).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    return results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Wilcoxon rank-sum (Mann–Whitney U) significance testing per dataset, "
            "using HEROS as baseline and comparing RIPPER/BioHEL scenarios.\n"
            "Only tests metrics: test_balanced_accuracy, test_coverage, rule_count, run_time."
        )
    )
    p.add_argument("--heros_csv", type=Path, default=Path(DEFAULT_HEROS_CSV), help="HEROS combined runs long CSV")
    p.add_argument("--other_csv", type=Path, default=Path(DEFAULT_OTHER_CSV), help="Combined runs long CSV (RIPPER/BioHEL)")
    p.add_argument("--out_csv", type=Path, default=Path(DEFAULT_OUT), help="Output CSV path")
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR-BH significance flag (default: 0.05)")
    return p


def main(argv: List[str]) -> int:
    args = build_argparser().parse_args(argv)
    res = run_tests(
        heros_csv=args.heros_csv,
        other_csv=args.other_csv,
        out_csv=args.out_csv,
        alpha=args.alpha,
    )
    print(f"Wrote: {args.out_csv}")
    print(f"Rows: {res.shape[0]} | Cols: {res.shape[1]}")
    print(f"Metrics tested (effective): {sorted(res['Metric'].unique().tolist())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
