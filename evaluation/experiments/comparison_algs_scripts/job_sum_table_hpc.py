#!/usr/bin/env python3
"""
Summarize BioHEL + RIPPER from ONLY TWO folder inputs, using metrics.json as source of truth.

Key requirement implemented:
- If BioHEL metrics.json contains BOTH "raw" and "final":
    * include TWO rows:
        - BioHEL_noRPE from "raw" (matches result_row_raw.csv semantics)
        - BioHEL_RPE   from "final" (matches result_row.csv semantics)
- If BioHEL metrics.json is flat (no raw/final):
    * include ONE row: BioHEL_noRPE
- If BioHEL metrics.json has only "final" (edge case):
    * include ONE row:
        - BioHEL_RPE if postprocess_* present
        - else BioHEL_noRPE

RPE runtime rule:
  final_runtime_seconds = wall_time + postprocess_wall_time 
(preprocess_time is extracted from metrics.json if present under known keys; else 0.)

RIPPER metrics.json is flat:
  include ONE row: RIPPER (final_runtime_seconds = wall_time)

Outputs:
  <out>/combined_runs_long.csv
  <out>/summary_by_dataset.csv
  <out>/summary_multiplexer_only.csv      (Ideal Solution k/n for MUX datasets)
  <out>/summary_gametes_only.csv
  
Run:
  python job_sum_table_hpc.py \
    --biohel_root /project/kamoun_shared/output_shared/biohel_gecco \
    --ripper_root /project/kamoun_shared/output_shared/ripper_gecco \
    --out /project/kamoun_shared/output_shared/paper_tables
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Multiplexer ideal rule counts
# -----------------------------
IDEAL_COUNTS: Dict[str, int] = {
    "A_multiplexer_6_bit_500_inst": 5,
    "B_multiplexer_11_bit_5000_inst": 9,
    "C_multiplexer_20_bit_10000_inst": 17,
    "D_multiplexer_37_bit_10000_inst": 33,
    "E_multiplexer_70_bit_20000_inst": 65,
}
IDEAL_ACC = 1.0


# -----------------------------
# Dataset name inference
# -----------------------------
CV_TRAIN_RE = re.compile(r"([^/]+)_CV_Train_\d+\.txt$")


def infer_dataset_name(train_file: str) -> str:
    if isinstance(train_file, str):
        m = CV_TRAIN_RE.search(train_file)
        if m:
            return m.group(1)
        parts = os.path.normpath(train_file).split(os.sep)
        if len(parts) >= 2:
            return parts[-2]
    return "UNKNOWN"


# -----------------------------
# File discovery / JSON parsing
# -----------------------------
def find_files(root: str, filename: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            out.append(os.path.join(dirpath, filename))
    return out


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def coerce_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def coerce_int(x: Any) -> Optional[int]:
    try:
        return int(float(x))
    except Exception:
        return None


def is_rpe_final_block(d: Dict[str, Any]) -> bool:
    return ("postprocess_wall_time" in d) or ("postprocess_num_rules" in d)


# -----------------------------
# Long-form row construction
# -----------------------------
def _make_row(
    algorithm_family: str,
    scenario: str,
    metrics_json_path: str,
    metrics_obj: Dict[str, Any],
    block: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Construct one normalized long-form row from a block dict.
    scenario determines runtime handling (BioHEL_RPE adds postprocess).
    """
    runtime = coerce_float(block.get("runtime"))
    postprocess = coerce_float(block.get("postprocess_wall_time")) if scenario == "BioHEL_RPE" else None

    final_runtime = None
    if runtime is not None:
        final_runtime = coerce_float(block.get("wall_time")) + float(postprocess or 0.0)

    # rule_count: for RPE prefer postprocess_num_rules if available
    if scenario == "BioHEL_RPE" and ("postprocess_num_rules" in block):
        rule_count = coerce_int(block.get("postprocess_num_rules"))
    else:
        rule_count = coerce_int(block.get("num_rules"))

    train_file = block.get("train_file")
    dataset = infer_dataset_name(train_file) if isinstance(train_file, str) else "UNKNOWN"

    return {
        "AlgorithmFamily": algorithm_family,
        "Scenario": scenario,
        "Dataset": dataset,
        "seed": coerce_int(block.get("seed")),
        "train_file": block.get("train_file"),
        "test_file": block.get("test_file"),
        "train_accuracy": coerce_float(block.get("train_accuracy")),
        "test_accuracy": coerce_float(block.get("test_accuracy")),
        "train_coverage": coerce_float(block.get("train_coverage")),
        "test_coverage": coerce_float(block.get("test_coverage")),
        "train_default_rate": coerce_float(block.get("train_default_rate")),
        "test_default_rate": coerce_float(block.get("test_default_rate")),
        "num_rules": coerce_int(block.get("num_rules")),
        "rule_count": rule_count,
        "runtime": runtime,
        "wall_time": coerce_float(block.get("wall_time")),
        "postprocess_wall_time": coerce_float(block.get("postprocess_wall_time")),
        "postprocess_num_rules": coerce_int(block.get("postprocess_num_rules")),
        "final_runtime": final_runtime,
        "run_time_minutes": (final_runtime / 60.0) if final_runtime is not None else None,
        "_source_json": metrics_json_path,
        "_source_dir": os.path.dirname(metrics_json_path),
    }


def build_long_rows_from_biohel(biohel_root: str) -> Tuple[pd.DataFrame, int]:
    """
    BioHEL:
      - If both "raw" and "final" exist -> create two rows (noRPE from raw, RPE from final).
      - Else if flat dict -> one row noRPE.
      - Else if only "final" -> one row, scenario depends on postprocess_* presence.
      - Else if only "raw" -> one row noRPE.

    """
    paths = find_files(biohel_root, "metrics.json")
    rows: List[Dict[str, Any]] = []

    for p in paths:
        obj = load_json(p)
        if obj is None or not isinstance(obj, dict):
            continue

        has_raw = isinstance(obj.get("raw"), dict)
        has_final = isinstance(obj.get("final"), dict)

        if has_raw and has_final:
            # Always include BOTH:
            # - raw as BioHEL_noRPE
            rows.append(_make_row("BioHEL", "BioHEL_noRPE", p, obj, obj["raw"]))
            # - final as BioHEL_RPE (even if postprocess keys absent, per your rule final == RPE output)
            rows.append(_make_row("BioHEL", "BioHEL_RPE", p, obj, obj["final"]))

        # Only final
        if has_final and not has_raw:
            final_block = obj["final"]
            scenario = "BioHEL_RPE" if is_rpe_final_block(final_block) else "BioHEL_noRPE"
            rows.append(_make_row("BioHEL", scenario, p, obj, final_block))

        # Only raw
        if has_raw and not has_final:
            rows.append(_make_row("BioHEL", "BioHEL_noRPE", p, obj, obj["raw"]))
            continue

        # Flat dict (no raw/final): treat as noRPE
        if "train_file" in obj and "runtime" in obj:
            rows.append(_make_row("BioHEL", "BioHEL_noRPE", p, obj, obj))
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_long_rows_from_ripper(ripper_root: str) -> pd.DataFrame:
    """
    RIPPER metrics.json is flat; no raw/final.
    """
    paths = find_files(ripper_root, "metrics.json")
    rows: List[Dict[str, Any]] = []

    for p in paths:
        obj = load_json(p)
        if obj is None or not isinstance(obj, dict):
            continue
        if "train_file" not in obj or "runtime" not in obj:
            continue

        rows.append(_make_row("RIPPER", "RIPPER", p, obj, obj))

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# -----------------------------
# Summary tables
# -----------------------------
def ideal_solution_k_over_n(sub: pd.DataFrame, dataset: str, scenario:str) -> str:
    if dataset not in IDEAL_COUNTS:
        return ""
    if sub.empty:
        return "0/0"
    acc = pd.to_numeric(sub["test_accuracy"], errors="coerce")
    rules = pd.to_numeric(sub["rule_count"], errors="coerce")
    n = int(sub.shape[0])
    ideal_count = IDEAL_COUNTS[dataset] if scenario != "RIPPER" else IDEAL_COUNTS[dataset] - 1
    k = int(((acc == IDEAL_ACC) & (rules == ideal_count)).sum())
    return f"{k}/{n}"


def fmt_mean_std(series: pd.Series, digits: Optional[int]) -> str:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "NA"
    mean = float(s.mean())
    std = float(s.std())
    if digits is None:
        if float(mean).is_integer() and float(std).is_integer():
            return f"{int(mean)} ({int(std)})"
        return f"{mean} ({std})"
    return f"{round(mean, digits)} ({round(std, digits)})"


def build_summary(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["Dataset", "Scenario", "Test Acc.", "Test Cover", "Rule Count", "Run Time", "Ideal Solution"])

    rows = []
    for (dataset, scenario), g in df_long.groupby(["Dataset", "Scenario"], dropna=False):
        rows.append({
            "Dataset": dataset,
            "Scenario": scenario,
            "Test Acc.": fmt_mean_std(g["test_accuracy"], 3),
            "Test Cover": fmt_mean_std(g["test_coverage"], 3),
            "Rule Count": fmt_mean_std(g["rule_count"], 3),
            "Run Time": fmt_mean_std(g["run_time_minutes"], 3),
            "Ideal Solution": ideal_solution_k_over_n(g, dataset, scenario),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["Dataset", "Scenario"], kind="mergesort")
        .reset_index(drop=True)
    )


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--biohel_root", required=True, type=str, help="Parent folder containing BioHEL no-RPE and RPE outputs.")
    ap.add_argument("--ripper_root", required=True, type=str, help="Root folder for RIPPER outputs.")
    ap.add_argument("--out", required=True, type=str, help="Output folder for summary CSVs.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    biohel_df = build_long_rows_from_biohel(args.biohel_root)
    ripper_df = build_long_rows_from_ripper(args.ripper_root)
    combined = pd.concat([biohel_df, ripper_df], axis=0, ignore_index=True, sort=False)

    long_cols = [
        "AlgorithmFamily", "Scenario", "Dataset",
        "seed", "train_file", "test_file",
        "test_accuracy", "test_coverage",
        "rule_count",
        "runtime", "postprocess_wall_time", "preprocess_time", "final_runtime", "run_time_minutes",
        "_source_json",
    ]
    long_out = combined[[c for c in long_cols if c in combined.columns]].copy()
    long_path = os.path.join(args.out, "combined_runs_long.csv")
    long_out.to_csv(long_path, index=False)

    summary = build_summary(combined)
    summary_path = os.path.join(args.out, "summary_by_dataset.csv")
    summary.to_csv(summary_path, index=False)

    mux = summary[summary["Dataset"].isin(IDEAL_COUNTS.keys())].copy()
    nonmux = summary[~summary["Dataset"].isin(IDEAL_COUNTS.keys())].copy()

    mux_path = os.path.join(args.out, "summary_multiplexer_only.csv")
    nonmux_path = os.path.join(args.out, "summary_gametes_only.csv")
    mux.to_csv(mux_path, index=False)
    nonmux.to_csv(nonmux_path, index=False)

    print(f"[OK] Wrote: {long_path}")
    print(f"[OK] Wrote: {summary_path}")
    print(f"[OK] Wrote: {mux_path}")
    print(f"[OK] Wrote: {nonmux_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
