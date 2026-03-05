#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("/project/kamoun_shared/gabe/I2C2-Documentation/output")
DEFAULT_OUTDIR = Path("/project/kamoun_shared/output_shared/paper_tables")
DEFAULT_OUTNAME = "combined_heros_cv_default_runs_long.csv"


def is_cv_default(path: Path) -> bool:
    """
    Keep ONLY cv_default runs.
    Exclude cv_equal, tree_init, fb_tree_init, etc.
    """
    p = str(path)
    return ("cv_default" in p) and ("cv_equal" not in p) and ("tree_init" not in p)


def parse_metadata(csv_path: Path) -> dict:
    """
    Extract metadata from directory structure:
      .../HEROS_mux_cv_default/A_multiplexer_6_bit_500_inst/all_default_model_200_evaluations.csv
    """
    dataset_dir = csv_path.parent.name               # A_multiplexer_6_bit_500_inst
    run_group_dir = csv_path.parent.parent.name      # HEROS_mux_cv_default

    alg = "HEROS"
    sim = "MUX" if "mux" in run_group_dir.lower() else "GAMETES"
    cv_scheme = "cv_default"
    rep = dataset_dir.split("_")[0]                  # A / B / C / ...

    return {
        "AlgorithmFamily": alg,
        "Simulation": sim,
        "CVScheme": cv_scheme,
        "Dataset": dataset_dir,
        "SimulationRep": rep,
        "Scenario": "default_model",
    }


def combine_cv_default_runs(root: Path) -> pd.DataFrame:
    csvs = [p for p in root.rglob("all_default_model_200_evaluations.csv") if is_cv_default(p)]
    if not csvs:
        raise RuntimeError(
            f"No cv_default evaluation files found under: {root}\n"
            "Expected files named: all_default_model_200_evaluations.csv"
        )

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(csvs):
        meta = parse_metadata(csv_path)
        df = pd.read_csv(csv_path)

        # Insert identifiers up front
        df.insert(0, "_source_csv", str(csv_path))
        for k, v in meta.items():
            df.insert(0, k, v)

        # Seed surrogate (row index) to mirror long-runs-style outputs
        df.insert(0, "seed", np.arange(len(df), dtype=int))

        # Normalize columns to long-runs style (best-effort)
        if "run_time" in df.columns:
            df["runtime"] = df["run_time"]
            df["final_runtime"] = df["run_time"]
            df["run_time_minutes"] = df["run_time"] / 60.0
        else:
            df["runtime"] = np.nan
            df["final_runtime"] = np.nan
            df["run_time_minutes"] = np.nan

        if "test_balanced_accuracy" in df.columns:
            df["test_accuracy"] = df["test_balanced_accuracy"]

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Recursively combine HEROS cv_default evaluation CSVs named "
            "'all_default_model_200_evaluations.csv' into a single long-runs-style CSV."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory to search recursively (default: {DEFAULT_ROOT})",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--outname",
        type=str,
        default=DEFAULT_OUTNAME,
        help=f"Output CSV filename (default: {DEFAULT_OUTNAME})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output; only print discovered file count and sample paths.",
    )
    return p


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv)

    root: Path = args.root
    outdir: Path = args.outdir
    outpath: Path = outdir / args.outname

    # Find first (for dry-run info too)
    csvs = [p for p in root.rglob("all_default_model_200_evaluations.csv") if is_cv_default(p)]
    if not csvs:
        print(
            f"ERROR: No cv_default evaluation files found under: {root}\n"
            "Expected files named: all_default_model_200_evaluations.csv",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(csvs)} cv_default files under: {root}")
    for pth in sorted(csvs)[:5]:
        print(f"  - {pth}")
    if len(csvs) > 5:
        print("  ...")

    if args.dry_run:
        print("Dry-run enabled; not writing output.")
        return 0

    combined = combine_cv_default_runs(root)

    outdir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(outpath, index=False)

    print(f"Wrote: {outpath}")
    print(f"Rows: {combined.shape[0]} | Cols: {combined.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
