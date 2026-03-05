#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate BioHEL per-dataset outputs into CSV + plots")
    p.add_argument("--project", required=True)
    p.add_argument("--out-path", required=True)
    args = p.parse_args()

    base = Path(args.out_path) / args.project
    results_root = base / "results"
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # collect per-dataset result_row.csv files
    rows: List[pd.DataFrame] = []
    for ds_dir in sorted(results_root.glob("*")):
        if not ds_dir.is_dir():
            continue
        row_csv = ds_dir / "result_row.csv"
        if row_csv.exists():
            rows.append(pd.read_csv(row_csv))

    if not rows:
        raise RuntimeError(f"No result_row.csv files found under: {results_root}")

    results_df = pd.concat(rows, ignore_index=True)

    # outputs/
    out_dir = base / "results_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "biohel_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved: {results_csv}")

    # plots
    datasets = results_df["Dataset"].tolist()
    x = np.arange(len(datasets))

    # accuracy plot
    fig, ax = plt.subplots(figsize=(12, 6))
    acc = results_df["Test Accuracy"].fillna(0).astype(float) * 100.0
    ax.bar(x, acc)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("BioHEL Test Accuracy by Dataset")
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(figures / "biohel_accuracy.png", dpi=300)
    plt.close(fig)

    # rules plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rules = results_df["Num Rules"].fillna(0).astype(int)
    ax.bar(x, rules)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=0)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Rules")
    ax.set_title("BioHEL Number of Rules by Dataset")
    plt.tight_layout()
    plt.savefig(figures / "biohel_rules.png", dpi=300)
    plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
