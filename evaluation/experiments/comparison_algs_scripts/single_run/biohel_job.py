#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MUX6": {
        "train_path": "../datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 6,
    },
    "MUX11": {
        "train_path": "../datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 11,
    },
    "MUX20": {
        "train_path": "../datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Train_1.txt",
        "test_path": "../datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "category": "Multiplexer",
        "features": 20,
    },
    "GAM_A": {
        "train_path": "../datasets/gametes/A_uni_4add_CV_Train_1.txt",
        "test_path": "../datasets/gametes/A_uni_4add_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_C": {
        "train_path": "../datasets/gametes/C_2way_epistasis_CV_Train_1.txt",
        "test_path": "../datasets/gametes/C_2way_epistasis_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_E": {
        "train_path": "../datasets/gametes/E_uni_4het_CV_Train_1.txt",
        "test_path": "../datasets/gametes/E_uni_4het_CV_Test_1.txt",
        "excluded_columns": ["Model", "InstanceID", "Class"],
        "category": "GAMETES",
        "features": 100,
    },
}


def load_dataset(config: Dict[str, Any]) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config["train_path"], sep="\t")
    test_df = pd.read_csv(config["test_path"], sep="\t")
    feature_names = [c for c in train_df.columns if c not in config["excluded_columns"]]
    return feature_names, train_df, test_df


def convert_to_arff(df: pd.DataFrame, feature_names: List[str], filename: Path, relation_name: str) -> None:
    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            unique_values = sorted(df[feat].unique())
            value_str = ",".join(map(str, [int(v) for v in unique_values]))
            f.write(f"@ATTRIBUTE {feat} {{{value_str}}}\n")

        class_values = sorted(df["Class"].unique())
        class_str = ",".join(map(str, [int(v) for v in class_values]))
        f.write(f"@ATTRIBUTE Class {{{class_str}}}\n\n")

        f.write("@DATA\n")
        for _, row in df.iterrows():
            values = [str(int(row[feat])) for feat in feature_names]
            values.append(str(int(row["Class"])))
            f.write(",".join(values) + "\n")


def create_biohel_config(output_path: Path) -> None:
    config_content = """crossover operator 1px
default class major
fitness function mdl
initialization min classifiers 20
initialization max classifiers 20
iterations 50
mdl initial tl ratio 0.25
mdl iteration 10
mdl weight relax factor 0.90
pop size 500
prob crossover 0.6
prob individual mutation 0.6
prob one 0.75
selection algorithm tournamentwor
tournament size 4
windowing ilas 1
dump evolution stats
smart init
class wise init
coverage breakpoint 0.01
repetitions of rule learning 2
coverage ratio 0.90
kr hyperrect
num expressed attributes init 15
hyperrectangle uses list of attributes
prob generalize list 0.10
prob specialize list 0.10
expected number of attributes 10
random seed 42
"""
    with open(output_path, "w") as f:
        f.write(config_content)


def parse_biohel_output(output: str, wall_time: float) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "wall_time": float(wall_time),
        "test_accuracy": None,
        "train_accuracy": None,
        "num_rules": 0,
        "runtime": None,
        "rules": [],
    }

    for line in output.splitlines():
        if "Train accuracy :" in line:
            m = re.search(r"Train accuracy\s*:\s*([\d.]+)", line)
            if m:
                results["train_accuracy"] = float(m.group(1))
        if "Test accuracy :" in line:
            m = re.search(r"Test accuracy\s*:\s*([\d.]+)", line)
            if m:
                results["test_accuracy"] = float(m.group(1))
        if "Total time:" in line:
            m = re.search(r"Total time:\s*([\d.]+)", line)
            if m:
                results["runtime"] = float(m.group(1))

    in_phenotype = False
    for line in output.splitlines():
        if line.startswith("Phenotype:"):
            in_phenotype = True
            continue
        if in_phenotype:
            if line.strip() and not line.startswith("Train"):
                results["rules"].append(line.strip())
                results["num_rules"] += 1
            else:
                break

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="BioHELJob: run one dataset and write outputs deterministically")
    p.add_argument("--project", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS.keys()))
    p.add_argument("--biohel-bin", default="./biohel", help="BioHEL binary path or command")
    args = p.parse_args()

    base = Path(args.out_path) / args.project
    ds = args.dataset
    cfg = DATASET_CONFIGS[ds]

    # results for this dataset
    ds_dir = base / "results" / ds
    ds_dir.mkdir(parents=True, exist_ok=True)

    # deterministic input artifacts for this dataset
    train_arff = ds_dir / "train.arff"
    test_arff = ds_dir / "test.arff"
    conf_path = ds_dir / "config.conf"

    # 1) Prepare inputs (ARFF + config) inside the Job so each job is independent
    feature_names, train_df, test_df = load_dataset(cfg)
    convert_to_arff(train_df, feature_names, train_arff, f"{ds}_Train")
    convert_to_arff(test_df, feature_names, test_arff, f"{ds}_Test")
    create_biohel_config(conf_path)

    # 2) Run BioHEL
    cmd = [args.biohel_bin, str(conf_path), str(train_arff), str(test_arff)]

    start = time.time()
    print(f"Running BioHEL for dataset {ds}...")
    print(f"Command: {' '.join(cmd)}")
    # proc = subprocess.run(cmd, cwd=str(ds_dir), capture_output=True, text=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.time() - start

    # always write raw logs for debugging
    (ds_dir / "biohel_stdout.txt").write_text(proc.stdout)
    (ds_dir / "biohel_stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        # fail loudly: scheduler will mark job failed; stderr is preserved
        raise RuntimeError(f"BioHEL failed for {ds} (code {proc.returncode}). See biohel_stderr.txt")

    # 3) Parse + write metrics
    metrics = parse_biohel_output(proc.stdout, wall_time)

    # write JSON metrics
    with open(ds_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "dataset": ds,
                "category": cfg["category"],
                "features": cfg["features"],
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # write rules CSV (if any)
    rules = metrics.get("rules", [])
    if rules:
        pd.DataFrame({"Rule": rules}).to_csv(ds_dir / "rules.csv", index=False)

    # write a single-row CSV for easy aggregation
    row = {
        "Dataset": ds,
        "Category": cfg["category"],
        "Features": cfg["features"],
        "Algorithm": "BioHEL",
        "Test Accuracy": metrics.get("test_accuracy"),
        "Train Accuracy": metrics.get("train_accuracy"),
        "Num Rules": metrics.get("num_rules"),
        "Training Time (s)": metrics.get("runtime"),
        "Wall Time (s)": metrics.get("wall_time"),
    }
    pd.DataFrame([row]).to_csv(ds_dir / "result_row.csv", index=False)

    print(f"Completed {ds}. Test Acc={metrics.get('test_accuracy')} Rules={metrics.get('num_rules')}")


if __name__ == "__main__":
    main()
