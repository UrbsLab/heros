# run_experiments.py
from __future__ import annotations

import os
import re
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Paths / setup
# -----------------------------
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "biohel_data"
DATA_DIR.mkdir(exist_ok=True)


# -----------------------------
# Dataset configs
# -----------------------------
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "MUX6": {
        "train_path": "datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "6-bit Multiplexer",
        "category": "Multiplexer",
        "features": 6,
    },
    "MUX11": {
        "train_path": "datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "11-bit Multiplexer",
        "category": "Multiplexer",
        "features": 11,
    },
    "MUX20": {
        "train_path": "datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Train_1.txt",
        "test_path": "datasets/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Test_1.txt",
        "excluded_columns": ["Group", "InstanceID", "Class"],
        "description": "20-bit Multiplexer",
        "category": "Multiplexer",
        "features": 20,
    },
    "GAM_A": {
        "train_path": "datasets/gametes/A_uni_4add_CV_Train_1.txt",
        "test_path": "datasets/gametes/A_uni_4add_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "description": "GAMETES 4 Additive Univariate",
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_C": {
        "train_path": "datasets/gametes/C_2way_epistasis_CV_Train_1.txt",
        "test_path": "datasets/gametes/C_2way_epistasis_CV_Test_1.txt",
        "excluded_columns": ["Class"],
        "description": "GAMETES 2-way Epistasis",
        "category": "GAMETES",
        "features": 100,
    },
    "GAM_E": {
        "train_path": "datasets/gametes/E_uni_4het_CV_Train_1.txt",
        "test_path": "datasets/gametes/E_uni_4het_CV_Test_1.txt",
        "excluded_columns": ["Model", "InstanceID", "Class"],
        "description": "GAMETES 4 Heterogeneous Univariate",
        "category": "GAMETES",
        "features": 100,
    },
}

DATASETS = list(DATASET_CONFIGS.keys())


# -----------------------------
# Helpers
# -----------------------------
def load_dataset(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(config["train_path"], sep="\t")
    test_df = pd.read_csv(config["test_path"], sep="\t")

    feature_names = [c for c in train_df.columns if c not in config["excluded_columns"]]
    X_train = train_df[feature_names].values
    y_train = train_df["Class"].values
    X_test = test_df[feature_names].values
    y_test = test_df["Class"].values

    row_id = train_df["InstanceID"].values if "InstanceID" in train_df.columns else np.arange(len(train_df))
    return X_train, y_train, X_test, y_test, row_id, feature_names, train_df, test_df


def convert_to_arff(df: pd.DataFrame, feature_names: List[str], filename: Path, relation_name: str) -> None:
    filename = Path(filename)
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
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(config_content)


# -----------------------------
# BioHEL runner (no docker)
# -----------------------------
class BioHELRunner:
    def __init__(self, biohel_path: Path | str, data_dir: Path):
        self.biohel_path = str(biohel_path)
        self.data_dir = Path(data_dir)

    def _check_executable(self) -> None:
        p = Path(self.biohel_path)
        if p.exists() and not os.access(str(p), os.X_OK):
            raise RuntimeError(f"BioHEL exists but is not executable: {p}")

    def run_biohel(self, dataset_name: str) -> Dict[str, Any] | None:
        self._check_executable()

        dataset_dir = self.data_dir / dataset_name
        required = ["config.conf", "train.arff", "test.arff"]
        for f in required:
            if not (dataset_dir / f).exists():
                print(f"Missing {dataset_dir / f}")
                return None

        print(f"Running BioHEL on {dataset_name}...")
        start_time = time.time()

        cmd = [
            self.biohel_path,
            str(dataset_dir / "config.conf"),
            str(dataset_dir / "train.arff"),
            str(dataset_dir / "test.arff"),
        ]

        result = subprocess.run(
            cmd,
            cwd=str(dataset_dir),
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - start_time

        (dataset_dir / "biohel_stdout.txt").write_text(result.stdout)
        (dataset_dir / "biohel_stderr.txt").write_text(result.stderr)

        if result.returncode != 0:
            print(f"BioHEL failed for {dataset_name} (code {result.returncode}).")
            print(result.stderr[:2000])
            return None

        return self.parse_output(result.stdout, elapsed)

    def parse_output(self, output: str, wall_time: float) -> Dict[str, Any]:
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


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    # Set BioHEL binary location:
    BIOHEL_BIN = BASE_DIR / "biohel"  # change to absolute path if needed

    biohel_runner = BioHELRunner(BIOHEL_BIN, DATA_DIR)
    print(f"Using BioHEL binary: {BIOHEL_BIN}")

    # 1) Prepare datasets (ARFF/config)
    for dataset_name, config in DATASET_CONFIGS.items():
        print(f"\nPreparing: {dataset_name}")

        dataset_dir = DATA_DIR / dataset_name
        dataset_dir.mkdir(exist_ok=True)

        X_train, y_train, X_test, y_test, row_id, feature_names, train_df, test_df = load_dataset(config)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        convert_to_arff(train_df, feature_names, dataset_dir / "train.arff", f"{dataset_name}_Train")
        convert_to_arff(test_df, feature_names, dataset_dir / "test.arff", f"{dataset_name}_Test")
        create_biohel_config(dataset_dir / "config.conf")

    print("\nDataset preparation done.")

    # 2) Run BioHEL on all datasets
    biohel_results: Dict[str, Dict[str, Any]] = {}
    for dataset_name in DATASETS:
        print(f"\nBioHEL: {dataset_name}")
        result = biohel_runner.run_biohel(dataset_name)
        if result:
            biohel_results[dataset_name] = result
            ta = result["test_accuracy"]
            nr = result["num_rules"]
            rt = result["runtime"]
            if ta is not None and rt is not None:
                print(f"BioHEL Acc: {ta:.4f}, Rules: {nr}, Time: {rt:.1f}s")
            else:
                print(f"BioHEL completed; parsed metrics missing. Rules: {nr}")
        else:
            print("BioHEL failed.")

    print("\nBioHEL done")

    # 3) Build BioHEL-only results dataframe
    rows: List[Dict[str, Any]] = []
    for dataset_name, config in DATASET_CONFIGS.items():
        if dataset_name not in biohel_results:
            continue
        b = biohel_results[dataset_name]
        rows.append(
            {
                "Dataset": dataset_name,
                "Category": config["category"],
                "Features": config["features"],
                "Algorithm": "BioHEL",
                "Test Accuracy": b["test_accuracy"],
                "Train Accuracy": b["train_accuracy"],
                "Num Rules": b["num_rules"],
                "Training Time (s)": b["runtime"],
                "Wall Time (s)": b["wall_time"],
            }
        )

    results_df = pd.DataFrame(rows)

    # 4) Console summary
    print("\nBioHEL Results")
    print("-" * 90)
    print(f"{'Dataset':<10} {'Category':<12} {'Test Acc':<12} {'Train Acc':<12} {'Rules':<10} {'Time(s)':<10}")
    print("-" * 90)

    for dataset_name in DATASETS:
        config = DATASET_CONFIGS[dataset_name]
        sub = results_df[results_df["Dataset"] == dataset_name]
        if len(sub) == 0:
            print(f"{dataset_name:<10} {config['category']:<12} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
            continue
        r = sub.iloc[0]
        test_acc = f"{float(r['Test Accuracy']):.1%}" if pd.notna(r["Test Accuracy"]) else "N/A"
        train_acc = f"{float(r['Train Accuracy']):.1%}" if pd.notna(r["Train Accuracy"]) else "N/A"
        rules = int(r["Num Rules"]) if pd.notna(r["Num Rules"]) else "N/A"
        tsec = f"{float(r['Training Time (s)']):.1f}" if pd.notna(r["Training Time (s)"]) else "N/A"
        print(f"{dataset_name:<10} {config['category']:<12} {test_acc:<12} {train_acc:<12} {rules:<10} {tsec:<10}")

    # 5) Plots (BioHEL only)
    plot_dir = DATA_DIR / "comparison_plots"
    plot_dir.mkdir(exist_ok=True)

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DATASETS))
    accs = []
    for d in DATASETS:
        sub = results_df[results_df["Dataset"] == d]
        accs.append(float(sub["Test Accuracy"].values[0]) * 100 if len(sub) and pd.notna(sub["Test Accuracy"].values[0]) else 0.0)

    ax.bar(x, accs)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("BioHEL Test Accuracy by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(plot_dir / "biohel_accuracy.png", dpi=300)
    plt.close(fig)

    # Rules plot
    fig, ax = plt.subplots(figsize=(12, 6))
    rules_vals = []
    for d in DATASETS:
        sub = results_df[results_df["Dataset"] == d]
        rules_vals.append(int(sub["Num Rules"].values[0]) if len(sub) else 0)

    ax.bar(x, rules_vals)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Number of Rules")
    ax.set_title("BioHEL Number of Rules by Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    plt.tight_layout()
    plt.savefig(plot_dir / "biohel_rules.png", dpi=300)
    plt.close(fig)

    # 6) CSV-only outputs
    out_dir = DATA_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)

    results_csv = out_dir / "biohel_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved: {results_csv}")

    rules_dir = out_dir / "biohel_rules"
    rules_dir.mkdir(exist_ok=True)

    for dataset_name, results in biohel_results.items():
        rules = results.get("rules", [])
        if rules:
            rules_df = pd.DataFrame({"Rule": rules})
            rules_csv = rules_dir / f"{dataset_name}_biohel_rules.csv"
            rules_df.to_csv(rules_csv, index=False)
            print(f"Saved: {rules_csv}")

    # Optional: reproducibility/debug JSON (remove if you want strictly CSV+plots only)
    raw_json = out_dir / "biohel_raw_results.json"
    with open(raw_json, "w") as f:
        json.dump(biohel_results, f, indent=2)
    print(f"Saved: {raw_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
