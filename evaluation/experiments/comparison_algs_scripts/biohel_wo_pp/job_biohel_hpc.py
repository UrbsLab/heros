#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import re
import json
import time
import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Data + ARFF helpers
# -----------------------------
def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def build_feature_names(df: pd.DataFrame, outcome_label: str, instanceID_label: str, excluded_column: str) -> List[str]:
    cols = list(df.columns)
    for c in [excluded_column, instanceID_label]:
        if c in cols:
            cols.remove(c)
    if outcome_label in cols:
        cols.remove(outcome_label)
    return cols


def convert_to_arff(df: pd.DataFrame, feature_names: List[str], outcome_label: str, filename: Path, relation_name: str) -> None:
    """
    Writes a simple discrete ARFF.
    Assumes all features + class are categorical integer-like values.
    """
    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            unique_values = sorted(df[feat].unique())
            value_str = ",".join(map(str, [int(v) for v in unique_values]))
            f.write(f"@ATTRIBUTE {feat} {{{value_str}}}\n")

        class_values = sorted(df[outcome_label].unique())
        class_str = ",".join(map(str, [int(v) for v in class_values]))
        f.write(f"@ATTRIBUTE {outcome_label} {{{class_str}}}\n\n")

        f.write("@DATA\n")
        for _, row in df.iterrows():
            vals = [str(int(row[feat])) for feat in feature_names]
            vals.append(str(int(row[outcome_label])))
            f.write(",".join(vals) + "\n")


def create_biohel_config(output_path: Path, seed: int) -> None:
    """
    Baseline config (your earlier one) + inject random seed.
    Adjust any values here as needed; this is the canonical single source.
    """
    config_content = f"""crossover operator 1px
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
random seed {seed}
"""
    output_path.write_text(config_content)


# -----------------------------
# Rule parsing + coverage
# -----------------------------
@dataclass(frozen=True)
class ParsedRule:
    rule_id: int
    conditions: Dict[str, int]   # {"A_0": 1, ...}
    prediction: int
    is_default: bool = False


_RULE_HEADER_RE = re.compile(r"^\s*(\d+)\s*:(.*)$")
_COND_RE = re.compile(r"Att\s+([A-Za-z0-9_]+)\s+is\s+([-+]?\d+)")
_DEFAULT_RE = re.compile(r"Default\s+rule\s*->\s*([-+]?\d+)", re.IGNORECASE)


def parse_biohel_rule_line(line: str) -> Optional[ParsedRule]:
    line = line.strip()
    if not line:
        return None

    # Default rule (may or may not have "12:" prefix)
    mdef = _DEFAULT_RE.search(line)
    if mdef:
        mid = _RULE_HEADER_RE.match(line)
        rule_id = int(mid.group(1)) if mid else -1
        pred = int(mdef.group(1))
        return ParsedRule(rule_id=rule_id, conditions={}, prediction=pred, is_default=True)

    m = _RULE_HEADER_RE.match(line)
    if not m:
        return None

    rule_id = int(m.group(1))
    rhs = m.group(2).strip()

    parts = [p.strip() for p in rhs.split("|") if p.strip()]
    if not parts:
        return None

    try:
        pred = int(parts[-1])
    except ValueError:
        return None

    cond_text = "|".join(parts[:-1])
    conds: Dict[str, int] = {}
    for feat, val in _COND_RE.findall(cond_text):
        conds[feat] = int(val)

    return ParsedRule(rule_id=rule_id, conditions=conds, prediction=pred, is_default=False)


def parse_biohel_rules(rule_lines: List[str]) -> Tuple[List[ParsedRule], Optional[ParsedRule]]:
    rules: List[ParsedRule] = []
    default_rule: Optional[ParsedRule] = None

    for ln in rule_lines:
        pr = parse_biohel_rule_line(ln)
        if pr is None:
            continue
        if pr.is_default:
            default_rule = pr
        else:
            rules.append(pr)

    rules.sort(key=lambda r: r.rule_id)
    return rules, default_rule


def compute_rule_coverage(df: pd.DataFrame, rules: List[ParsedRule], excluded_columns: List[str]) -> Dict[str, float]:
    """
    Coverage = fraction matched by at least one NON-default rule.
    Default fall-through rate = 1 - coverage.
    """
    X = df.drop(columns=[c for c in excluded_columns if c in df.columns], errors="ignore")
    n = len(X)
    if n == 0:
        return {"coverage": float("nan"), "default_rate": float("nan")}

    matched_any = np.zeros(n, dtype=bool)

    for r in rules:
        if not r.conditions:
            matched_any |= True
            continue

        mask = np.ones(n, dtype=bool)
        for feat, val in r.conditions.items():
            if feat not in X.columns:
                mask &= False
                break
            mask &= (X[feat].astype(int).values == int(val))

        matched_any |= mask
        if matched_any.all():
            break

    coverage = float(matched_any.mean())
    return {"coverage": coverage, "default_rate": float(1.0 - coverage)}


def per_rule_match_counts(df: pd.DataFrame, rules: List[ParsedRule], excluded_columns: List[str]) -> pd.DataFrame:
    X = df.drop(columns=[c for c in excluded_columns if c in df.columns], errors="ignore")
    n = len(X)
    rows = []
    for r in rules:
        mask = np.ones(n, dtype=bool)
        for feat, val in r.conditions.items():
            if feat not in X.columns:
                mask &= False
                break
            mask &= (X[feat].astype(int).values == int(val))
        rows.append(
            {
                "rule_id": r.rule_id,
                "prediction": r.prediction,
                "n_conditions": len(r.conditions),
                "n_matched": int(mask.sum()),
                "frac_matched": float(mask.mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("rule_id")


# -----------------------------
# BioHEL stdout parsing
# -----------------------------
def extract_phenotype_rules(stdout: str) -> List[str]:
    """
    Extract rule lines between 'Phenotype:' and the first blank line / Train line.
    Matches your earlier extractor but slightly more defensive.
    """
    lines = stdout.splitlines()
    rules: List[str] = []
    in_pheno = False
    for ln in lines:
        if ln.startswith("Phenotype:"):
            in_pheno = True
            continue
        if in_pheno:
            if not ln.strip():
                break
            if ln.startswith("Train"):
                break
            rules.append(ln.rstrip())
    return rules


def parse_biohel_metrics(stdout: str, wall_time: float) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "wall_time": float(wall_time),
        "train_accuracy": None,
        "test_accuracy": None,
        "runtime": None,
        "num_rules": 0,
        "rules": [],
    }

    for line in stdout.splitlines():
        if "Train accuracy :" in line:
            m = re.search(r"Train accuracy\s*:\s*([\d.]+)", line)
            if m:
                metrics["train_accuracy"] = float(m.group(1))
        if "Test accuracy :" in line:
            m = re.search(r"Test accuracy\s*:\s*([\d.]+)", line)
            if m:
                metrics["test_accuracy"] = float(m.group(1))
        if "Total time:" in line:
            m = re.search(r"Total time:\s*([\d.]+)", line)
            if m:
                metrics["runtime"] = float(m.group(1))

    rule_lines = extract_phenotype_rules(stdout)
    metrics["rules"] = [r.strip() for r in rule_lines if r.strip()]
    metrics["num_rules"] = sum(1 for r in metrics["rules"] if "Default rule" not in r)

    return metrics


# -----------------------------
# Main
# -----------------------------
def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL HPC Job: one CV fold + one seed")

    parser.add_argument("--d", dest="full_data_path", type=str, required=True,
                        help="Path to training fold: <dataset>_CV_Train_k.txt")
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Output directory for this fold")

    parser.add_argument("--biohel", dest="biohel_bin", type=str, required=True,
                        help="BioHEL binary path or command on PATH")

    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    parser.add_argument("--rs", dest="random_state", type=int, default=42)
    parser.add_argument("--v", dest="verbose", action="store_true")

    opts = parser.parse_args(argv[1:])

    full_data_path = opts.full_data_path
    outputPath = Path(opts.outputPath)
    outputPath.mkdir(parents=True, exist_ok=True)

    biohel_bin = opts.biohel_bin
    outcome_label = opts.outcome_label
    instanceID_label = opts.instanceID_label
    excluded_column = opts.excluded_column
    seed = int(opts.random_state)

    test_data_path = full_data_path.replace("Train", "Test")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Expected test fold at: {test_data_path}")

    # Load data
    train_df = load_df(full_data_path)
    test_df = load_df(test_data_path)

    # Build ARFF + config
    feature_names = build_feature_names(train_df, outcome_label, instanceID_label, excluded_column)

    train_arff = outputPath / "train.arff"
    test_arff = outputPath / "test.arff"
    conf_path = outputPath / "config.conf"

    convert_to_arff(train_df, feature_names, outcome_label, train_arff, relation_name="Train")
    convert_to_arff(test_df, feature_names, outcome_label, test_arff, relation_name="Test")
    create_biohel_config(conf_path, seed=seed)

    # Run BioHEL
    cmd = [biohel_bin, str(conf_path), str(train_arff), str(test_arff)]
    start = time.time()
    proc = subprocess.run(cmd, cwd=str(outputPath), capture_output=True, text=True)
    wall_time = time.time() - start

    (outputPath / "biohel_stdout.txt").write_text(proc.stdout)
    (outputPath / "biohel_stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"BioHEL failed (code {proc.returncode}). See biohel_stderr.txt")

    metrics = parse_biohel_metrics(proc.stdout, wall_time)

    # Persist exact rule block
    raw_rules = metrics.get("rules", [])
    (outputPath / "rules_raw.txt").write_text("\n".join(raw_rules) + ("\n" if raw_rules else ""))

    # Save rules.csv (raw strings)
    if raw_rules:
        pd.DataFrame({"Rule": raw_rules}).to_csv(outputPath / "rules.csv", index=False)

    # Parse rules + compute coverage
    rules, default_rule = parse_biohel_rules(raw_rules)
    excluded_cols = [excluded_column, instanceID_label, outcome_label]

    cov_train = compute_rule_coverage(train_df, rules, excluded_cols)
    cov_test = compute_rule_coverage(test_df, rules, excluded_cols)

    metrics["train_coverage"] = cov_train["coverage"]
    metrics["test_coverage"] = cov_test["coverage"]
    metrics["train_default_rate"] = cov_train["default_rate"]
    metrics["test_default_rate"] = cov_test["default_rate"]

    # Save parsed rules as a structured table
    if rules or default_rule:
        rows = []
        for r in rules:
            rows.append(
                {
                    "rule_id": r.rule_id,
                    "is_default": False,
                    "prediction": r.prediction,
                    "conditions_json": json.dumps(r.conditions, sort_keys=True),
                    "n_conditions": len(r.conditions),
                }
            )
        if default_rule is not None:
            rows.append(
                {
                    "rule_id": default_rule.rule_id,
                    "is_default": True,
                    "prediction": default_rule.prediction,
                    "conditions_json": json.dumps({}, sort_keys=True),
                    "n_conditions": 0,
                }
            )
        pd.DataFrame(rows).sort_values(["is_default", "rule_id"]).to_csv(outputPath / "rules_parsed.csv", index=False)

    # Save per-rule match diagnostics
    if rules:
        per_rule_match_counts(train_df, rules, excluded_cols).to_csv(outputPath / "train_rule_match_counts.csv", index=False)
        per_rule_match_counts(test_df, rules, excluded_cols).to_csv(outputPath / "test_rule_match_counts.csv", index=False)

    # Save metrics.json
    with open(outputPath / "metrics.json", "w") as f:
        json.dump(
            {
                "train_file": full_data_path,
                "test_file": test_data_path,
                "seed": seed,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    # Sentinel row CSV (used by main/check/resub + summary)
    row = {
        "train_file": full_data_path,
        "test_file": test_data_path,
        "seed": seed,
        "train_accuracy": metrics.get("train_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
        "train_coverage": metrics.get("train_coverage"),
        "test_coverage": metrics.get("test_coverage"),
        "train_default_rate": metrics.get("train_default_rate"),
        "test_default_rate": metrics.get("test_default_rate"),
        "num_rules": metrics.get("num_rules"),
        "runtime": metrics.get("runtime"),
        "wall_time": metrics.get("wall_time"),
    }
    pd.DataFrame([row]).to_csv(outputPath / "result_row.csv", index=False)

    if opts.verbose:
        print(
            "Done.",
            f"seed={seed}",
            f"test_acc={row['test_accuracy']}",
            f"test_cov={row['test_coverage']}",
            f"rules={row['num_rules']}",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
