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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# ARFF + config
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


def infer_arff_schema(train_df: pd.DataFrame, feature_names: List[str], outcome_label: str) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Infer ARFF nominal domains from TRAIN ONLY (prevents test peeking and guarantees consistent schema).
    Returns:
      feat_domains: dict feat -> sorted list of allowed int values
      class_domain: sorted list of allowed int class values
    """
    feat_domains: Dict[str, List[int]] = {}
    for feat in feature_names:
        vals = sorted(pd.Series(train_df[feat]).dropna().astype(int).unique().tolist())
        feat_domains[feat] = vals

    class_domain = sorted(pd.Series(train_df[outcome_label]).dropna().astype(int).unique().tolist())
    return feat_domains, class_domain


def write_arff_with_schema(
    df: pd.DataFrame,
    feature_names: List[str],
    outcome_label: str,
    filename: Path,
    relation_name: str,
    feat_domains: Dict[str, List[int]],
    class_domain: List[int],
    unknown_as_missing: bool = True,
) -> None:
    """
    Write ARFF using a fixed schema (domains), typically inferred from TRAIN.
    If unknown_as_missing=True, any value not in the domain is written as '?'.
    """
    with open(filename, "w") as f:
        f.write(f"@RELATION {relation_name}\n\n")

        for feat in feature_names:
            dom = feat_domains[feat]
            value_str = ",".join(map(str, dom))
            f.write(f"@ATTRIBUTE {feat} {{{value_str}}}\n")

        class_str = ",".join(map(str, class_domain))
        f.write(f"@ATTRIBUTE {outcome_label} {{{class_str}}}\n\n")

        f.write("@DATA\n")
        for _, row in df.iterrows():
            vals_out: List[str] = []
            for feat in feature_names:
                v = int(row[feat])
                if v in feat_domains[feat]:
                    vals_out.append(str(v))
                else:
                    if unknown_as_missing:
                        vals_out.append("?")
                    else:
                        raise ValueError(f"Value {v} for feature '{feat}' not in TRAIN domain {feat_domains[feat]}")

            y = int(row[outcome_label])
            if y in class_domain:
                vals_out.append(str(y))
            else:
                if unknown_as_missing:
                    vals_out.append("?")
                else:
                    raise ValueError(f"Class value {y} not in TRAIN class domain {class_domain}")

            f.write(",".join(vals_out) + "\n")


# Wrong: this peeks at test data to infer schema, which is not allowed.
def convert_to_arff(df: pd.DataFrame, feature_names: List[str], outcome_label: str, filename: Path, relation_name: str) -> None:
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
    config_content = f"""crossover operator 1px
default class major
fitness function mdl
initialization min classifiers 20
initialization max classifiers 20
iterations 2000
mdl initial tl ratio 0.25
mdl iteration 50
mdl weight relax factor 0.90
pop size 2000
prob crossover 0.8
prob individual mutation 0.1
prob one 0.75
selection algorithm tournamentwor
tournament size 5
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
# Rule parsing + application
# Supports:
#   - BioHEL phenotype lines: "0:Att A_0 is 1|...|1"
#   - RPE ruleset lines:      "Att A_0 is 1|...|1"
#   - Default rule:           "Default rule -> 0"
# -----------------------------
@dataclass(frozen=True)
class ParsedRule:
    order: int
    conditions: Dict[str, int]
    prediction: int
    is_default: bool = False


_COND_RE = re.compile(r"Att\s+([A-Za-z0-9_]+)\s+is\s+([-+]?\d+)")
_DEFAULT_RE = re.compile(r"^\s*(?:\d+:)?\s*Default rule\s*->\s*([-+]?\d+)\s*$", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^\s*\d+\s*:\s*")


def parse_ruleset_lines(lines: List[str]) -> Tuple[List[ParsedRule], Optional[ParsedRule]]:
    rules: List[ParsedRule] = []
    default_rule: Optional[ParsedRule] = None

    order = 0
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        mdef = _DEFAULT_RE.match(line)
        if mdef:
            default_rule = ParsedRule(order=10**9, conditions={}, prediction=int(mdef.group(1)), is_default=True)
            continue

        # strip "0:" prefix if present
        line = _PREFIX_RE.sub("", line)

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue

        try:
            pred = int(parts[-1])
        except ValueError:
            continue

        cond_text = "|".join(parts[:-1])
        conds: Dict[str, int] = {}
        for feat, val in _COND_RE.findall(cond_text):
            conds[feat] = int(val)

        rules.append(ParsedRule(order=order, conditions=conds, prediction=pred, is_default=False))
        order += 1

    return rules, default_rule


def apply_ruleset_predict(df: pd.DataFrame, rules: List[ParsedRule], default_rule: Optional[ParsedRule],
                          excluded_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_pred: predicted labels for ALL instances (default fills uncovered)
      covered: boolean array True if covered by ANY non-default rule
    """
    X = df.drop(columns=[c for c in excluded_columns if c in df.columns], errors="ignore")
    n = len(X)
    covered = np.zeros(n, dtype=bool)

    # initialize with default prediction
    if default_rule is None:
        # If absent, choose the most frequent class in the data if Class exists; else 0
        default_pred = int(df["Class"].mode().iloc[0]) if "Class" in df.columns and not df["Class"].mode().empty else 0
    else:
        default_pred = default_rule.prediction

    y_pred = np.full(n, default_pred, dtype=int)

    # apply in order: first match wins (typical ruleset semantics)
    undecided = np.ones(n, dtype=bool)

    for r in sorted(rules, key=lambda rr: rr.order):
        if not r.conditions:
            # condition-less rule matches everything still undecided
            hit = undecided.copy()
        else:
            hit = undecided.copy()
            for feat, val in r.conditions.items():
                if feat not in X.columns:
                    hit &= False
                    break
                hit &= (X[feat].astype(int).values == int(val))

        if hit.any():
            y_pred[hit] = int(r.prediction)
            covered[hit] = True
            undecided[hit] = False

        if not undecided.any():
            break

    return y_pred, covered


def coverage_from_covered(covered: np.ndarray) -> float:
    return float(np.mean(covered)) if covered.size else float("nan")


# -----------------------------
# BioHEL output extraction
# -----------------------------
def extract_phenotype_rules(stdout: str) -> List[str]:
    lines = stdout.splitlines()
    out: List[str] = []
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
            out.append(ln.rstrip())
    return out


def parse_biohel_train_test_accuracy(stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    train_acc = None
    test_acc = None
    runtime = None
    for line in stdout.splitlines():
        if "Train accuracy :" in line:
            m = re.search(r"Train accuracy\s*:\s*([\d.]+)", line)
            if m:
                train_acc = float(m.group(1))
        if "Test accuracy :" in line:
            m = re.search(r"Test accuracy\s*:\s*([\d.]+)", line)
            if m:
                test_acc = float(m.group(1))
        if "Total time:" in line:
            m = re.search(r"Total time:\s*([\d.]+)", line)
            if m:
                runtime = float(m.group(1))
    return train_acc, test_acc, runtime


# -----------------------------
# RPE integration
# -----------------------------
_RULE_ID_PREFIX = re.compile(r"^\s*\d+\s*:\s*")


def write_ruleset_for_rpe(phenotype_lines: List[str], out_path: Path) -> int:
    """
    Convert BioHEL phenotype to RPE ruleset:
      - strip leading "id:" from each rule
      - include Default rule line
    """
    rules_out = []
    default_line = None

    for ln in phenotype_lines:
        s = ln.strip()
        if not s:
            continue
        if "Default rule" in s:
            default_line = _RULE_ID_PREFIX.sub("", s)
            continue
        # normal rule
        rules_out.append(_RULE_ID_PREFIX.sub("", s))

    if default_line is None:
        # BioHEL sometimes prints default outside phenotype; try to locate in original lines
        for ln in phenotype_lines:
            if "Default rule" in ln:
                default_line = _RULE_ID_PREFIX.sub("", ln.strip())
                break

    if default_line is None:
        raise ValueError("Default rule not found in phenotype rules.")

    out_path.write_text("\n".join(rules_out + [default_line]) + "\n")
    return len(rules_out)


def run_postprocess(postprocess_bin: str, conf_path: str, ruleset_path: Path,
                    train_arff: Path, test_arff: Path, cwd: Path) -> str:
    cmd = [postprocess_bin, conf_path, str(ruleset_path), str(train_arff), str(test_arff)]
    # print(f"Running RPE postprocess: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"postprocess failed (rc={proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def extract_rpe_rules_from_stdout(pp_stdout: str) -> List[str]:
    """
    Conservative heuristic: keep lines that look like BioHEL rules + default.
    """
    lines = pp_stdout.splitlines()
    out: List[str] = []
    in_pheno = False
    for ln in lines:
        if ln.startswith("Phenotype:"):
            in_pheno = True
            continue
        if in_pheno:
            if not ln.strip():
                break
            if ln.startswith("Total"):
                break
            out.append(ln.rstrip())
    return out


# -----------------------------
# Main
# -----------------------------
def main(argv):
    parser = argparse.ArgumentParser(description="BioHEL job: one CV fold + one seed (+ optional RPE)")

    parser.add_argument("--d", dest="full_data_path", type=str, required=True,
                        help="Path to training fold: <dataset>_CV_Train_k.txt")
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Output directory for this seed+cv fold")

    parser.add_argument("--biohel", dest="biohel_bin", type=str, default="./biohel")

    parser.add_argument("--enable_rpe", dest="enable_rpe", action="store_true")
    parser.add_argument("--postprocess_bin", dest="postprocess_bin", type=str, default="./postprocess")
    parser.add_argument("--postprocess_conf", dest="postprocess_conf", type=str, default="./postprocess.conf")

    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    parser.add_argument("--rs", dest="random_state", type=int, default=42)
    parser.add_argument("--v", dest="verbose", action="store_true")

    opts = parser.parse_args(argv[1:])

    train_path = opts.full_data_path
    test_path = train_path.replace("Train", "Test")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Expected test fold at: {test_path}")

    outdir = Path(opts.outputPath)
    outdir.mkdir(parents=True, exist_ok=True)

    outcome_label = opts.outcome_label
    instanceID_label = opts.instanceID_label
    excluded_column = opts.excluded_column
    seed = int(opts.random_state)

    # Load raw dataframes (used for deterministic evaluation + coverage)
    train_df = load_df(train_path)
    test_df = load_df(test_path)

    # Build ARFF + config
    feature_names = build_feature_names(train_df, outcome_label, instanceID_label, excluded_column)

    train_arff = outdir / "train.arff"
    test_arff = outdir / "test.arff"
    conf_path = outdir / "config.conf"

    # Infer a single ARFF schema from TRAIN ONLY (consistent header; avoids test peeking)
    feat_domains, class_domain = infer_arff_schema(train_df, feature_names, outcome_label)

    write_arff_with_schema(
        train_df, feature_names, outcome_label,
        filename=train_arff, relation_name="Train",
        feat_domains=feat_domains, class_domain=class_domain,
    )

    write_arff_with_schema(
        test_df, feature_names, outcome_label,
        filename=test_arff, relation_name="Test",
        feat_domains=feat_domains, class_domain=class_domain,
    )

    create_biohel_config(conf_path, seed=seed)

    # Run BioHEL
    cmd = [opts.biohel_bin, str(conf_path), str(train_arff), str(test_arff)]
    t0 = time.time()
    # print(f"Running BioHEL: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_time = time.time() - t0

    (outdir / "biohel_stdout.txt").write_text(proc.stdout)
    (outdir / "biohel_stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(f"BioHEL failed (rc={proc.returncode}). See biohel_stderr.txt")

    # Extract phenotype rules (raw)
    phenotype_lines = extract_phenotype_rules(proc.stdout)
    (outdir / "rules_raw.txt").write_text("\n".join(phenotype_lines) + ("\n" if phenotype_lines else ""))

    # Compute raw evaluation from raw rules (deterministic)
    rules_lines_for_eval = []
    for ln in phenotype_lines:
        s = ln.strip()
        if not s:
            continue
        rules_lines_for_eval.append(s)

    rules_raw, default_raw = parse_ruleset_lines(rules_lines_for_eval)

    excluded_cols = [excluded_column, instanceID_label, outcome_label]

    yhat_train_raw, covered_train_raw = apply_ruleset_predict(train_df, rules_raw, default_raw, excluded_cols)
    yhat_test_raw, covered_test_raw = apply_ruleset_predict(test_df, rules_raw, default_raw, excluded_cols)

    y_train = train_df[outcome_label].astype(int).values
    y_test = test_df[outcome_label].astype(int).values

    train_acc_raw = float(np.mean(yhat_train_raw == y_train))
    test_acc_raw = float(np.mean(yhat_test_raw == y_test))
    train_cov_raw = coverage_from_covered(covered_train_raw)
    test_cov_raw = coverage_from_covered(covered_test_raw)

    num_rules_raw = len(rules_raw)

    # Parse BioHEL-reported runtime (optional; not used for correctness)
    train_acc_reported, test_acc_reported, runtime_reported = parse_biohel_train_test_accuracy(proc.stdout)

    raw_row = {
        "train_file": train_path,
        "test_file": test_path,
        "seed": seed,

        "train_accuracy": train_acc_raw,
        "test_accuracy": test_acc_raw,
        "train_coverage": train_cov_raw,
        "test_coverage": test_cov_raw,
        "train_default_rate": float(1.0 - train_cov_raw),
        "test_default_rate": float(1.0 - test_cov_raw),

        "num_rules": int(num_rules_raw),

        "runtime": runtime_reported,
        "wall_time": float(wall_time),

        # trace: what BioHEL printed (if present)
        "train_accuracy_reported": train_acc_reported,
        "test_accuracy_reported": test_acc_reported,
    }

    pd.DataFrame([raw_row]).to_csv(outdir / "result_row_raw.csv", index=False)

    # Optional: run RPE and use postprocessed rules as "final"
    final_row = dict(raw_row)
    final_row["postprocess_wall_time"] = None
    final_row["postprocess_num_rules"] = None

    if opts.enable_rpe:
        ruleset_path = outdir / "ruleset_for_rpe.txt"
        write_ruleset_for_rpe(phenotype_lines, ruleset_path)

        t1 = time.time()
        pp_stdout = run_postprocess(
            postprocess_bin=opts.postprocess_bin,
            conf_path=opts.postprocess_conf,
            ruleset_path=ruleset_path,
            train_arff=train_arff,
            test_arff=test_arff,
            cwd=outdir,
        )
        pp_wall = time.time() - t1

        (outdir / "postprocess_stdout.txt").write_text(pp_stdout)

        pp_rules_lines = extract_rpe_rules_from_stdout(pp_stdout)
        (outdir / "rules_postprocessed.txt").write_text("\n".join(pp_rules_lines) + ("\n" if pp_rules_lines else ""))

        # Deterministic evaluation from postprocessed rules
        rules_pp, default_pp = parse_ruleset_lines(pp_rules_lines)

        yhat_train_pp, covered_train_pp = apply_ruleset_predict(train_df, rules_pp, default_pp, excluded_cols)
        yhat_test_pp, covered_test_pp = apply_ruleset_predict(test_df, rules_pp, default_pp, excluded_cols)

        train_acc_pp = float(np.mean(yhat_train_pp == y_train))
        test_acc_pp = float(np.mean(yhat_test_pp == y_test))
        train_cov_pp = coverage_from_covered(covered_train_pp)
        test_cov_pp = coverage_from_covered(covered_test_pp)

        final_row.update({
            "train_accuracy": train_acc_pp,
            "test_accuracy": test_acc_pp,
            "train_coverage": train_cov_pp,
            "test_coverage": test_cov_pp,
            "train_default_rate": float(1.0 - train_cov_pp),
            "test_default_rate": float(1.0 - test_cov_pp),
            "num_rules": int(len(rules_pp)),
            "postprocess_wall_time": float(pp_wall),
            "postprocess_num_rules": int(len(rules_pp)),
        })

    # Write final sentinel row used by summary + check/resub logic
    pd.DataFrame([final_row]).to_csv(outdir / "result_row.csv", index=False)

    # Write a metrics.json for convenience
    (outdir / "metrics.json").write_text(json.dumps({"raw": raw_row, "final": final_row}, indent=2) + "\n")

    if opts.verbose:
        print(
            f"Done seed={seed} "
            f"test_acc={final_row['test_accuracy']:.4f} "
            f"test_cov={final_row['test_coverage']:.4f} "
            f"rules={final_row['num_rules']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
