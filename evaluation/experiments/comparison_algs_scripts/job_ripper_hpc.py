#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# wittgenstein
from wittgenstein import RIPPER


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def preprocess_df(df: pd.DataFrame, outcome_label: str, instanceID_label: str, excluded_column: str) -> pd.DataFrame:
    # Keep all feature cols + outcome_label; drop bookkeeping cols if present
    drop_cols = [c for c in [instanceID_label, excluded_column] if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def compute_rule_coverage(model: RIPPER, X: pd.DataFrame) -> float:
    """
    Coverage defined as fraction of rows covered by at least one learned rule (excluding default).
    Uses wittgenstein internals if available; otherwise falls back to conservative parsing.
    """
    n = len(X)
    if n == 0:
        return float("nan")

    # Preferred: model.ruleset_.covers(...) exists for many versions
    rs = getattr(model, "ruleset_", None)
    if rs is not None:
        # Try vectorized covers
        covers_fn = getattr(rs, "covers", None)
        if callable(covers_fn):
            try:
                covered = covers_fn(X)
                # Some versions return list[bool], np.array, or pandas Series
                covered = np.asarray(covered, dtype=bool)
                return float(np.mean(covered))
            except Exception:
                pass

        # Try iterating rules
        rules = getattr(rs, "rules", None)
        if rules is None:
            rules = getattr(rs, "rule_list", None)

        if rules is not None:
            covered_any = np.zeros(n, dtype=bool)
            for r in rules:
                # rule.covers(X) is common
                rc = getattr(r, "covers", None)
                if callable(rc):
                    try:
                        c = np.asarray(rc(X), dtype=bool)
                        covered_any |= c
                    except Exception:
                        continue
            return float(np.mean(covered_any))

    # Fallback: if we cannot evaluate coverage reliably, return NaN
    return float("nan")


def maybe_cap_rules(model: RIPPER, max_rules: Optional[int]) -> int:
    rs = getattr(model, "ruleset_", None)
    if rs is None or max_rules is None:
        return len(getattr(rs, "rules", [])) if rs is not None else 0
    try:
        rules = list(getattr(rs, "rules", []))
        if len(rules) <= max_rules:
            return len(rules)
        # truncate ruleset (best-effort)
        rs.rules = rules[:max_rules]
        return len(rs.rules)
    except Exception:
        return len(getattr(rs, "rules", []))


def write_rules(model: RIPPER, out_path: Path) -> int:
    rs = getattr(model, "ruleset_", None)
    if rs is None:
        out_path.write_text("")
        return 0
    # wittgenstein has readable string for ruleset
    s = str(rs)
    out_path.write_text(s + ("\n" if not s.endswith("\n") else ""))
    # Approximate rule count
    rules = getattr(rs, "rules", None)
    if rules is not None:
        return len(rules)
    return s.count("IF ") if "IF " in s else 0


def main(argv):
    parser = argparse.ArgumentParser(description="RIPPER job: one CV fold + one seed")

    parser.add_argument("--d", dest="full_data_path", type=str, required=True,
                        help="Path to training fold: <dataset>_CV_Train_k.txt")
    parser.add_argument("--o", dest="outputPath", type=str, required=True,
                        help="Output directory for this seed+cv fold")

    parser.add_argument("--ol", dest="outcome_label", type=str, default="Class")
    parser.add_argument("--il", dest="instanceID_label", type=str, default="InstanceID")
    parser.add_argument("--el", dest="excluded_column", type=str, default="Group")

    parser.add_argument("--rs", dest="random_state", type=int, default=42)
    parser.add_argument("--max_rules", dest="max_rules", type=int, default=None)
    parser.add_argument("--verbosity", dest="verbosity", type=int, default=0)
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

    # Load
    train_df_raw = load_df(train_path)
    test_df_raw = load_df(test_path)

    train_df = preprocess_df(train_df_raw, outcome_label, instanceID_label, excluded_column)
    test_df = preprocess_df(test_df_raw, outcome_label, instanceID_label, excluded_column)

    if outcome_label not in train_df.columns or outcome_label not in test_df.columns:
        raise ValueError(f"Outcome label '{outcome_label}' must exist in train and test files.")

    X_train = train_df.drop(columns=[outcome_label])
    y_train = train_df[outcome_label].astype(int).values

    X_test = test_df.drop(columns=[outcome_label])
    y_test = test_df[outcome_label].astype(int).values

    # Train RIPPER
    model = RIPPER(k=20, random_state=seed, verbosity=opts.verbosity)

    t0 = time.time()
    model.fit(X_train, y_train)
    wall_time = time.time() - t0

    # Optionally cap rules (best-effort)
    rule_count = maybe_cap_rules(model, opts.max_rules)

    # Predict
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    train_acc = float(accuracy_score(y_train, yhat_train))
    test_acc = float(accuracy_score(y_test, yhat_test))

    train_cov = compute_rule_coverage(model, X_train)
    test_cov = compute_rule_coverage(model, X_test)

    # Write rules + outputs
    rules_path = outdir / "rules.txt"
    rule_count_written = write_rules(model, rules_path)
    if rule_count_written > 0:
        rule_count = rule_count_written

    row = {
        "train_file": train_path,
        "test_file": test_path,
        "seed": seed,

        "train_accuracy": train_acc,
        "test_accuracy": test_acc,

        # Coverage = fraction covered by any learned rule; if not available => NaN
        "train_coverage": train_cov,
        "test_coverage": test_cov,
        "train_default_rate": (float(1.0 - train_cov) if np.isfinite(train_cov) else np.nan),
        "test_default_rate": (float(1.0 - test_cov) if np.isfinite(test_cov) else np.nan),

        "num_rules": int(rule_count),

        "runtime": float(wall_time),               # RIPPER does not report internal runtime; keep for consistency
        "wall_time": float(wall_time),
    }

    pd.DataFrame([row]).to_csv(outdir / "result_row.csv", index=False)

    # HEROS-style fold eval interface
    eval_df = pd.DataFrame([{
        "Row Indexes": "final",
        **{k: row.get(k) for k in [
            "train_accuracy", "test_accuracy",
            "train_coverage", "test_coverage",
            "train_default_rate", "test_default_rate",
            "num_rules", "runtime", "wall_time"
        ]}
    }])
    eval_df.to_csv(outdir / "evaluation_summary.csv", index=False)

    (outdir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")

    if opts.verbose:
        print(f"Done seed={seed} test_acc={test_acc:.4f} test_cov={test_cov} rules={rule_count}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
