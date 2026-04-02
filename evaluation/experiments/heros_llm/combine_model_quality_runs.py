"""Combine baseline and properly trained HEROS+LLM runs into comparison tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


PROFILE_LABELS = {
    "corrected_baseline": "Baseline",
    "properly_trained": "Properly Trained",
}

CONDITION_LABELS = {
    "condition_b": "Experiment 1",
    "condition_c": "Experiment 2",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _records_to_rows(records: Iterable[Dict[str, Any]], run_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        metadata = record["experiment_metadata"]
        packet = record["packet"]
        prompt = record["prompt"]
        pm = record["programmatic_metrics"]
        jm = record.get("judge_metrics", {})
        rows.append(
            {
                "run_name": run_name,
                "dataset": metadata["dataset_name"],
                "heros_profile": PROFILE_LABELS.get(
                    metadata.get("heros_training_profile", "baseline"),
                    metadata.get("heros_training_profile", "baseline"),
                ),
                "condition": CONDITION_LABELS.get(prompt["condition"], prompt["condition"]),
                "audience": prompt["audience"].title(),
                "instance_id": packet["instance_id"],
                "mean_matching_rules": packet["model_context"]["num_matching_rules"],
                "heros_train_accuracy": metadata.get("heros_train_accuracy"),
                "heros_test_accuracy": metadata.get("heros_test_accuracy"),
                "heros_test_balanced_accuracy": metadata.get("heros_test_balanced_accuracy"),
                "heros_test_coverage": metadata.get("heros_test_coverage"),
                "heros_top_model_rule_count": metadata.get("heros_top_model_rule_count"),
                "heros_rule_population_size": metadata.get("heros_rule_population_size"),
                "heros_model_population_size": metadata.get("heros_model_population_size"),
                "ideal_rules_in_top_model": metadata.get("ideal_rules_in_top_model"),
                "ideal_rule_fraction_in_top_model": metadata.get("ideal_rule_fraction_in_top_model"),
                "evidence_precision": pm.get("evidence_precision"),
                "evidence_recall": pm.get("evidence_recall"),
                "evidence_f1": pm.get("evidence_f1"),
                "hallucination_rate": 1.0 if pm.get("hallucination_present") else 0.0,
                "prediction_explanation_agreement": pm.get("prediction_explanation_agreement"),
                "rule_coverage": pm.get("rule_coverage"),
                "conflict_acknowledgment_score": pm.get("conflict_acknowledgment_score"),
                "uncertainty_ack_rate": 1.0 if pm.get("uncertainty_ack_present") else 0.0,
                "causal_overclaim_rate": 1.0 if pm.get("causal_overclaim_present") else 0.0,
                "audience_understandability": jm.get("audience_understandability_score"),
                "audience_technical_fit": jm.get("audience_technical_fit_score"),
                "flesch_reading_ease": pm.get("flesch_reading_ease"),
                "flesch_kincaid_grade_level": pm.get("flesch_kincaid_grade_level"),
                "word_count": pm.get("word_count"),
            }
        )
    return rows


def _aggregate(df: pd.DataFrame, group_columns: List[str]) -> pd.DataFrame:
    return (
        df.groupby(group_columns, dropna=False)
        .agg(
            n=("instance_id", "size"),
            unique_instances=("instance_id", "nunique"),
            heros_train_accuracy=("heros_train_accuracy", "mean"),
            heros_test_accuracy=("heros_test_accuracy", "mean"),
            heros_test_balanced_accuracy=("heros_test_balanced_accuracy", "mean"),
            heros_test_coverage=("heros_test_coverage", "mean"),
            heros_top_model_rule_count=("heros_top_model_rule_count", "mean"),
            heros_rule_population_size=("heros_rule_population_size", "mean"),
            heros_model_population_size=("heros_model_population_size", "mean"),
            ideal_rules_in_top_model=("ideal_rules_in_top_model", "mean"),
            ideal_rule_fraction_in_top_model=("ideal_rule_fraction_in_top_model", "mean"),
            mean_matching_rules=("mean_matching_rules", "mean"),
            evidence_precision=("evidence_precision", "mean"),
            evidence_recall=("evidence_recall", "mean"),
            evidence_f1=("evidence_f1", "mean"),
            hallucination_rate=("hallucination_rate", "mean"),
            prediction_explanation_agreement=("prediction_explanation_agreement", "mean"),
            rule_coverage=("rule_coverage", "mean"),
            conflict_acknowledgment_score=("conflict_acknowledgment_score", "mean"),
            uncertainty_ack_rate=("uncertainty_ack_rate", "mean"),
            causal_overclaim_rate=("causal_overclaim_rate", "mean"),
            audience_understandability=("audience_understandability", "mean"),
            audience_technical_fit=("audience_technical_fit", "mean"),
            flesch_reading_ease=("flesch_reading_ease", "mean"),
            flesch_kincaid_grade_level=("flesch_kincaid_grade_level", "mean"),
            word_count=("word_count", "mean"),
        )
        .reset_index()
    )


def _to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator] + body)


def _format_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    percentage_columns = [
        "heros_train_accuracy",
        "heros_test_accuracy",
        "heros_test_balanced_accuracy",
        "heros_test_coverage",
        "ideal_rule_fraction_in_top_model",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "hallucination_rate",
        "prediction_explanation_agreement",
        "rule_coverage",
        "conflict_acknowledgment_score",
        "uncertainty_ack_rate",
        "causal_overclaim_rate",
    ]
    for column in percentage_columns:
        if column in out:
            out[column] = out[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value) * 100.0:.1f}"
            )
    score_columns = [
        "audience_understandability",
        "audience_technical_fit",
        "flesch_reading_ease",
        "flesch_kincaid_grade_level",
        "mean_matching_rules",
        "heros_top_model_rule_count",
        "heros_rule_population_size",
        "heros_model_population_size",
        "ideal_rules_in_top_model",
        "word_count",
    ]
    for column in score_columns:
        if column in out:
            out[column] = out[column].map(
                lambda value: "NA" if pd.isna(value) else f"{float(value):.2f}"
            )
    for column in ["n", "unique_instances"]:
        if column in out:
            out[column] = out[column].map(lambda value: f"{int(value)}")
    return out


def combine_runs(run_dirs: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        records_path = run_dir / "records.jsonl"
        if not records_path.exists():
            raise FileNotFoundError(f"Missing records.jsonl in {run_dir}")
        all_rows.extend(_records_to_rows(_load_jsonl(records_path), run_dir.name))

    df = pd.DataFrame(all_rows)
    profile_overall = _aggregate(df, ["heros_profile"])
    dataset_profile = _aggregate(df, ["dataset", "heros_profile"])
    dataset_profile_condition = _aggregate(df, ["dataset", "heros_profile", "condition"])
    profile_audience = _aggregate(df, ["heros_profile", "audience"])

    profile_overall.to_csv(output_dir / "profile_overall_summary.csv", index=False)
    dataset_profile.to_csv(output_dir / "dataset_profile_summary.csv", index=False)
    dataset_profile_condition.to_csv(output_dir / "dataset_profile_condition_summary.csv", index=False)
    profile_audience.to_csv(output_dir / "profile_audience_summary.csv", index=False)

    markdown = "\n".join(
        [
            "# HEROS Model-Quality Comparison",
            "",
            "## Table 1. Overall Comparison by HEROS Training Profile",
            _to_markdown(_format_summary(profile_overall)),
            "",
            "## Table 2. Dataset x HEROS Training Profile",
            _to_markdown(_format_summary(dataset_profile)),
            "",
            "## Table 3. Dataset x HEROS Training Profile x Experiment",
            _to_markdown(_format_summary(dataset_profile_condition)),
            "",
            "## Table 4. HEROS Training Profile x Audience",
            _to_markdown(_format_summary(profile_audience)),
        ]
    )
    (output_dir / "combined_model_quality_tables.md").write_text(markdown, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Run directory containing records.jsonl. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write combined summaries into.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    combine_runs([Path(path) for path in args.run_dirs], Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
