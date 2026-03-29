"""Artifact writing utilities for experiment runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from .config import OutputConfig, save_config_snapshot
from .data_models import ExplanationRecord, SerializableDataclass, to_serializable


PROGRAMMATIC_METRIC_ALIASES = {
    "evidence_grounding_precision": ["evidence_grounding_precision", "feature_grounding_score"],
    "hallucination_present": ["hallucination_present"],
    "key_evidence_coverage": ["key_evidence_coverage", "key_feature_coverage"],
    "prediction_explanation_agreement": ["prediction_explanation_agreement", "prediction_consistency"],
    "word_count": ["word_count"],
    "uncertainty_ack_required": ["uncertainty_ack_required"],
    "uncertainty_ack_present": ["uncertainty_ack_present"],
    "causal_overclaim_present": ["causal_overclaim_present"],
    "rule_coverage": ["rule_coverage", "rule_mention_coverage"],
    "conflict_acknowledgment_score": ["conflict_acknowledgment_score", "conflict_awareness_score"],
    "confidence_wording_calibration": ["confidence_wording_calibration", "confidence_calibration_pass"],
    "flesch_reading_ease": ["flesch_reading_ease"],
    "flesch_kincaid_grade_level": ["flesch_kincaid_grade_level"],
}

JUDGE_METRIC_ALIASES = {
    "audience_understandability_score": [
        "audience_understandability_score",
        "clarity_score",
    ],
    "audience_technical_fit_score": [
        "audience_technical_fit_score",
        "technical_appropriateness_score",
    ],
}

SUMMARY_COLUMN_ORDER = [
    "Condition",
    "Audience",
    "n",
    "Evidence Grounding Precision",
    "Hallucination Rate",
    "Key Evidence Coverage",
    "Prediction-Explanation Agreement",
    "Uncertainty Acknowledgment Rate",
    "Causal Overclaim Rate",
    "Rule Coverage",
    "Conflict Acknowledgment Score",
    "Confidence Wording Calibration",
    "Audience Understandability",
    "Audience Technical Fit",
    "WordCount",
]


def flatten_dict(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts using dot-separated keys."""
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        compound_key = "{0}.{1}".format(prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, compound_key))
        elif isinstance(value, list):
            flattened[compound_key] = json.dumps(value, sort_keys=True)
        else:
            flattened[compound_key] = value
    return flattened


def _metric_value(payload: Dict[str, Any], aliases: Dict[str, List[str]], key: str) -> Any:
    for candidate in aliases[key]:
        if candidate in payload:
            return payload[candidate]
    return None


def _mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def aggregate_serialized_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from serialized records with legacy key support."""
    if not records:
        return {
            "record_count": 0,
            "hallucination_rate": None,
            "mean_evidence_grounding_precision": None,
            "mean_key_evidence_coverage": None,
            "mean_prediction_explanation_agreement": None,
            "mean_word_count": None,
            "uncertainty_ack_rate_on_required_cases": None,
            "mean_rule_coverage": None,
            "mean_conflict_acknowledgment_score": None,
            "confidence_wording_calibration_rate": None,
            "causal_overclaim_rate": None,
            "mean_audience_understandability_score": None,
            "mean_audience_technical_fit_score": None,
            "mean_flesch_reading_ease": None,
            "mean_flesch_kincaid_grade_level": None,
        }

    programmatic_payloads = [record["programmatic_metrics"] for record in records]
    judge_payloads = [record.get("judge_metrics", {}) for record in records]

    uncertainty_required = [
        metrics
        for metrics in programmatic_payloads
        if _metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "uncertainty_ack_required")
    ]

    return {
        "record_count": len(records),
        "hallucination_rate": _mean(
            [
                1.0 if _metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "hallucination_present") else 0.0
                for metrics in programmatic_payloads
            ]
        ),
        "mean_evidence_grounding_precision": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "evidence_grounding_precision")]
                if value is not None
            ]
        ),
        "mean_key_evidence_coverage": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "key_evidence_coverage")]
                if value is not None
            ]
        ),
        "mean_prediction_explanation_agreement": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "prediction_explanation_agreement")]
                if value is not None
            ]
        ),
        "mean_word_count": _mean(
            [
                float(_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "word_count") or 0.0)
                for metrics in programmatic_payloads
            ]
        ),
        "uncertainty_ack_rate_on_required_cases": _mean(
            [
                1.0 if _metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "uncertainty_ack_present") else 0.0
                for metrics in uncertainty_required
            ]
        ),
        "mean_rule_coverage": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "rule_coverage")]
                if value is not None
            ]
        ),
        "mean_conflict_acknowledgment_score": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "conflict_acknowledgment_score")]
                if value is not None
            ]
        ),
        "confidence_wording_calibration_rate": _mean(
            [
                1.0 if _metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "confidence_wording_calibration") else 0.0
                for metrics in programmatic_payloads
            ]
        ),
        "causal_overclaim_rate": _mean(
            [
                1.0 if _metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "causal_overclaim_present") else 0.0
                for metrics in programmatic_payloads
            ]
        ),
        "mean_audience_understandability_score": _mean(
            [
                float(value)
                for metrics in judge_payloads
                for value in [_metric_value(metrics, JUDGE_METRIC_ALIASES, "audience_understandability_score")]
                if value is not None
            ]
        ),
        "mean_audience_technical_fit_score": _mean(
            [
                float(value)
                for metrics in judge_payloads
                for value in [_metric_value(metrics, JUDGE_METRIC_ALIASES, "audience_technical_fit_score")]
                if value is not None
            ]
        ),
        "mean_flesch_reading_ease": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "flesch_reading_ease")]
                if value is not None
            ]
        ),
        "mean_flesch_kincaid_grade_level": _mean(
            [
                float(value)
                for metrics in programmatic_payloads
                for value in [_metric_value(metrics, PROGRAMMATIC_METRIC_ALIASES, "flesch_kincaid_grade_level")]
                if value is not None
            ]
        ),
    }


def _summary_rows_from_serialized(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        prompt = record["prompt"]
        pm = record["programmatic_metrics"]
        jm = record.get("judge_metrics", {})
        rows.append(
            {
                "Condition": "B" if prompt["condition"] == "condition_b" else "C",
                "Audience": prompt["audience"].title(),
                "Evidence Grounding Precision": _metric_value(
                    pm, PROGRAMMATIC_METRIC_ALIASES, "evidence_grounding_precision"
                ),
                "Hallucination Rate": 1.0
                if _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "hallucination_present")
                else 0.0,
                "Key Evidence Coverage": _metric_value(
                    pm, PROGRAMMATIC_METRIC_ALIASES, "key_evidence_coverage"
                ),
                "Prediction-Explanation Agreement": _metric_value(
                    pm, PROGRAMMATIC_METRIC_ALIASES, "prediction_explanation_agreement"
                ),
                "Uncertainty Acknowledgment Rate": 1.0
                if _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "uncertainty_ack_present")
                else 0.0,
                "Causal Overclaim Rate": 1.0
                if _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "causal_overclaim_present")
                else 0.0,
                "Rule Coverage": _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "rule_coverage"),
                "Conflict Acknowledgment Score": _metric_value(
                    pm, PROGRAMMATIC_METRIC_ALIASES, "conflict_acknowledgment_score"
                ),
                "Confidence Wording Calibration": 1.0
                if _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "confidence_wording_calibration")
                else 0.0,
                "Audience Understandability": _metric_value(
                    jm, JUDGE_METRIC_ALIASES, "audience_understandability_score"
                ),
                "Audience Technical Fit": _metric_value(
                    jm, JUDGE_METRIC_ALIASES, "audience_technical_fit_score"
                ),
                "WordCount": _metric_value(pm, PROGRAMMATIC_METRIC_ALIASES, "word_count"),
            }
        )
    return rows


def write_summary_tables(run_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Write grouped CSV and Markdown summaries using renamed metrics."""
    import pandas as pd

    rows = _summary_rows_from_serialized(records)
    if not rows:
        for filename in [
            "summary_by_condition.csv",
            "summary_by_audience.csv",
            "summary_condition_by_audience.csv",
            "summary_tables.md",
        ]:
            (run_dir / filename).write_text("", encoding="utf-8")
        return
    df = pd.DataFrame(rows)
    agg_map = {
        "n": ("Condition", "size"),
        "Evidence Grounding Precision": ("Evidence Grounding Precision", "mean"),
        "Hallucination Rate": ("Hallucination Rate", "mean"),
        "Key Evidence Coverage": ("Key Evidence Coverage", "mean"),
        "Prediction-Explanation Agreement": ("Prediction-Explanation Agreement", "mean"),
        "Uncertainty Acknowledgment Rate": ("Uncertainty Acknowledgment Rate", "mean"),
        "Causal Overclaim Rate": ("Causal Overclaim Rate", "mean"),
        "Rule Coverage": ("Rule Coverage", "mean"),
        "Conflict Acknowledgment Score": ("Conflict Acknowledgment Score", "mean"),
        "Confidence Wording Calibration": ("Confidence Wording Calibration", "mean"),
        "Audience Understandability": ("Audience Understandability", "mean"),
        "Audience Technical Fit": ("Audience Technical Fit", "mean"),
        "WordCount": ("WordCount", "mean"),
    }

    condition = df.groupby(["Condition"], dropna=False).agg(**agg_map).reset_index()
    audience = df.groupby(["Audience"], dropna=False).agg(**agg_map).reset_index()
    condition_audience = df.groupby(["Condition", "Audience"], dropna=False).agg(**agg_map).reset_index()

    condition = condition[["Condition"] + SUMMARY_COLUMN_ORDER[2:]]
    audience = audience[["Audience"] + SUMMARY_COLUMN_ORDER[2:]]
    condition_audience = condition_audience[["Condition", "Audience"] + SUMMARY_COLUMN_ORDER[2:]]

    condition.to_csv(run_dir / "summary_by_condition.csv", index=False)
    audience.to_csv(run_dir / "summary_by_audience.csv", index=False)
    condition_audience.to_csv(run_dir / "summary_condition_by_audience.csv", index=False)

    def pct(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return "{0:.1f}".format(float(value) * 100.0)

    def score(value: Any) -> str:
        if value is None or pd.isna(value):
            return "NA"
        return "{0:.3f}".format(float(value))

    def count(value: Any) -> str:
        return str(int(value))

    def wc(value: Any) -> str:
        return "{0:.1f}".format(float(value))

    def display_df(dataframe: "pd.DataFrame") -> "pd.DataFrame":
        output = dataframe.copy()
        percentage_columns = [
            "Evidence Grounding Precision",
            "Hallucination Rate",
            "Key Evidence Coverage",
            "Prediction-Explanation Agreement",
            "Uncertainty Acknowledgment Rate",
            "Causal Overclaim Rate",
            "Rule Coverage",
            "Conflict Acknowledgment Score",
            "Confidence Wording Calibration",
        ]
        for column in percentage_columns:
            output[column] = output[column].map(pct)
        for column in ["Audience Understandability", "Audience Technical Fit"]:
            output[column] = output[column].map(score)
        output["n"] = output["n"].map(count)
        output["WordCount"] = output["WordCount"].map(wc)
        return output

    def to_markdown_table(dataframe: "pd.DataFrame") -> str:
        columns = list(dataframe.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        body = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in dataframe.iterrows()
        ]
        return "\n".join([header, separator] + body)

    markdown = "\n".join(
        [
            "# MUX6 HEROS-LLM Summary Tables",
            "",
            "Source run: `{0}`".format(run_dir.name),
            "",
            "Metrics reported as means. Percentage-style metrics are shown on a 0-100 scale.",
            "",
            "## Table 1. Condition Comparison",
            to_markdown_table(display_df(condition)),
            "",
            "## Table 2. Audience Comparison",
            to_markdown_table(display_df(audience)),
            "",
            "## Table 3. Condition x Audience",
            to_markdown_table(display_df(condition_audience)),
        ]
    )
    (run_dir / "summary_tables.md").write_text(markdown, encoding="utf-8")


@dataclass
class RunPaths:
    run_dir: Path
    cache_dir: Path
    config_snapshot_path: Path
    sample_manifest_path: Path
    packets_path: Path
    prompts_path: Path
    generations_path: Path
    judge_requests_path: Path
    judge_results_path: Path
    records_path: Path
    records_csv_path: Path
    aggregate_metrics_path: Path


class ResultsWriter:
    """Manage run directories and artifact files."""

    def __init__(self, output_config: OutputConfig, run_id: str):
        self.run_paths = self._build_run_paths(output_config, run_id)

    @staticmethod
    def _build_run_paths(output_config: OutputConfig, run_id: str) -> RunPaths:
        run_dir = Path(output_config.base_dir) / run_id
        cache_dir = run_dir / output_config.cache_dir_name
        run_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return RunPaths(
            run_dir=run_dir,
            cache_dir=cache_dir,
            config_snapshot_path=run_dir / "config_snapshot.json",
            sample_manifest_path=run_dir / "sample_manifest.csv",
            packets_path=run_dir / "packets.jsonl",
            prompts_path=run_dir / "prompts.jsonl",
            generations_path=run_dir / "generations.jsonl",
            judge_requests_path=run_dir / "judge_requests.jsonl",
            judge_results_path=run_dir / "judge_results.jsonl",
            records_path=run_dir / "records.jsonl",
            records_csv_path=run_dir / "records.csv",
            aggregate_metrics_path=run_dir / "aggregate_metrics.json",
        )

    def write_config_snapshot(self, config: SerializableDataclass) -> None:
        save_config_snapshot(config, str(self.run_paths.config_snapshot_path))

    @staticmethod
    def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        flat_rows = [flatten_dict(row) for row in rows]
        if not flat_rows:
            with path.open("w", encoding="utf-8") as handle:
                handle.write("")
            return
        fieldnames: List[str] = sorted({key for row in flat_rows for key in row.keys()})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in flat_rows:
                writer.writerow(row)

    @staticmethod
    def aggregate_records(records: Iterable[ExplanationRecord]) -> Dict[str, Any]:
        serialized = [to_serializable(record) for record in records]
        return aggregate_serialized_records(serialized)

    def write_aggregate_metrics(self, records: Iterable[ExplanationRecord]) -> None:
        aggregate = self.aggregate_records(records)
        with self.run_paths.aggregate_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)

    def serialize_rows(self, rows: Iterable[Any]) -> List[Dict[str, Any]]:
        return [to_serializable(row) for row in rows]
