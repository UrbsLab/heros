"""Artifact writing utilities for experiment runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

from .config import OutputConfig, save_config_snapshot
from .data_models import ExplanationRecord, SerializableDataclass, to_serializable


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
        record_list = list(records)
        if not record_list:
            return {
                "record_count": 0,
                "hallucination_rate": None,
                "mean_feature_grounding_score": None,
                "mean_key_feature_coverage": None,
                "mean_prediction_consistency": None,
                "mean_word_count": None,
                "uncertainty_ack_rate_on_required_cases": None,
                "mean_rule_mention_coverage": None,
                "mean_conflict_awareness_score": None,
                "confidence_calibration_pass_rate": None,
                "causal_overclaim_rate": None,
                "mean_clarity_score": None,
                "mean_technical_appropriateness_score": None,
            }

        def _mean(values: List[float]) -> Any:
            return mean(values) if values else None

        feature_grounding = [
            value
            for value in (
                record.programmatic_metrics.feature_grounding_score for record in record_list
            )
            if value is not None
        ]
        key_feature_coverage = [
            value
            for value in (record.programmatic_metrics.key_feature_coverage for record in record_list)
            if value is not None
        ]
        prediction_consistency = [
            value
            for value in (record.programmatic_metrics.prediction_consistency for record in record_list)
            if value is not None
        ]
        rule_mention_coverage = [
            value
            for value in (record.programmatic_metrics.rule_mention_coverage for record in record_list)
            if value is not None
        ]
        conflict_awareness = [
            value
            for value in (record.programmatic_metrics.conflict_awareness_score for record in record_list)
            if value is not None
        ]
        uncertainty_required = [
            record.programmatic_metrics for record in record_list if record.programmatic_metrics.uncertainty_ack_required
        ]
        clarity_scores = [
            value for value in (record.judge_metrics.clarity_score for record in record_list) if value is not None
        ]
        technical_scores = [
            value
            for value in (
                record.judge_metrics.technical_appropriateness_score for record in record_list
            )
            if value is not None
        ]

        return {
            "record_count": len(record_list),
            "hallucination_rate": _mean(
                [1.0 if record.programmatic_metrics.hallucination_present else 0.0 for record in record_list]
            ),
            "mean_feature_grounding_score": _mean(feature_grounding),
            "mean_key_feature_coverage": _mean(key_feature_coverage),
            "mean_prediction_consistency": _mean([float(value) for value in prediction_consistency]),
            "mean_word_count": _mean(
                [float(record.programmatic_metrics.word_count) for record in record_list]
            ),
            "uncertainty_ack_rate_on_required_cases": _mean(
                [1.0 if metrics.uncertainty_ack_present else 0.0 for metrics in uncertainty_required]
            ),
            "mean_rule_mention_coverage": _mean(rule_mention_coverage),
            "mean_conflict_awareness_score": _mean(conflict_awareness),
            "confidence_calibration_pass_rate": _mean(
                [
                    1.0 if record.programmatic_metrics.confidence_calibration_pass else 0.0
                    for record in record_list
                ]
            ),
            "causal_overclaim_rate": _mean(
                [
                    1.0 if record.programmatic_metrics.causal_overclaim_present else 0.0
                    for record in record_list
                ]
            ),
            "mean_clarity_score": _mean(clarity_scores),
            "mean_technical_appropriateness_score": _mean(technical_scores),
        }

    def write_aggregate_metrics(self, records: Iterable[ExplanationRecord]) -> None:
        aggregate = self.aggregate_records(records)
        with self.run_paths.aggregate_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, sort_keys=True)

    def serialize_rows(self, rows: Iterable[Any]) -> List[Dict[str, Any]]:
        return [to_serializable(row) for row in rows]
