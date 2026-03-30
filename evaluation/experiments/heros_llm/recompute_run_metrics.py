"""Recompute programmatic metrics for an existing run without rerunning the LLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import load_experiment_config
from .data_models import (
    ActiveRule,
    FeatureGlossaryEntry,
    InstanceExplanationPacket,
    ModelContextSummary,
    RuleCondition,
    RuleMetadata,
)
from .metrics_programmatic import compute_programmatic_metrics
from .results import (
    ResultsWriter,
    aggregate_serialized_records,
    flatten_dict,
    write_summary_tables,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _restore_rule_condition(payload: Dict[str, Any]) -> RuleCondition:
    return RuleCondition(**payload)


def _restore_rule_metadata(payload: Dict[str, Any]) -> RuleMetadata:
    return RuleMetadata(**payload)


def _restore_active_rule(payload: Dict[str, Any]) -> ActiveRule:
    return ActiveRule(
        rule_id=payload["rule_id"],
        action=payload["action"],
        supports_prediction=payload["supports_prediction"],
        contradicts_prediction=payload["contradicts_prediction"],
        conditions=[_restore_rule_condition(item) for item in payload["conditions"]],
        if_then_text=payload["if_then_text"],
        metadata=_restore_rule_metadata(payload.get("metadata", {})),
    )


def _restore_model_context(payload: Dict[str, Any]) -> ModelContextSummary:
    return ModelContextSummary(**payload)


def _restore_glossary_entry(payload: Dict[str, Any]) -> FeatureGlossaryEntry:
    return FeatureGlossaryEntry(**payload)


def _restore_packet(payload: Dict[str, Any]) -> InstanceExplanationPacket:
    return InstanceExplanationPacket(
        dataset_name=payload["dataset_name"],
        split=payload["split"],
        instance_id=payload["instance_id"],
        feature_values=payload["feature_values"],
        active_rules=[_restore_active_rule(item) for item in payload["active_rules"]],
        model_context=_restore_model_context(payload["model_context"]),
        heros_description=payload["heros_description"],
        glossary=[_restore_glossary_entry(item) for item in payload.get("glossary", [])],
        condition=payload.get("condition"),
        audience=payload.get("audience"),
        prompt_flags=payload.get("prompt_flags", {}),
    )


def recompute_run_metrics(run_dir: str) -> Path:
    target_dir = Path(run_dir).expanduser().resolve()
    records_path = target_dir / "records.jsonl"
    config_snapshot_path = target_dir / "config_snapshot.json"
    records = _load_jsonl(records_path)
    if not records:
        raise RuntimeError("No records found at {0}".format(records_path))

    config = load_experiment_config(str(config_snapshot_path))
    top_k_rules = config.prompt.key_rules_top_k

    for record in records:
        packet = _restore_packet(record["packet"])
        explanation_text = record["generation"]["raw_text"]
        metrics = compute_programmatic_metrics(
            packet=packet,
            explanation_text=explanation_text,
            top_k_rules=top_k_rules,
        )
        record["programmatic_metrics"] = metrics.to_dict()

    _write_jsonl(records_path, records)
    ResultsWriter.write_csv(target_dir / "records.csv", [flatten_dict(record) for record in records])
    with (target_dir / "aggregate_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_serialized_records(records), handle, indent=2, sort_keys=True)
    write_summary_tables(target_dir, records)
    return target_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute programmatic metrics for an existing run.")
    parser.add_argument("--run-dir", required=True, help="Path to an existing run directory.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    updated_dir = recompute_run_metrics(args.run_dir)
    print("Recomputed programmatic metrics for {0}".format(updated_dir))


if __name__ == "__main__":
    main()
