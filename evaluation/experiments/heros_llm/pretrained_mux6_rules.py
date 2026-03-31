"""Utilities for exporting and reusing a trained MUX6 HEROS rule set."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report

from .config import load_experiment_config
from .dataset_registry import get_dataset_definition
from .heros_adapter import train_heros_model

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "evaluation/experiments/heros_llm/configs/mux6_50_test.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/heros_llm/mux6_pretrained_rules"


@dataclass(frozen=True)
class Mux6RuleArtifacts:
    """Paths for exported MUX6 HEROS rule-inspection artifacts."""

    output_dir: Path
    summary_path: Path
    top_model_rules_path: Path
    model_population_path: Path
    rule_population_path: Path


def _artifact_paths(output_dir: Path) -> Mux6RuleArtifacts:
    return Mux6RuleArtifacts(
        output_dir=output_dir,
        summary_path=output_dir / "summary.json",
        top_model_rules_path=output_dir / "top_model_rules.csv",
        model_population_path=output_dir / "model_population.csv",
        rule_population_path=output_dir / "rule_population.csv",
    )


def _artifacts_exist(artifacts: Mux6RuleArtifacts) -> bool:
    return all(
        path.exists()
        for path in (
            artifacts.summary_path,
            artifacts.top_model_rules_path,
            artifacts.model_population_path,
            artifacts.rule_population_path,
        )
    )


def _writable_matplotlib_cache() -> Path:
    cache_dir = REPO_ROOT / "output" / "matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _train_and_export(artifacts: Mux6RuleArtifacts) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(_writable_matplotlib_cache()))

    config = load_experiment_config(str(DEFAULT_CONFIG_PATH))
    config.llm.enabled = False
    config.judge.enabled = False
    config.heros.verbose = False

    context = train_heros_model(config, get_dataset_definition("MUX6"))
    model = context.model
    best_model_index = context.target_model_index

    train_predictions = model.predict(
        context.train_split.X,
        whole_rule_pop=False,
        target_model=best_model_index,
    )
    test_predictions = model.predict(
        context.test_split.X,
        whole_rule_pop=False,
        target_model=best_model_index,
    )

    top_model_rules = model.get_model_rules(best_model_index)
    model_population = model.get_model_pop()
    rule_population = model.rule_population.export_rule_population()

    artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    top_model_rules.to_csv(artifacts.top_model_rules_path, index=False)
    model_population.to_csv(artifacts.model_population_path, index=False)
    rule_population.to_csv(artifacts.rule_population_path, index=False)

    summary = {
        "dataset_name": "MUX6",
        "config_path": str(DEFAULT_CONFIG_PATH),
        "train_path": str(context.dataset_definition.train_path),
        "test_path": str(context.dataset_definition.test_path),
        "best_model_index": int(best_model_index),
        "train_instances": len(context.train_split.instance_ids),
        "test_instances": len(context.test_split.instance_ids),
        "feature_names": list(context.train_split.feature_names),
        "train_accuracy": float(accuracy_score(context.train_split.y, train_predictions)),
        "test_accuracy": float(accuracy_score(context.test_split.y, test_predictions)),
        "top_model_rule_count": int(len(top_model_rules)),
        "model_population_size": int(len(model_population)),
        "rule_population_size": int(len(rule_population)),
        "train_classification_report": classification_report(
            context.train_split.y,
            train_predictions,
            digits=4,
            output_dict=True,
        ),
        "test_classification_report": classification_report(
            context.test_split.y,
            test_predictions,
            digits=4,
            output_dict=True,
        ),
    }
    artifacts.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def ensure_mux6_rule_artifacts(
    force_retrain: bool = False,
    output_dir: Path | None = None,
) -> Mux6RuleArtifacts:
    """Return saved MUX6 rule artifacts, retraining HEROS if needed."""

    resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    artifacts = _artifact_paths(resolved_output_dir)
    if force_retrain or not _artifacts_exist(artifacts):
        _train_and_export(artifacts)
    return artifacts


def load_mux6_rule_summary(output_dir: Path | None = None) -> Dict[str, Any]:
    """Load the saved summary JSON for the exported MUX6 rule artifacts."""

    artifacts = ensure_mux6_rule_artifacts(force_retrain=False, output_dir=output_dir)
    return json.loads(artifacts.summary_path.read_text(encoding="utf-8"))


def main() -> None:
    artifacts = ensure_mux6_rule_artifacts(force_retrain=False)
    summary = load_mux6_rule_summary(artifacts.output_dir)
    print("Saved MUX6 HEROS rule artifacts:")
    print("  output_dir =", artifacts.output_dir)
    print("  best_model_index =", summary["best_model_index"])
    print("  train_accuracy =", summary["train_accuracy"])
    print("  test_accuracy =", summary["test_accuracy"])
    print("  top_model_rule_count =", summary["top_model_rule_count"])
    print("  model_population_size =", summary["model_population_size"])
    print("  rule_population_size =", summary["rule_population_size"])


if __name__ == "__main__":
    main()
