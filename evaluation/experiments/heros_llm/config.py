"""Configuration objects and JSON loading helpers for the experiment pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from .data_models import SerializableDataclass, to_serializable

T = TypeVar("T")


DEFAULT_HEROS_DESCRIPTION = (
    "HEROS is an interpretable rule-based model that represents predictions using IF–THEN rules. "
    "Each rule specifies conditions on feature values that support a class prediction. For a given "
    "instance, the prediction depends on which rules are satisfied by its feature values. The model "
    "output is provided as a set of relevant rules along with the instance's feature values. Each "
    "rule should be interpreted as contributing evidence toward the prediction. If multiple rules "
    "apply, their combined evidence determines the final prediction. Explanations should describe "
    "how the provided rules and feature values support the prediction. These rules reflect patterns "
    "in data and do not imply causation."
)


@dataclass
class SamplingConfig(SerializableDataclass):
    strategy: str = "stratified_prediction_rule_bucket"
    sample_size: int = 50
    seed: int = 42
    use_full_test_set: bool = True


@dataclass
class HerosConfig(SerializableDataclass):
    outcome_type: str = "class"
    iterations: int = 10000
    pop_size: int = 500
    cross_prob: float = 0.8
    mut_prob: float = 0.04
    nu: float = 1.0
    beta: float = 0.2
    theta_sel: float = 0.5
    fitness_function: str = "pareto"
    subsumption: str = "both"
    rsl: int = 0
    feat_track: Optional[str] = None
    model_iterations: int = 40
    model_pop_size: int = 100
    model_pop_init: str = "target_acc"
    new_gen: float = 1.0
    merge_prob: float = 0.1
    rule_pop_init: Optional[str] = None
    compaction: Optional[str] = "sub"
    track_performance: int = 0
    model_tracking: bool = False
    stored_rule_iterations: Optional[str] = None
    stored_model_iterations: Optional[str] = None
    random_state: int = 42
    verbose: bool = False
    target_model_index: Optional[int] = None


@dataclass
class PromptConfig(SerializableDataclass):
    heros_description: str = DEFAULT_HEROS_DESCRIPTION
    prompt_version: str = "v1"
    prediction_prompt_mode: str = "explicit"
    metadata_mode: str = "expert_only"
    strong_evidence_threshold: float = 0.8
    mixed_evidence_threshold: float = 0.6
    key_rules_top_k: int = 3
    conditions: List[str] = field(default_factory=lambda: ["condition_b", "condition_c"])
    audiences: List[str] = field(default_factory=lambda: ["layman", "clinician", "expert"])


@dataclass
class LLMConfig(SerializableDataclass):
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 400
    request_timeout_seconds: int = 60


@dataclass
class JudgeConfig(SerializableDataclass):
    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 300
    prompt_version: str = "v1"


@dataclass
class OutputConfig(SerializableDataclass):
    base_dir: str = "output/heros_llm"
    write_packets: bool = True
    write_prompts: bool = True
    write_generations: bool = True
    write_records: bool = True
    write_csv: bool = True
    cache_dir_name: str = "cache"


@dataclass
class ExperimentConfig(SerializableDataclass):
    run_name: str = "mux6_50_test"
    dataset_name: str = "MUX6"
    split: str = "test"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    heros: HerosConfig = field(default_factory=HerosConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_dataclass(model_cls: Type[T], payload: Dict[str, Any]) -> T:
    kwargs: Dict[str, Any] = {}
    for field_name, field_info in model_cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
        if field_name not in payload:
            continue
        value = payload[field_name]
        field_type = field_info.type
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[field_name] = _build_dataclass(field_type, value)
        else:
            kwargs[field_name] = value
    return model_cls(**kwargs)


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load an experiment config from a JSON file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ExperimentConfig(
        run_name=payload.get("run_name", ExperimentConfig.run_name),
        dataset_name=payload.get("dataset_name", ExperimentConfig.dataset_name),
        split=payload.get("split", ExperimentConfig.split),
        sampling=_build_dataclass(SamplingConfig, payload.get("sampling", {})),
        heros=_build_dataclass(HerosConfig, payload.get("heros", {})),
        prompt=_build_dataclass(PromptConfig, payload.get("prompt", {})),
        llm=_build_dataclass(LLMConfig, payload.get("llm", {})),
        judge=_build_dataclass(JudgeConfig, payload.get("judge", {})),
        output=_build_dataclass(OutputConfig, payload.get("output", {})),
    )


def save_config_snapshot(config: ExperimentConfig, path: str) -> None:
    """Persist a JSON snapshot of the resolved experiment config."""
    snapshot_path = Path(path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(config), handle, indent=2, sort_keys=True)
