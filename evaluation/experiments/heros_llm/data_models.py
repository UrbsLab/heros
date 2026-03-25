"""Serializable data models for the HEROS to LLM experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_scalar(value: Any) -> Any:
    """Best-effort conversion for non-standard scalar objects."""
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def to_serializable(value: Any) -> Any:
    """Recursively convert dataclass content into JSON-safe primitives."""
    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return _normalize_scalar(value)


@dataclass
class SerializableDataclass:
    """Common serializer mixin for experiment dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return to_serializable(self)


@dataclass
class RuleCondition(SerializableDataclass):
    feature_index: int
    feature_name: str
    operator: str
    value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    is_categorical: bool = True
    human_text: str = ""


@dataclass
class RuleMetadata(SerializableDataclass):
    fitness: Optional[float] = None
    numerosity: Optional[int] = None
    accuracy: Optional[float] = None
    match_cover: Optional[int] = None
    correct_cover: Optional[int] = None
    useful_accuracy: Optional[float] = None
    useful_coverage: Optional[float] = None
    vote_contribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActiveRule(SerializableDataclass):
    rule_id: str
    action: Any
    supports_prediction: bool
    contradicts_prediction: bool
    conditions: List[RuleCondition]
    if_then_text: str
    metadata: RuleMetadata = field(default_factory=RuleMetadata)


@dataclass
class ModelContextSummary(SerializableDataclass):
    prediction: Any
    prediction_probabilities: Dict[str, float]
    covered: bool
    num_matching_rules: int
    num_supporting_rules: int
    num_contradictory_rules: int
    agreement_status: str
    conflict_present: bool
    prediction_margin: float
    selection_reason: Optional[str]
    evidence_strength_label: str


@dataclass
class FeatureGlossaryEntry(SerializableDataclass):
    feature_name: str
    short_label: str
    one_sentence_definition: str
    source: str = "mux_default"


@dataclass
class InstanceExplanationPacket(SerializableDataclass):
    dataset_name: str
    split: str
    instance_id: Any
    feature_values: Dict[str, Any]
    active_rules: List[ActiveRule]
    model_context: ModelContextSummary
    heros_description: str
    glossary: List[FeatureGlossaryEntry] = field(default_factory=list)
    condition: Optional[str] = None
    audience: Optional[str] = None
    prompt_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptBundle(SerializableDataclass):
    condition: str
    audience: str
    prompt_version: str
    system_prompt: str
    user_prompt: str
    metadata_included: bool
    glossary_included: bool


@dataclass
class GeneratedExplanation(SerializableDataclass):
    condition: str
    audience: str
    system_prompt: str
    user_prompt: str
    raw_text: str
    model_name: str
    temperature: float
    created_at: str


@dataclass
class ProgrammaticMetrics(SerializableDataclass):
    feature_grounding_score: Optional[float] = None
    hallucination_present: bool = False
    unsupported_feature_mentions: List[str] = field(default_factory=list)
    unsupported_claim_spans: List[str] = field(default_factory=list)
    key_feature_coverage: Optional[float] = None
    prediction_consistency: Optional[int] = None
    word_count: int = 0
    uncertainty_ack_required: bool = False
    uncertainty_ack_present: bool = False
    causal_overclaim_present: bool = False
    rule_mention_coverage: Optional[float] = None
    conflict_awareness_score: Optional[float] = None
    confidence_calibration_pass: bool = True
    raw_feature_mentions: List[str] = field(default_factory=list)
    raw_rule_mentions: List[str] = field(default_factory=list)
    raw_flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeMetrics(SerializableDataclass):
    clarity_score: Optional[float] = None
    technical_appropriateness_score: Optional[float] = None
    judge_notes: str = ""
    judge_model: Optional[str] = None
    judge_prompt_version: Optional[str] = None


@dataclass
class ExperimentMetadata(SerializableDataclass):
    run_id: str
    git_sha: str
    dataset_name: str
    split: str
    sample_size: int
    sample_seed: int
    sampling_strategy: str
    target_model_selection: str
    llm_model: str
    judge_model: str
    temperature: float
    timestamp: str
    config_hash: str
    train_path: str = ""
    test_path: str = ""


@dataclass
class ExplanationRecord(SerializableDataclass):
    experiment_metadata: ExperimentMetadata
    packet: InstanceExplanationPacket
    prompt: PromptBundle
    generation: GeneratedExplanation
    programmatic_metrics: ProgrammaticMetrics = field(default_factory=ProgrammaticMetrics)
    judge_metrics: JudgeMetrics = field(default_factory=JudgeMetrics)


def dataclass_field_names(model_cls: Any) -> List[str]:
    """Return declared dataclass field names for a model."""
    return [field_info.name for field_info in fields(model_cls)]
