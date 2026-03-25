"""Training, packet extraction, and dataset preparation around HEROS."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

from .config import ExperimentConfig, PromptConfig
from .data_models import (
    ActiveRule,
    FeatureGlossaryEntry,
    InstanceExplanationPacket,
    ModelContextSummary,
    RuleCondition,
    RuleMetadata,
)
from .dataset_registry import DatasetDefinition
from .glossary import build_mux_glossary
from .rule_text import render_rule_line

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class SplitData:
    X: Any
    y: Any
    feature_names: List[str]
    instance_ids: List[Any]


@dataclass
class TrainedHerosContext:
    model: Any
    dataset_definition: DatasetDefinition
    train_split: SplitData
    test_split: SplitData
    target_model_index: int
    glossary: List[FeatureGlossaryEntry]


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _load_split(dataset_definition: DatasetDefinition, split: str) -> SplitData:
    import pandas as pd

    path = dataset_definition.train_path if split == "train" else dataset_definition.test_path
    dataframe = pd.read_csv(path, sep="\t")
    columns_to_drop = [dataset_definition.outcome_label] + list(dataset_definition.excluded_columns)
    features_df = dataframe.drop(columns=columns_to_drop, errors="ignore")
    feature_names = list(features_df.columns)
    instance_ids = (
        dataframe[dataset_definition.instance_id_label].tolist()
        if dataset_definition.instance_id_label in dataframe.columns
        else list(range(len(dataframe)))
    )
    return SplitData(
        X=features_df.values,
        y=dataframe[dataset_definition.outcome_label].values,
        feature_names=feature_names,
        instance_ids=instance_ids,
    )


def train_heros_model(
    config: ExperimentConfig, dataset_definition: DatasetDefinition
) -> TrainedHerosContext:
    """Train HEROS on the dataset train split and select the target model."""
    sys.path.append(Path(__file__).resolve().parents[2] / "src")
    from skheros.heros import HEROS

    train_split = _load_split(dataset_definition, "train")
    test_split = _load_split(dataset_definition, "test")
    cat_feat_indexes = list(range(len(train_split.feature_names)))
    model = HEROS(
        outcome_type=config.heros.outcome_type,
        iterations=config.heros.iterations,
        pop_size=config.heros.pop_size,
        cross_prob=config.heros.cross_prob,
        mut_prob=config.heros.mut_prob,
        nu=config.heros.nu,
        beta=config.heros.beta,
        theta_sel=config.heros.theta_sel,
        fitness_function=config.heros.fitness_function,
        subsumption=config.heros.subsumption,
        rsl=config.heros.rsl,
        feat_track=config.heros.feat_track,
        model_iterations=config.heros.model_iterations,
        model_pop_size=config.heros.model_pop_size,
        model_pop_init=config.heros.model_pop_init,
        new_gen=config.heros.new_gen,
        merge_prob=config.heros.merge_prob,
        rule_pop_init=config.heros.rule_pop_init,
        compaction=config.heros.compaction,
        track_performance=config.heros.track_performance,
        model_tracking=config.heros.model_tracking,
        stored_rule_iterations=config.heros.stored_rule_iterations,
        stored_model_iterations=config.heros.stored_model_iterations,
        random_state=config.heros.random_state,
        verbose=config.heros.verbose,
    )
    model.fit(
        train_split.X,
        train_split.y,
        row_id=train_split.instance_ids,
        cat_feat_indexes=cat_feat_indexes,
        ek=None,
    )
    if config.heros.target_model_index is not None:
        target_model_index = config.heros.target_model_index
    else:
        target_model_index = model.auto_select_top_model(
            test_split.X, test_split.y, verbose=config.heros.verbose
        )
    return TrainedHerosContext(
        model=model,
        dataset_definition=dataset_definition,
        train_split=train_split,
        test_split=test_split,
        target_model_index=target_model_index,
        glossary=build_mux_glossary(train_split.feature_names),
    )


def _prediction_margin(prediction_probabilities: Dict[str, float]) -> float:
    values = sorted(prediction_probabilities.values(), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def _agreement_status(covered: bool, num_matching_rules: int, num_contradictory_rules: int) -> str:
    if not covered or num_matching_rules == 0:
        return "no_match"
    if num_matching_rules == 1:
        return "single_rule"
    if num_contradictory_rules == 0:
        return "consistent"
    return "mostly_supporting" if num_contradictory_rules < num_matching_rules else "mixed"


def _evidence_strength_label(
    agreement_status: str,
    covered: bool,
    margin: float,
    num_contradictory_rules: int,
    prompt_config: PromptConfig,
) -> str:
    if not covered or agreement_status == "no_match":
        return "no_match"
    if num_contradictory_rules > 0:
        if margin >= prompt_config.strong_evidence_threshold:
            return "mostly_supporting"
        return "mixed"
    if margin >= prompt_config.strong_evidence_threshold:
        return "strong"
    return "mixed"


def _convert_rule(rule_payload: Dict[str, Any], fallback_id: str, prediction: Any) -> ActiveRule:
    conditions: List[RuleCondition] = []
    for condition in rule_payload.get("conditions", []):
        operator = condition.get("operator", "=")
        is_categorical = condition.get("type") == "categorical"
        conditions.append(
            RuleCondition(
                feature_index=int(condition.get("feature_index", 0)),
                feature_name=str(condition.get("feature_name")),
                operator=operator,
                value=_normalize_scalar(condition.get("value")),
                min_value=_normalize_scalar(condition.get("min")),
                max_value=_normalize_scalar(condition.get("max")),
                is_categorical=is_categorical,
                human_text=str(condition.get("human_readable", "")),
            )
        )

    metadata = RuleMetadata(
        fitness=_normalize_scalar(rule_payload.get("fitness")),
        numerosity=_normalize_scalar(rule_payload.get("numerosity")),
        accuracy=_normalize_scalar(rule_payload.get("accuracy")),
        match_cover=_normalize_scalar(rule_payload.get("match_cover")),
        correct_cover=_normalize_scalar(rule_payload.get("correct_cover")),
        useful_accuracy=_normalize_scalar(rule_payload.get("useful_accuracy")),
        useful_coverage=_normalize_scalar(rule_payload.get("useful_coverage")),
        vote_contribution={
            str(key): float(value)
            for key, value in (rule_payload.get("vote_contribution") or {}).items()
        },
    )
    action = _normalize_scalar(rule_payload.get("action"))
    supports_prediction = str(action) == str(prediction)
    active_rule = ActiveRule(
        rule_id=str(rule_payload.get("rule_id") or fallback_id),
        action=action,
        supports_prediction=supports_prediction,
        contradicts_prediction=not supports_prediction,
        conditions=conditions,
        if_then_text="",
        metadata=metadata,
    )
    active_rule.if_then_text = render_rule_line(active_rule, include_metadata=True)
    return active_rule


def build_packet_for_instance(
    context: TrainedHerosContext,
    x_instance: Any,
    instance_id: Any,
    split: str,
    prompt_config: PromptConfig,
) -> InstanceExplanationPacket:
    """Build a serializable explanation packet for a single instance."""
    structured = context.model.predict_explanation(
        x_instance,
        context.train_split.feature_names,
        whole_rule_pop=False,
        target_model=context.target_model_index,
        verbose=False,
    )

    feature_values: Dict[str, Any] = OrderedDict()
    for feature in structured.get("features", []):
        feature_values[str(feature["feature_name"])] = _normalize_scalar(feature["value"])

    prediction = _normalize_scalar(structured.get("outcome_prediction"))
    supporting_rules = list(structured.get("supporting_rules", []))
    contradictory_rules = list(structured.get("contradictory_rules", []))
    all_rule_payloads = supporting_rules + contradictory_rules
    active_rules = [
        _convert_rule(rule_payload, "R{0}".format(index + 1), prediction)
        for index, rule_payload in enumerate(all_rule_payloads)
    ]

    prediction_probabilities = {
        str(key): float(value)
        for key, value in (structured.get("prediction_probabilities") or {}).items()
    }
    covered = bool(structured.get("covered"))
    num_matching_rules = int(structured.get("num_matching_rules", len(active_rules)))
    num_supporting_rules = len([rule for rule in active_rules if rule.supports_prediction])
    num_contradictory_rules = len([rule for rule in active_rules if rule.contradicts_prediction])
    margin = _prediction_margin(prediction_probabilities)
    agreement_status = _agreement_status(covered, num_matching_rules, num_contradictory_rules)
    evidence_strength_label = _evidence_strength_label(
        agreement_status,
        covered,
        margin,
        num_contradictory_rules,
        prompt_config,
    )

    model_context = ModelContextSummary(
        prediction=prediction,
        prediction_probabilities=prediction_probabilities,
        covered=covered,
        num_matching_rules=num_matching_rules,
        num_supporting_rules=num_supporting_rules,
        num_contradictory_rules=num_contradictory_rules,
        agreement_status=agreement_status,
        conflict_present=num_contradictory_rules > 0,
        prediction_margin=margin,
        selection_reason=structured.get("selection_reason"),
        evidence_strength_label=evidence_strength_label,
    )

    return InstanceExplanationPacket(
        dataset_name=context.dataset_definition.name,
        split=split,
        instance_id=_normalize_scalar(instance_id),
        feature_values=feature_values,
        active_rules=active_rules,
        model_context=model_context,
        heros_description=prompt_config.heros_description,
        glossary=list(context.glossary),
        prompt_flags={
            "prediction_prompt_mode": prompt_config.prediction_prompt_mode,
            "metadata_mode": prompt_config.metadata_mode,
        },
    )


def build_packets_for_split(
    context: TrainedHerosContext,
    split: str,
    prompt_config: PromptConfig,
) -> List[InstanceExplanationPacket]:
    """Build explanation packets for every instance in a split."""
    split_data = context.test_split if split == "test" else context.train_split
    packets: List[InstanceExplanationPacket] = []
    for row_index, x_instance in enumerate(split_data.X):
        packets.append(
            build_packet_for_instance(
                context=context,
                x_instance=x_instance,
                instance_id=split_data.instance_ids[row_index],
                split=split,
                prompt_config=prompt_config,
            )
        )
    return packets
