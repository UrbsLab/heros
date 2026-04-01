"""Training, packet extraction, and dataset preparation around HEROS."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import ast
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    heros_train_accuracy: Optional[float]
    heros_test_accuracy: Optional[float]
    heros_test_balanced_accuracy: Optional[float]
    heros_test_coverage: Optional[float]
    heros_top_model_rule_count: Optional[int]
    heros_rule_population_size: Optional[int]
    heros_model_population_size: Optional[int]
    ideal_rules_in_top_model: Optional[int]
    ideal_rule_fraction_in_top_model: Optional[float]


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


def _load_split_with_config(
    config: ExperimentConfig, dataset_definition: DatasetDefinition, split: str
) -> SplitData:
    import pandas as pd

    path = dataset_definition.train_path if split == "train" else dataset_definition.test_path
    dataframe = pd.read_csv(path, sep="\t")
    columns_to_drop = [dataset_definition.outcome_label] + list(dataset_definition.excluded_columns)
    for column in config.extra_excluded_columns:
        if column not in columns_to_drop:
            columns_to_drop.append(column)
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


def _int_to_binary_list(num: int, width: int) -> List[int]:
    return [int(bit) for bit in format(num, "0{0}b".format(width))]


def _gen_mux_ideal_rules(num_bits: int) -> Set[Tuple[str, str, int]]:
    address_bits = {6: 2, 11: 3, 20: 4, 37: 5, 70: 6, 135: 7}
    if num_bits not in address_bits:
        return set()
    ideal_rules: Set[Tuple[str, str, int]] = set()
    register_bits = num_bits - address_bits[num_bits]
    for register_index in range(register_bits):
        condition_indexes = list(range(address_bits[num_bits])) + [
            register_index + address_bits[num_bits]
        ]
        zero_values = _int_to_binary_list(register_index, address_bits[num_bits]) + [0]
        one_values = _int_to_binary_list(register_index, address_bits[num_bits]) + [1]
        ideal_rules.add((str(condition_indexes), str(zero_values), 0))
        ideal_rules.add((str(condition_indexes), str(one_values), 1))
    return ideal_rules


def _canonical_rule_signature(condition_indexes: Sequence[Any], condition_values: Sequence[Any], action: Any) -> Tuple[str, str, int]:
    paired = sorted(
        [
            (int(index), int(value))
            for index, value in zip(condition_indexes, condition_values)
        ],
        key=lambda item: item[0],
    )
    indexes = [item[0] for item in paired]
    values = [item[1] for item in paired]
    return str(indexes), str(values), int(action)


def _mux_bits_from_dataset_name(dataset_name: str) -> Optional[int]:
    if dataset_name.startswith("MUX"):
        try:
            return int(dataset_name.replace("MUX", ""))
        except ValueError:
            return None
    return None


def _count_ideal_rules_in_model(model_rules_df: Any, dataset_name: str) -> Tuple[Optional[int], Optional[float]]:
    mux_bits = _mux_bits_from_dataset_name(dataset_name)
    if mux_bits is None:
        return None, None
    ideal_rules = _gen_mux_ideal_rules(mux_bits)
    if not ideal_rules:
        return None, None
    normalized_ideal_rules = {
        _canonical_rule_signature(ast.literal_eval(indexes), ast.literal_eval(values), action)
        for indexes, values, action in ideal_rules
    }
    found = 0
    for _, row in model_rules_df.iterrows():
        candidate = _canonical_rule_signature(
            ast.literal_eval(str(row["Condition Indexes"])),
            ast.literal_eval(str(row["Condition Values"])),
            row["Action"],
        )
        if candidate in normalized_ideal_rules:
            found += 1
    return found, (float(found) / float(len(ideal_rules)) if ideal_rules else None)


def train_heros_model(
    config: ExperimentConfig, dataset_definition: DatasetDefinition
) -> TrainedHerosContext:
    """Train HEROS on the dataset train split and select the target model."""
    sys.path.append(Path(__file__).resolve().parents[2] / "src")
    from skheros.heros import HEROS
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    train_split = _load_split_with_config(config, dataset_definition, "train")
    test_split = _load_split_with_config(config, dataset_definition, "test")
    cat_feat_indexes = list(range(len(train_split.feature_names)))
    model = HEROS(
        mode=config.heros.mode,
        feedback=config.heros.feedback,
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
    train_predictions = model.predict(
        train_split.X,
        whole_rule_pop=False,
        target_model=target_model_index,
    )
    test_predictions = model.predict(
        test_split.X,
        whole_rule_pop=False,
        target_model=target_model_index,
    )
    test_covered = model.predict_covered(
        test_split.X,
        whole_rule_pop=False,
        target_model=target_model_index,
    )
    model_rules_df = model.get_model_rules(target_model_index)
    ideal_rules_in_top_model, ideal_rule_fraction_in_top_model = _count_ideal_rules_in_model(
        model_rules_df,
        dataset_definition.name,
    )
    return TrainedHerosContext(
        model=model,
        dataset_definition=dataset_definition,
        train_split=train_split,
        test_split=test_split,
        target_model_index=target_model_index,
        glossary=build_mux_glossary(train_split.feature_names),
        heros_train_accuracy=float(accuracy_score(train_split.y, train_predictions)),
        heros_test_accuracy=float(accuracy_score(test_split.y, test_predictions)),
        heros_test_balanced_accuracy=float(
            balanced_accuracy_score(test_split.y, test_predictions)
        ),
        heros_test_coverage=float(sum(test_covered) / len(test_covered)) if len(test_covered) else None,
        heros_top_model_rule_count=int(len(model_rules_df)),
        heros_rule_population_size=int(len(model.get_pop())),
        heros_model_population_size=int(len(model.get_model_pop())),
        ideal_rules_in_top_model=ideal_rules_in_top_model,
        ideal_rule_fraction_in_top_model=ideal_rule_fraction_in_top_model,
    )


def _prediction_margin(prediction_probabilities: Dict[str, float]) -> float:
    values = sorted(prediction_probabilities.values(), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def _predicted_class_probability(prediction: Any, prediction_probabilities: Dict[str, float]) -> float:
    return float(prediction_probabilities.get(str(prediction), 0.0))


def _confidence_bin(predicted_probability: float, prompt_config: PromptConfig) -> str:
    if predicted_probability >= prompt_config.confidence_strong_threshold:
        return "strong"
    if predicted_probability >= prompt_config.confidence_moderate_threshold:
        return "moderate"
    if predicted_probability >= prompt_config.confidence_slight_threshold:
        return "slight_lean"
    return "uncertain"


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


def _convert_rule(
    rule_payload: Dict[str, Any],
    fallback_id: str,
    prediction: Any,
    train_instance_count: int,
) -> ActiveRule:
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

    match_cover = _normalize_scalar(rule_payload.get("match_cover"))
    correct_cover = _normalize_scalar(rule_payload.get("correct_cover"))
    accuracy = _normalize_scalar(rule_payload.get("accuracy"))
    if match_cover not in (None, 0) and correct_cover is not None:
        predicted_class_share_given_match = float(correct_cover) / float(match_cover)
    else:
        predicted_class_share_given_match = None
    match_fraction_train = (
        float(match_cover) / float(train_instance_count)
        if match_cover is not None and train_instance_count
        else None
    )
    correct_fraction_train = (
        float(correct_cover) / float(train_instance_count)
        if correct_cover is not None and train_instance_count
        else None
    )

    metadata = RuleMetadata(
        fitness=_normalize_scalar(rule_payload.get("fitness")),
        numerosity=_normalize_scalar(rule_payload.get("numerosity")),
        accuracy=accuracy,
        match_cover=match_cover,
        correct_cover=correct_cover,
        match_fraction_train=match_fraction_train,
        correct_fraction_train=correct_fraction_train,
        predicted_class_share_given_match=predicted_class_share_given_match,
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
        _convert_rule(
            rule_payload,
            "R{0}".format(index + 1),
            prediction,
            len(context.train_split.instance_ids),
        )
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
    predicted_probability = _predicted_class_probability(prediction, prediction_probabilities)
    agreement_status = _agreement_status(covered, num_matching_rules, num_contradictory_rules)
    evidence_strength_label = _evidence_strength_label(
        agreement_status,
        covered,
        margin,
        num_contradictory_rules,
        prompt_config,
    )
    confidence_bin = _confidence_bin(predicted_probability, prompt_config)

    model_context = ModelContextSummary(
        prediction=prediction,
        prediction_probabilities=prediction_probabilities,
        predicted_class_probability=predicted_probability,
        confidence_bin=confidence_bin,
        covered=covered,
        num_matching_rules=num_matching_rules,
        num_supporting_rules=num_supporting_rules,
        num_contradictory_rules=num_contradictory_rules,
        agreement_status=agreement_status,
        conflict_present=num_contradictory_rules > 0,
        prediction_margin=margin,
        selection_reason=structured.get("selection_reason"),
        evidence_strength_label=evidence_strength_label,
        train_instance_count=len(context.train_split.instance_ids),
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
