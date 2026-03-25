"""Programmatic metric scaffolding for HEROS explanation evaluation."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set

from .data_models import ActiveRule, InstanceExplanationPacket, ProgrammaticMetrics


FEATURE_TOKEN_PATTERN = re.compile(r"\b[A-Z]_\d+\b")
RULE_TOKEN_PATTERN = re.compile(r"\bR\d+\b")
UNCERTAINTY_PATTERN = re.compile(
    r"\b(mixed|conflict|conflicting|uncertain|uncertainty|caveat|caution|may|might|limitation)\b",
    re.IGNORECASE,
)
CONFLICT_PATTERN = re.compile(r"\b(mixed|conflict|conflicting|opposing|contradict)\b", re.IGNORECASE)
CAUSAL_PATTERN = re.compile(
    r"\b(caused by|because of|leads to|results in|proves that|shows that)\b",
    re.IGNORECASE,
)
REALITY_CLAIM_PATTERN = re.compile(
    r"\b(in reality|actually true|true in reality|proves the outcome)\b",
    re.IGNORECASE,
)
STRONG_CERTAINTY_PATTERN = re.compile(
    r"\b(definitely|certainly|guaranteed|always|undeniably|proof)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b\w+\b")


def _extract_explicit_feature_mentions(text: str) -> List[str]:
    return sorted(set(FEATURE_TOKEN_PATTERN.findall(text)))


def _extract_rule_mentions(text: str) -> List[str]:
    return sorted(set(RULE_TOKEN_PATTERN.findall(text)))


def _rule_weight(rule: ActiveRule, prediction: str) -> float:
    contribution = rule.metadata.vote_contribution or {}
    if prediction in contribution:
        return float(contribution[prediction])
    if rule.metadata.numerosity is not None and rule.metadata.accuracy is not None:
        return float(rule.metadata.numerosity) * float(rule.metadata.accuracy)
    if rule.metadata.numerosity is not None:
        return float(rule.metadata.numerosity)
    return 1.0


def _derive_key_rules(packet: InstanceExplanationPacket, top_k: int) -> List[ActiveRule]:
    prediction = str(packet.model_context.prediction)
    supporting_rules = [rule for rule in packet.active_rules if rule.supports_prediction]
    supporting_rules.sort(key=lambda rule: _rule_weight(rule, prediction), reverse=True)
    return supporting_rules[:top_k]


def _derive_key_features(rules: Sequence[ActiveRule]) -> Set[str]:
    features: Set[str] = set()
    for rule in rules:
        for condition in rule.conditions:
            features.add(condition.feature_name)
    return features


def _mentions_conflict(text: str) -> bool:
    return bool(CONFLICT_PATTERN.search(text))


def _mentions_uncertainty(text: str) -> bool:
    return bool(UNCERTAINTY_PATTERN.search(text))


def _extract_spans(pattern: re.Pattern[str], text: str) -> List[str]:
    return sorted(set(match.group(0) for match in pattern.finditer(text)))


def _prediction_consistency(packet: InstanceExplanationPacket, explanation_text: str) -> int:
    prediction = str(packet.model_context.prediction)
    class_labels = sorted(packet.model_context.prediction_probabilities.keys())
    mentioned_labels = [label for label in class_labels if re.search(r"\b{0}\b".format(re.escape(str(label))), explanation_text)]
    if not mentioned_labels:
        return 1
    if prediction in mentioned_labels:
        return 1
    return 0


def _rule_mention_coverage(key_rules: Sequence[ActiveRule], mentioned_features: Set[str]) -> float:
    if not key_rules:
        return 0.0
    hits = 0
    for rule in key_rules:
        rule_features = {condition.feature_name for condition in rule.conditions}
        if rule_features.intersection(mentioned_features):
            hits += 1
    return float(hits) / float(len(key_rules))


def compute_programmatic_metrics(
    packet: InstanceExplanationPacket,
    explanation_text: str,
    top_k_rules: int = 3,
) -> ProgrammaticMetrics:
    """Compute objective, heuristic metrics from a packet and generated explanation."""
    explicit_feature_mentions = _extract_explicit_feature_mentions(explanation_text)
    supported_features = set(packet.feature_values.keys())
    unsupported_features = sorted(set(explicit_feature_mentions) - supported_features)
    unsupported_claim_spans = _extract_spans(REALITY_CLAIM_PATTERN, explanation_text)
    causal_spans = _extract_spans(CAUSAL_PATTERN, explanation_text)
    mentioned_feature_set = set(explicit_feature_mentions)
    key_rules = _derive_key_rules(packet, top_k_rules)
    key_features = _derive_key_features(key_rules)

    if explicit_feature_mentions:
        feature_grounding_score = float(
            len([feature for feature in explicit_feature_mentions if feature in supported_features])
        ) / float(len(explicit_feature_mentions))
    else:
        feature_grounding_score = None

    if key_features:
        key_feature_coverage = float(len(key_features.intersection(mentioned_feature_set))) / float(len(key_features))
    else:
        key_feature_coverage = 0.0

    uncertainty_required = packet.model_context.conflict_present or packet.model_context.evidence_strength_label in {
        "mixed",
        "mostly_supporting",
        "no_match",
    }
    uncertainty_present = _mentions_uncertainty(explanation_text)
    conflict_present = packet.model_context.conflict_present
    conflict_mentioned = _mentions_conflict(explanation_text)
    if conflict_present:
        conflict_awareness_score = 1.0 if conflict_mentioned else 0.0
    else:
        conflict_awareness_score = 1.0 if not conflict_mentioned else 0.0

    confidence_calibration_pass = True
    if packet.model_context.evidence_strength_label != "strong" and STRONG_CERTAINTY_PATTERN.search(explanation_text):
        confidence_calibration_pass = False

    return ProgrammaticMetrics(
        feature_grounding_score=feature_grounding_score,
        hallucination_present=bool(unsupported_features or unsupported_claim_spans),
        unsupported_feature_mentions=unsupported_features,
        unsupported_claim_spans=unsupported_claim_spans,
        key_feature_coverage=key_feature_coverage,
        prediction_consistency=_prediction_consistency(packet, explanation_text),
        word_count=len(WORD_PATTERN.findall(explanation_text)),
        uncertainty_ack_required=uncertainty_required,
        uncertainty_ack_present=uncertainty_present,
        causal_overclaim_present=bool(causal_spans),
        rule_mention_coverage=_rule_mention_coverage(key_rules, mentioned_feature_set),
        conflict_awareness_score=conflict_awareness_score,
        confidence_calibration_pass=confidence_calibration_pass,
        raw_feature_mentions=explicit_feature_mentions,
        raw_rule_mentions=_extract_rule_mentions(explanation_text),
        raw_flags={
            "causal_spans": causal_spans,
            "conflict_mentioned": conflict_mentioned,
            "uncertainty_mentioned": uncertainty_present,
            "key_features": sorted(key_features),
            "key_rule_ids": [rule.rule_id for rule in key_rules],
        },
    )
