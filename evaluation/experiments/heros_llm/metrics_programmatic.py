"""Programmatic metric scaffolding for HEROS explanation evaluation."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .data_models import ActiveRule, InstanceExplanationPacket, ProgrammaticMetrics


RULE_TOKEN_PATTERN = re.compile(r"\bR\d+\b")
UNCERTAINTY_PATTERN = re.compile(
    r"\b(mixed|conflict|conflicting|uncertain|uncertainty|caveat|caution|may|might|limitation|leaned)\b",
    re.IGNORECASE,
)
CONFLICT_PATTERN = re.compile(r"\b(mixed|conflict|conflicting|opposing|contradict)\b", re.IGNORECASE)
CAUSAL_PATTERN = re.compile(
    r"\b(caused by|because of|leads to|results in|drives|proves that|shows that)\b",
    re.IGNORECASE,
)
REALITY_CLAIM_PATTERN = re.compile(
    r"\b(in reality|actually true|true in reality|proves the outcome|is ground truth|reflects ground truth|is definitely true)\b",
    re.IGNORECASE,
)
STRONG_CERTAINTY_PATTERN = re.compile(
    r"\b(definitely|certainly|guaranteed|always|undeniably|proof)\b",
    re.IGNORECASE,
)
GENERIC_FEATURE_LIKE_PATTERN = re.compile(
    r"\b(?:[A-Za-z]+(?:_[A-Za-z0-9]+)+|[A-Za-z]+(?:-\w+)+|[A-Z]{2,}[A-Z0-9_-]*)\b"
)
WORD_PATTERN = re.compile(r"\b\w+\b")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")
VOWEL_GROUP_PATTERN = re.compile(r"[aeiouy]+", re.IGNORECASE)
RESERVED_META_IDENTIFIERS = {
    "active-rules",
    "class-0",
    "class-1",
    "ground-truth",
    "model-based",
    "predicted-class",
    "rule-based",
    "support-label",
    "training-data",
    "training-instance",
    "training-instances",
    "vote-contribution",
}


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


def _prediction_explanation_agreement(packet: InstanceExplanationPacket, explanation_text: str) -> int:
    prediction = str(packet.model_context.prediction)
    class_labels = sorted(packet.model_context.prediction_probabilities.keys())
    mentioned_labels = [
        label
        for label in class_labels
        if re.search(r"\b{0}\b".format(re.escape(str(label))), explanation_text)
    ]
    if not mentioned_labels:
        return 1
    if prediction in mentioned_labels:
        return 1
    return 0


def _rule_coverage(key_rules: Sequence[ActiveRule], mentioned_features: Set[str]) -> float:
    if not key_rules:
        return 0.0
    hits = 0
    for rule in key_rules:
        rule_features = {condition.feature_name for condition in rule.conditions}
        if rule_features.intersection(mentioned_features):
            hits += 1
    return float(hits) / float(len(key_rules))


def _feature_alias_map(packet: InstanceExplanationPacket) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for feature_name in packet.feature_values.keys():
        alias_map[feature_name] = feature_name
    for entry in packet.glossary:
        alias_map[entry.feature_name] = entry.feature_name
        if entry.short_label:
            alias_map[entry.short_label] = entry.feature_name
    return alias_map


def _find_alias_mentions(text: str, alias_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    mentioned_features: Set[str] = set()
    matched_aliases: Set[str] = set()
    for alias, canonical_feature in alias_map.items():
        pattern = re.compile(r"(?<!\w){0}(?!\w)".format(re.escape(alias)), re.IGNORECASE)
        if pattern.search(text):
            mentioned_features.add(canonical_feature)
            matched_aliases.add(alias)
    return sorted(mentioned_features), sorted(matched_aliases)


def _extract_unsupported_feature_mentions(
    text: str,
    alias_map: Dict[str, str],
    supported_features: Set[str],
) -> List[str]:
    unsupported: Set[str] = set()
    generic_candidates = set(GENERIC_FEATURE_LIKE_PATTERN.findall(text))
    supported_identifiers = {feature for feature in supported_features if GENERIC_FEATURE_LIKE_PATTERN.fullmatch(feature)}
    supported_identifiers.update(
        alias for alias in alias_map.keys() if GENERIC_FEATURE_LIKE_PATTERN.fullmatch(alias)
    )
    for candidate in generic_candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in RESERVED_META_IDENTIFIERS:
            continue
        if candidate_lower.startswith("class-") or candidate_lower.endswith("-class"):
            continue
        if candidate not in supported_identifiers:
            unsupported.add(candidate)
    return sorted(unsupported)


def _count_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    groups = VOWEL_GROUP_PATTERN.findall(cleaned)
    syllables = len(groups)
    if cleaned.endswith("e") and not cleaned.endswith(("le", "ye")) and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def _readability_metrics(text: str) -> Tuple[float, float]:
    words = WORD_PATTERN.findall(text)
    sentences = [segment for segment in SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]
    sentence_count = max(len(sentences), 1)
    word_count = max(len(words), 1)
    syllable_count = sum(_count_syllables(word) for word in words)
    flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (
        syllable_count / word_count
    )
    flesch_kincaid_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (
        syllable_count / word_count
    ) - 15.59
    return flesch_reading_ease, flesch_kincaid_grade_level


def compute_programmatic_metrics(
    packet: InstanceExplanationPacket,
    explanation_text: str,
    top_k_rules: int = 3,
) -> ProgrammaticMetrics:
    """Compute objective, heuristic metrics from a packet and generated explanation."""
    alias_map = _feature_alias_map(packet)
    explicit_feature_mentions, matched_aliases = _find_alias_mentions(explanation_text, alias_map)
    supported_features = set(packet.feature_values.keys())
    unsupported_features = _extract_unsupported_feature_mentions(
        explanation_text,
        alias_map=alias_map,
        supported_features=supported_features,
    )
    unsupported_claim_spans = _extract_spans(REALITY_CLAIM_PATTERN, explanation_text)
    causal_spans = _extract_spans(CAUSAL_PATTERN, explanation_text)
    mentioned_feature_set = set(explicit_feature_mentions)
    key_rules = _derive_key_rules(packet, top_k_rules)
    key_features = _derive_key_features(key_rules)

    if explicit_feature_mentions:
        evidence_grounding_precision = float(
            len([feature for feature in explicit_feature_mentions if feature in supported_features])
        ) / float(len(explicit_feature_mentions))
    else:
        evidence_grounding_precision = None

    if key_features:
        key_evidence_coverage = float(
            len(key_features.intersection(mentioned_feature_set))
        ) / float(len(key_features))
    else:
        key_evidence_coverage = 0.0

    uncertainty_required = packet.model_context.conflict_present or packet.model_context.evidence_strength_label in {
        "mixed",
        "mostly_supporting",
        "no_match",
    }
    uncertainty_present = _mentions_uncertainty(explanation_text)
    conflict_present = packet.model_context.conflict_present
    conflict_mentioned = _mentions_conflict(explanation_text)
    if conflict_present:
        conflict_acknowledgment_score = 1.0 if conflict_mentioned else 0.0
    else:
        conflict_acknowledgment_score = 1.0 if not conflict_mentioned else 0.0

    confidence_wording_calibration = True
    if packet.model_context.evidence_strength_label != "strong" and STRONG_CERTAINTY_PATTERN.search(
        explanation_text
    ):
        confidence_wording_calibration = False

    flesch_reading_ease = None
    flesch_kincaid_grade_level = None
    if packet.audience == "layman":
        flesch_reading_ease, flesch_kincaid_grade_level = _readability_metrics(explanation_text)

    return ProgrammaticMetrics(
        evidence_grounding_precision=evidence_grounding_precision,
        hallucination_present=bool(unsupported_features or unsupported_claim_spans),
        unsupported_feature_mentions=unsupported_features,
        unsupported_claim_spans=unsupported_claim_spans,
        key_evidence_coverage=key_evidence_coverage,
        prediction_explanation_agreement=_prediction_explanation_agreement(packet, explanation_text),
        word_count=len(WORD_PATTERN.findall(explanation_text)),
        uncertainty_ack_required=uncertainty_required,
        uncertainty_ack_present=uncertainty_present,
        causal_overclaim_present=bool(causal_spans),
        rule_coverage=_rule_coverage(key_rules, mentioned_feature_set),
        conflict_acknowledgment_score=conflict_acknowledgment_score,
        confidence_wording_calibration=confidence_wording_calibration,
        flesch_reading_ease=flesch_reading_ease,
        flesch_kincaid_grade_level=flesch_kincaid_grade_level,
        raw_feature_mentions=explicit_feature_mentions,
        raw_rule_mentions=_extract_rule_mentions(explanation_text),
        raw_flags={
            "causal_spans": causal_spans,
            "conflict_mentioned": conflict_mentioned,
            "uncertainty_mentioned": uncertainty_present,
            "key_features": sorted(key_features),
            "key_rule_ids": [rule.rule_id for rule in key_rules],
            "matched_feature_aliases": matched_aliases,
        },
    )
