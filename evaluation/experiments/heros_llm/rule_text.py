"""Rule and instance rendering helpers for prompt construction."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .data_models import ActiveRule, FeatureGlossaryEntry, RuleCondition


def format_scalar(value: Any) -> str:
    """Render a scalar value in a stable, human-readable way."""
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        if value.is_integer():
            return str(int(value))
        return "{0:.4f}".format(value).rstrip("0").rstrip(".")
    return str(value)


def render_condition(condition: RuleCondition) -> str:
    """Convert a rule condition into readable IF-clause text."""
    if condition.is_categorical or condition.operator == "=":
        return "{0} = {1}".format(condition.feature_name, format_scalar(condition.value))
    return "{0} in [{1}, {2}]".format(
        condition.feature_name,
        format_scalar(condition.min_value),
        format_scalar(condition.max_value),
    )


def render_rule_line(rule: ActiveRule, include_metadata: bool = True) -> str:
    """Render a single active rule line for the LLM prompt."""
    support_label = "supports prediction" if rule.supports_prediction else "contradicts prediction"
    condition_text = " AND ".join(render_condition(condition) for condition in rule.conditions)
    line = "Rule {0} [{1}]: IF {2} THEN predict class {3}.".format(
        rule.rule_id,
        support_label,
        condition_text,
        format_scalar(rule.action),
    )
    if include_metadata:
        metadata_parts: List[str] = []
        if rule.metadata.numerosity is not None:
            metadata_parts.append("numerosity={0}".format(rule.metadata.numerosity))
        if rule.metadata.fitness is not None:
            metadata_parts.append("fitness={0}".format(format_scalar(rule.metadata.fitness)))
        if rule.metadata.accuracy is not None:
            metadata_parts.append("accuracy={0}".format(format_scalar(rule.metadata.accuracy)))
        if rule.metadata.match_cover is not None:
            metadata_parts.append("match_cover={0}".format(rule.metadata.match_cover))
        if rule.metadata.correct_cover is not None:
            metadata_parts.append("correct_cover={0}".format(rule.metadata.correct_cover))
        if rule.metadata.match_fraction_train is not None:
            metadata_parts.append(
                "match_fraction_train={0}".format(format_scalar(rule.metadata.match_fraction_train))
            )
        if rule.metadata.correct_fraction_train is not None:
            metadata_parts.append(
                "correct_fraction_train={0}".format(format_scalar(rule.metadata.correct_fraction_train))
            )
        if rule.metadata.predicted_class_share_given_match is not None:
            metadata_parts.append(
                "predicted_class_share_given_match={0}".format(
                    format_scalar(rule.metadata.predicted_class_share_given_match)
                )
            )
        if metadata_parts:
            line += " Metadata: {0}.".format("; ".join(metadata_parts))
    return line


def render_rule_lines(rules: Iterable[ActiveRule], include_metadata: bool = True) -> str:
    """Render active rules as newline-separated text."""
    rule_lines = [render_rule_line(rule, include_metadata=include_metadata) for rule in rules]
    return "\n".join(rule_lines) if rule_lines else "No active rules were supplied."


def render_instance_value_lines(
    feature_values: Dict[str, Any], selected_feature_names: Optional[Sequence[str]] = None
) -> str:
    """Render ordered instance feature values as bullet-like lines."""
    wanted = set(selected_feature_names) if selected_feature_names is not None else None
    lines = [
        "- {0}: {1}".format(feature_name, format_scalar(value))
        for feature_name, value in feature_values.items()
        if wanted is None or feature_name in wanted
    ]
    return "\n".join(lines) if lines else "No relevant instance feature values were supplied."


def render_glossary_lines(glossary_entries: Iterable[FeatureGlossaryEntry]) -> str:
    """Render glossary entries as newline-separated lines."""
    entries = list(glossary_entries)
    if not entries:
        return "No glossary entries were provided."
    return "\n".join(
        "- {0}: {1}".format(entry.feature_name, entry.one_sentence_definition)
        for entry in entries
    )


def relevant_feature_names_from_rules(rules: Iterable[ActiveRule]) -> List[str]:
    """Return feature names used by the supplied active rules in stable order."""
    seen = set()
    ordered: List[str] = []
    for rule in rules:
        for condition in rule.conditions:
            if condition.feature_name not in seen:
                seen.add(condition.feature_name)
                ordered.append(condition.feature_name)
    return ordered


def confidence_phrase(
    audience: str,
    predicted_probability: float,
    predicted_class: Any,
    prediction_probabilities: Dict[str, float],
    confidence_bin: str,
) -> str:
    """Render audience-specific confidence wording."""
    layman_labels = {
        "strong": "The model strongly leaned toward class {0}.",
        "moderate": "The model leaned toward class {0}.",
        "slight_lean": "The model slightly leaned toward class {0}.",
        "uncertain": "The model was uncertain and only slightly favored class {0}.",
    }
    qualitative = {
        "strong": "strong",
        "moderate": "moderate",
        "slight_lean": "slight lean",
        "uncertain": "very mixed",
    }
    if audience == "layman":
        return layman_labels[confidence_bin].format(format_scalar(predicted_class))
    if audience == "clinician":
        percentage = int(round(predicted_probability * 100.0))
        return "Predicted class confidence: about {0}% ({1}).".format(
            percentage,
            qualitative[confidence_bin],
        )
    ordered = sorted(prediction_probabilities.items(), key=lambda item: item[0])
    distribution = ", ".join(
        "class {0}={1:.3f}".format(label, float(probability))
        for label, probability in ordered
    )
    return "Predicted class probability: {0:.3f} ({1}); full distribution: {2}.".format(
        predicted_probability,
        qualitative[confidence_bin],
        distribution,
    )


def rule_support_label(rule: ActiveRule, strong_threshold: float, moderate_threshold: float) -> str:
    """Return a qualitative support label from matched-training support."""
    share = rule.metadata.predicted_class_share_given_match
    if share is None and rule.metadata.accuracy is not None:
        share = rule.metadata.accuracy
    if share is None:
        return "unknown"
    if float(share) >= strong_threshold:
        return "strong"
    if float(share) >= moderate_threshold:
        return "moderate"
    return "mixed"


def render_prompt_rule_line(
    rule: ActiveRule,
    audience: str,
    support_strong_threshold: float,
    support_moderate_threshold: float,
) -> str:
    """Render a prompt-facing rule line with audience-specific detail."""
    support_label = "supports the predicted class" if rule.supports_prediction else "supports a different class"
    condition_text = " AND ".join(render_condition(condition) for condition in rule.conditions)
    if audience == "layman":
        return "- IF {0}, the rule points to class {1} and {2}.".format(
            condition_text,
            format_scalar(rule.action),
            support_label,
        )

    qualitative_support = rule_support_label(
        rule,
        strong_threshold=support_strong_threshold,
        moderate_threshold=support_moderate_threshold,
    )
    share = rule.metadata.predicted_class_share_given_match
    share_text = (
        "about {0}% of matched training instances had class {1}".format(
            int(round(float(share) * 100.0)),
            format_scalar(rule.action),
        )
        if share is not None
        else "training class share was unavailable"
    )
    match_cover_text = (
        "matched {0} training instances".format(rule.metadata.match_cover)
        if rule.metadata.match_cover is not None
        else "training match count unavailable"
    )

    if audience == "clinician":
        return "- IF {0}, this rule predicts class {1}. Training support: {2}; {3}; support label: {4}.".format(
            condition_text,
            format_scalar(rule.action),
            match_cover_text,
            share_text,
            qualitative_support,
        )

    metadata_parts: List[str] = [match_cover_text, share_text, "support label: {0}".format(qualitative_support)]
    if rule.metadata.correct_cover is not None:
        metadata_parts.append("correct_cover={0}".format(rule.metadata.correct_cover))
    if rule.metadata.accuracy is not None:
        metadata_parts.append("accuracy={0}".format(format_scalar(rule.metadata.accuracy)))
    if rule.metadata.fitness is not None:
        metadata_parts.append("fitness={0}".format(format_scalar(rule.metadata.fitness)))
    if rule.metadata.numerosity is not None:
        metadata_parts.append("numerosity={0}".format(rule.metadata.numerosity))
    if rule.metadata.vote_contribution:
        ordered_votes = ", ".join(
            "{0}={1}".format(label, format_scalar(value))
            for label, value in sorted(rule.metadata.vote_contribution.items())
        )
        metadata_parts.append("vote_contribution={0}".format(ordered_votes))
    return "- Rule {0}: IF {1}, predict class {2}; {3}.".format(
        rule.rule_id,
        condition_text,
        format_scalar(rule.action),
        "; ".join(metadata_parts),
    )


def render_prompt_rule_lines(
    rules: Iterable[ActiveRule],
    audience: str,
    support_strong_threshold: float,
    support_moderate_threshold: float,
) -> str:
    """Render prompt-facing rules with audience-specific detail."""
    rendered = [
        render_prompt_rule_line(
            rule,
            audience=audience,
            support_strong_threshold=support_strong_threshold,
            support_moderate_threshold=support_moderate_threshold,
        )
        for rule in rules
    ]
    return "\n".join(rendered) if rendered else "No active rules were supplied."
