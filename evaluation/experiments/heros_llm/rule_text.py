"""Rule and instance rendering helpers for prompt construction."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

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
        if metadata_parts:
            line += " Metadata: {0}.".format("; ".join(metadata_parts))
    return line


def render_rule_lines(rules: Iterable[ActiveRule], include_metadata: bool = True) -> str:
    """Render active rules as newline-separated text."""
    rule_lines = [render_rule_line(rule, include_metadata=include_metadata) for rule in rules]
    return "\n".join(rule_lines) if rule_lines else "No active rules were supplied."


def render_instance_value_lines(feature_values: Dict[str, Any]) -> str:
    """Render ordered instance feature values as bullet-like lines."""
    return "\n".join(
        "- {0}: {1}".format(feature_name, format_scalar(value))
        for feature_name, value in feature_values.items()
    )


def render_glossary_lines(glossary_entries: Iterable[FeatureGlossaryEntry]) -> str:
    """Render glossary entries as newline-separated lines."""
    entries = list(glossary_entries)
    if not entries:
        return "No glossary entries were provided."
    return "\n".join(
        "- {0}: {1}".format(entry.feature_name, entry.one_sentence_definition)
        for entry in entries
    )
