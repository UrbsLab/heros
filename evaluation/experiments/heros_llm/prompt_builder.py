"""Prompt construction for audience and condition-specific explanations."""

from __future__ import annotations

import json
from typing import List

from .config import PromptConfig
from .data_models import FeatureGlossaryEntry, InstanceExplanationPacket, PromptBundle
from .glossary import select_glossary_entries
from .prompt_templates import (
    get_audience_addon,
    get_base_system_prompt,
    get_condition_template,
)
from .rule_text import (
    confidence_phrase,
    relevant_feature_names_from_rules,
    render_glossary_lines,
    render_instance_value_lines,
    render_prompt_rule_lines,
    render_rule_lines,
)


AUDIENCE_LABELS = {
    "layman": "Layman / patient",
    "clinician": "Clinician / informed user",
    "expert": "Expert / data scientist",
}


def should_include_metadata(audience: str, metadata_mode: str) -> bool:
    """Determine whether rule metadata should be exposed for the prompt."""
    if metadata_mode == "all":
        return True
    if metadata_mode == "clinician_and_expert":
        return audience in {"clinician", "expert"}
    if metadata_mode == "expert_only":
        return audience == "expert"
    return False


def select_condition_glossary(
    packet: InstanceExplanationPacket, condition: str
) -> List[FeatureGlossaryEntry]:
    """Return glossary entries for the prompt condition."""
    if condition != "condition_c":
        return []
    relevant_features = relevant_feature_names_from_rules(packet.active_rules)
    return select_glossary_entries(packet.glossary, relevant_features)


def _agreement_summary(packet: InstanceExplanationPacket) -> str:
    if packet.model_context.agreement_status == "single_rule":
        return "one rule matched and supports the prediction"
    if packet.model_context.agreement_status == "consistent":
        return "all active rules support the prediction"
    if packet.model_context.agreement_status == "mostly_supporting":
        return "most active rules support the prediction, but some conflict"
    if packet.model_context.agreement_status == "mixed":
        return "supporting and conflicting rules are both active"
    return "no matching active rules were provided"


def build_prompt_bundle(
    packet: InstanceExplanationPacket,
    condition: str,
    audience: str,
    prompt_config: PromptConfig,
) -> PromptBundle:
    """Build the versioned prompt bundle for one explanation request."""
    metadata_included = should_include_metadata(audience, prompt_config.metadata_mode)
    glossary_entries = select_condition_glossary(packet, condition)
    relevant_feature_names = relevant_feature_names_from_rules(packet.active_rules)
    confidence_text = confidence_phrase(
        audience=audience,
        predicted_probability=packet.model_context.predicted_class_probability,
        predicted_class=packet.model_context.prediction,
        prediction_probabilities=packet.model_context.prediction_probabilities,
        confidence_bin=packet.model_context.confidence_bin,
    )
    system_prompt = "{0}\n\n{1}".format(
        get_base_system_prompt(prompt_config.prompt_version),
        get_audience_addon(prompt_config.prompt_version, audience),
    )
    template = get_condition_template(prompt_config.prompt_version, condition)
    if prompt_config.prompt_version == "v1":
        active_rule_lines = render_rule_lines(packet.active_rules, include_metadata=metadata_included)
        instance_value_lines = render_instance_value_lines(packet.feature_values)
        selection_reason = packet.model_context.selection_reason or "none"
        prediction_probabilities = json.dumps(
            packet.model_context.prediction_probabilities, sort_keys=True
        )
    else:
        active_rule_lines = render_prompt_rule_lines(
            packet.active_rules,
            audience=audience,
            support_strong_threshold=prompt_config.rule_support_strong_threshold,
            support_moderate_threshold=prompt_config.rule_support_moderate_threshold,
        )
        instance_value_lines = render_instance_value_lines(
            packet.feature_values, selected_feature_names=relevant_feature_names
        )
        selection_reason = ""
        prediction_probabilities = ""

    user_prompt = template.format(
        audience_label=AUDIENCE_LABELS[audience],
        prediction=packet.model_context.prediction,
        prediction_probabilities=prediction_probabilities,
        confidence_text=confidence_text,
        covered=packet.model_context.covered,
        num_matching_rules=packet.model_context.num_matching_rules,
        num_supporting_rules=packet.model_context.num_supporting_rules,
        num_contradictory_rules=packet.model_context.num_contradictory_rules,
        agreement_status=packet.model_context.agreement_status,
        agreement_summary=_agreement_summary(packet),
        evidence_strength_label=packet.model_context.evidence_strength_label,
        selection_reason=selection_reason,
        instance_value_lines=instance_value_lines,
        active_rule_lines=active_rule_lines,
        glossary_lines=render_glossary_lines(glossary_entries),
    )
    return PromptBundle(
        condition=condition,
        audience=audience,
        prompt_version=prompt_config.prompt_version,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata_included=metadata_included,
        glossary_included=bool(glossary_entries),
    )
