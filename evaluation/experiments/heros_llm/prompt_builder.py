"""Prompt construction for audience and condition-specific explanations."""

from __future__ import annotations

import json
from typing import Iterable, List

from .config import PromptConfig
from .data_models import FeatureGlossaryEntry, InstanceExplanationPacket, PromptBundle
from .prompt_templates import AUDIENCE_ADDONS, BASE_SYSTEM_PROMPT, CONDITION_USER_TEMPLATES
from .rule_text import render_glossary_lines, render_instance_value_lines, render_rule_lines


AUDIENCE_LABELS = {
    "layman": "Layman / patient",
    "clinician": "Clinician / informed user",
    "expert": "Expert / data scientist",
}


def should_include_metadata(audience: str, metadata_mode: str) -> bool:
    """Determine whether rule metadata should be exposed for the prompt."""
    if metadata_mode == "all":
        return True
    if metadata_mode == "expert_only":
        return audience == "expert"
    return False


def select_condition_glossary(
    packet: InstanceExplanationPacket, condition: str
) -> List[FeatureGlossaryEntry]:
    """Return glossary entries for the prompt condition."""
    if condition != "condition_c":
        return []
    return list(packet.glossary)


def build_prompt_bundle(
    packet: InstanceExplanationPacket,
    condition: str,
    audience: str,
    prompt_config: PromptConfig,
) -> PromptBundle:
    """Build the versioned prompt bundle for one explanation request."""
    if condition not in CONDITION_USER_TEMPLATES:
        raise KeyError("Unknown condition: {0}".format(condition))
    if audience not in AUDIENCE_ADDONS:
        raise KeyError("Unknown audience: {0}".format(audience))

    metadata_included = should_include_metadata(audience, prompt_config.metadata_mode)
    glossary_entries = select_condition_glossary(packet, condition)
    system_prompt = "{0}\n\n{1}".format(BASE_SYSTEM_PROMPT, AUDIENCE_ADDONS[audience])
    user_prompt = CONDITION_USER_TEMPLATES[condition].format(
        audience_label=AUDIENCE_LABELS[audience],
        prediction=packet.model_context.prediction,
        prediction_probabilities=json.dumps(packet.model_context.prediction_probabilities, sort_keys=True),
        covered=packet.model_context.covered,
        num_matching_rules=packet.model_context.num_matching_rules,
        agreement_status=packet.model_context.agreement_status,
        evidence_strength_label=packet.model_context.evidence_strength_label,
        selection_reason=packet.model_context.selection_reason or "none",
        instance_value_lines=render_instance_value_lines(packet.feature_values),
        active_rule_lines=render_rule_lines(packet.active_rules, include_metadata=metadata_included),
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
