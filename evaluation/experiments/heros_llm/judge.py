"""Judge evaluation scaffolding for subjective explanation metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import JudgeConfig
from .data_models import ExplanationRecord, JudgeMetrics


JUDGE_SYSTEM_PROMPT = """You are a strict evaluation assistant for a model explanation experiment.

Evaluate the explanation only against the supplied packet and audience label. Return JSON with:
- clarity_score: float from 0 to 1
- technical_appropriateness_score: float from 0 to 1
- judge_notes: short justification

Do not invent facts not present in the packet."""


@dataclass
class JudgePrompt:
    system_prompt: str
    user_prompt: str
    prompt_version: str


def build_judge_prompt(record: ExplanationRecord, judge_config: JudgeConfig) -> JudgePrompt:
    """Build the judge-model prompt for one explanation record."""
    user_prompt = """Audience: {audience}
Condition: {condition}
Prediction: {prediction}
Agreement status: {agreement_status}
Evidence strength: {evidence_strength}

Explanation:
{explanation}

Score clarity and technical appropriateness for the intended audience, then return only JSON.""".format(
        audience=record.prompt.audience,
        condition=record.prompt.condition,
        prediction=record.packet.model_context.prediction,
        agreement_status=record.packet.model_context.agreement_status,
        evidence_strength=record.packet.model_context.evidence_strength_label,
        explanation=record.generation.raw_text,
    )
    return JudgePrompt(
        system_prompt=JUDGE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        prompt_version=judge_config.prompt_version,
    )


def parse_judge_response(response_text: str, judge_config: JudgeConfig) -> JudgeMetrics:
    """Parse a JSON judge-model response into the experiment schema."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return JudgeMetrics(
            judge_notes=response_text,
            judge_model=judge_config.model,
            judge_prompt_version=judge_config.prompt_version,
        )
    return JudgeMetrics(
        clarity_score=payload.get("clarity_score"),
        technical_appropriateness_score=payload.get("technical_appropriateness_score"),
        judge_notes=payload.get("judge_notes", ""),
        judge_model=judge_config.model,
        judge_prompt_version=judge_config.prompt_version,
    )
