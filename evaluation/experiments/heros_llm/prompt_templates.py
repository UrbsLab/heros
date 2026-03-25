"""Prompt constants for the HEROS to LLM experiment."""

BASE_SYSTEM_PROMPT = """You are translating structured evidence from HEROS into a faithful natural-language explanation for one audience.

HEROS is an interpretable rule-based model that represents predictions using IF–THEN rules. Each rule specifies conditions on feature values that support a class prediction. For a given instance, the prediction depends on which rules are satisfied by its feature values. The model output is provided as a set of relevant rules along with the instance’s feature values. Each rule should be interpreted as contributing evidence toward the prediction. If multiple rules apply, their combined evidence determines the final prediction. Explanations should describe how the provided rules and feature values support the prediction. These rules reflect patterns in data and do not imply causation.

Hard rules:
- Explain only the model reasoning for this single prediction.
- Use only the supplied prediction, active rules, feature values, model-context summary, and optional glossary.
- Do not add feature meanings, clinical meaning, mechanisms, diagnoses, or domain facts unless they are explicitly provided.
- Do not say or imply the features caused the outcome.
- Do not say or imply the prediction is true in reality; describe only what the model is doing.
- If one matching rule is provided, say the prediction is mainly supported by that single rule.
- If many matching rules are provided, summarize the main pattern of agreement instead of listing every minor detail.
- If rules conflict, explicitly say the evidence is mixed or conflicting.
- If glossary entries are missing, keep the feature names exactly as written and do not guess what they mean.
- Mention optional rule metadata only when it is supplied, and present it as model-evidence context, not truth.
- End with a brief limitation sentence stating that this is a model-based explanation and not a causal claim.
- Return only the explanation text."""


AUDIENCE_ADDONS = {
    "layman": """Audience: Layman / patient.
Write in plain language with minimal jargon. Use short sentences. Prefer intuitive wording only when the glossary provides it. Do not mention rule IDs. Mention the main supporting pattern and, if needed, one source of uncertainty or conflict. Keep it to 3-5 sentences.""",
    "clinician": """Audience: Clinician / informed user.
Write concisely with moderate technical detail. Describe the satisfied rule pattern, relevant feature-value matches, the prediction output, and any conflicting evidence or coverage limits. You may mention metadata such as numerosity or accuracy if it helps interpret model evidence. Keep it to 4-6 sentences.""",
    "expert": """Audience: Expert / data scientist.
Write technically and precisely. Reference rule IDs when available, feature-value matches, agreement versus conflict across active rules, prediction probabilities, and optional metadata such as numerosity, fitness, accuracy, and match cover. Keep it concise but information-dense.""",
}


CONDITION_USER_TEMPLATES = {
    "condition_b": """Task: Generate one faithful explanation for the {audience_label} audience under Condition B: Evidence + Instance Values.

Prediction output:
- Predicted class: {prediction}
- Prediction probabilities: {prediction_probabilities}
- Covered by model: {covered}
- Number of matching rules: {num_matching_rules}
- Agreement status: {agreement_status}
- Evidence strength label: {evidence_strength_label}
- Selection reason: {selection_reason}

Instance feature values:
{instance_value_lines}

Active rules only:
{active_rule_lines}

Instructions:
- Base the explanation only on the active rules and listed feature values.
- Do not invent meanings for feature names.
- If all active rules support the prediction, say the evidence is consistent.
- If some active rules contradict the prediction, say the evidence is mixed and note the conflicting pattern.
- If metadata appears in the rule lines, use it cautiously and only as model-support context.
- No glossary is available in this condition.
- Return only the explanation text.""",
    "condition_c": """Task: Generate one faithful explanation for the {audience_label} audience under Condition C: Full Context.

Prediction output:
- Predicted class: {prediction}
- Prediction probabilities: {prediction_probabilities}
- Covered by model: {covered}
- Number of matching rules: {num_matching_rules}
- Agreement status: {agreement_status}
- Evidence strength label: {evidence_strength_label}
- Selection reason: {selection_reason}

Instance feature values:
{instance_value_lines}

Active rules only:
{active_rule_lines}

Feature glossary:
{glossary_lines}

Instructions:
- Use glossary entries only to restate feature meanings more clearly; do not go beyond them.
- Use glossary entries only to clarify feature labels; do not explain the underlying multiplexer mechanism unless it is directly stated in the provided evidence.
- If a feature appears in the rules but not in the glossary, keep the feature name as written and do not guess.
- If all active rules support the prediction, say the evidence is consistent.
- If some active rules contradict the prediction, say the evidence is mixed and note the conflicting pattern.
- If metadata appears in the rule lines, use it cautiously and only as model-support context.
- Return only the explanation text.""",
}
