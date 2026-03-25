# %% [markdown]
# # HEROS + LLM Single-Instance Demo
# This notebook reruns the `MUX6` experiment pipeline on a single held-out test
# instance, then shows all six explanation cases for that one packet:
# 
# - Condition B + Layman
# - Condition B + Clinician
# - Condition B + Expert
# - Condition C + Layman
# - Condition C + Clinician
# - Condition C + Expert
# 
# Unlike the earlier summary notebook, this one performs a fresh end-to-end run:
# 
# 1. train HEROS on the `450`-instance `MUX6` training fold
# 2. select one held-out `MUX6` test instance
# 3. generate all six explanations
# 4. compute programmatic metrics
# 5. run the judge model for `Clarity` and `Technical Appropriateness`
# 6. display the saved outputs

# %% [markdown]
# ## Demo Setup
# Change `DEMO_INSTANCE_ID` below if you want to rerun the same notebook for a
# different held-out `MUX6` test case. The default remains `165` because it
# activates three rules with one conflict, which makes the explanation differences
# easy to see during a live demo.

# %%
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from IPython.display import Markdown, display
except ImportError:
    def Markdown(text: str) -> str:
        return text

    def display(value: Any) -> None:
        print(value)

from evaluation.experiments.heros_llm.config import load_experiment_config
from evaluation.experiments.heros_llm.env_utils import discover_default_env_files, load_env_file
from evaluation.experiments.heros_llm.runner import run_experiment


CONFIG_PATH = Path("evaluation/experiments/heros_llm/configs/mux6_single_instance_demo.json")
DEMO_INSTANCE_ID = 165
CONDITION_ORDER = ["condition_b", "condition_c"]
AUDIENCE_ORDER = ["layman", "clinician", "expert"]

CONDITION_LABELS = {
    "condition_b": "Condition B: Evidence + Instance Values",
    "condition_c": "Condition C: Full Context",
}

AUDIENCE_LABELS = {
    "layman": "Layman / Patient",
    "clinician": "Clinician / Informed User",
    "expert": "Expert / Data Scientist",
}


def _parse_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _metric_cell(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


for env_path in discover_default_env_files():
    load_env_file(env_path)

config = load_experiment_config(str(CONFIG_PATH))
config.run_name = f"mux6_single_instance_demo_{DEMO_INSTANCE_ID}"
config.sampling.instance_ids = [str(DEMO_INSTANCE_ID)]
config.sampling.sample_size = 1
config.sampling.use_full_test_set = False
config.judge.enabled = True
config.llm.enabled = True

run_dir = Path(run_experiment(config))
records_df = pd.read_csv(run_dir / "records.csv")
demo_df = records_df[
    records_df["packet.instance_id"].astype(str) == str(DEMO_INSTANCE_ID)
].copy()

if len(demo_df) != 6:
    raise ValueError(
        f"Expected 6 rows for instance {DEMO_INSTANCE_ID}, found {len(demo_df)}."
    )

first_row = demo_df.iloc[0]
active_rules = _parse_json(first_row["packet.active_rules"]) or []
glossary_entries = _parse_json(first_row["packet.glossary"]) or []

display(
    Markdown(
        "\n".join(
            [
                "## Fresh Run Complete",
                "",
                f"- Run directory: `{run_dir}`",
                f"- Demo instance: `{DEMO_INSTANCE_ID}`",
                f"- Records: `{len(demo_df)}` explanation rows",
            ]
        )
    )
)

# %% [markdown]
# ## Selected HEROS Packet
# The tables below show the structured HEROS evidence for the single held-out test
# instance used in this fresh run.

# %%
packet_summary = pd.DataFrame(
    [
        {
            "dataset": first_row["packet.dataset_name"],
            "instance_id": int(first_row["packet.instance_id"]),
            "prediction": int(first_row["packet.model_context.prediction"]),
            "prob_class_0": float(first_row["packet.model_context.prediction_probabilities.0"]),
            "prob_class_1": float(first_row["packet.model_context.prediction_probabilities.1"]),
            "agreement_status": first_row["packet.model_context.agreement_status"],
            "conflict_present": _as_bool(first_row["packet.model_context.conflict_present"]),
            "matching_rules": int(first_row["packet.model_context.num_matching_rules"]),
            "supporting_rules": int(first_row["packet.model_context.num_supporting_rules"]),
            "contradictory_rules": int(first_row["packet.model_context.num_contradictory_rules"]),
            "evidence_strength": first_row["packet.model_context.evidence_strength_label"],
        }
    ]
)

feature_columns = sorted(
    column
    for column in demo_df.columns
    if column.startswith("packet.feature_values.")
)
feature_rows = []
for column in feature_columns:
    feature_name = column.split("packet.feature_values.", 1)[1]
    feature_rows.append({"feature": feature_name, "value": first_row[column]})
feature_df = pd.DataFrame(feature_rows)

rule_rows = []
for rule in active_rules:
    metadata = rule.get("metadata") or {}
    meta_parts = []
    for field in ["numerosity", "fitness", "accuracy", "match_cover", "correct_cover"]:
        value = metadata.get(field)
        if value is None:
            continue
        if isinstance(value, float):
            meta_parts.append(f"{field}={value:.4f}")
        else:
            meta_parts.append(f"{field}={value}")
    rule_rows.append(
        {
            "rule_id": rule["rule_id"],
            "support_label": (
                "supports prediction"
                if rule.get("supports_prediction")
                else "contradicts prediction"
            ),
            "predicted_class": rule["action"],
            "conditions": " AND ".join(
                condition["human_text"] for condition in rule.get("conditions", [])
            ),
            "metadata": "; ".join(meta_parts),
        }
    )
rules_df = pd.DataFrame(rule_rows)

display(packet_summary)
display(feature_df)
display(rules_df)

# %% [markdown]
# ## Condition C Glossary Entries
# These one-sentence feature definitions are included only for Condition C.

# %%
glossary_df = pd.DataFrame(glossary_entries)
display(
    glossary_df
    if not glossary_df.empty
    else pd.DataFrame(
        [{"feature_name": "None", "one_sentence_definition": "No glossary entries available."}]
    )
)

# %% [markdown]
# ## Six Demo Explanations
# These are the six outputs generated in the fresh single-instance run.

# %%
for condition in CONDITION_ORDER:
    display(Markdown(f"### {CONDITION_LABELS[condition]}"))

    for audience in AUDIENCE_ORDER:
        row = demo_df[
            (demo_df["generation.condition"] == condition)
            & (demo_df["generation.audience"] == audience)
        ].iloc[0]

        hallucination = _as_bool(row["programmatic_metrics.hallucination_present"])
        causal_overclaim = _as_bool(row["programmatic_metrics.causal_overclaim_present"])

        body = "\n".join(
            [
                f"#### {AUDIENCE_LABELS[audience]}",
                "",
                "**Explanation**",
                "",
                row["generation.raw_text"],
                "",
                (
                    "**Scores**: "
                    f"Clarity={_metric_cell(row['judge_metrics.clarity_score'])}, "
                    f"TAS={_metric_cell(row['judge_metrics.technical_appropriateness_score'])}, "
                    f"WordCount={int(float(row['programmatic_metrics.word_count']))}, "
                    f"Hallucination={hallucination}, "
                    f"CausalOverclaim={causal_overclaim}"
                ),
                "",
                f"**Judge note**: {row['judge_metrics.judge_notes']}",
            ]
        )
        display(Markdown(body))

# %% [markdown]
# ## Compact Comparison Table
# This final table gives a quick side-by-side view of the six fresh outputs.

# %%
comparison_df = demo_df[
    [
        "generation.condition",
        "generation.audience",
        "judge_metrics.clarity_score",
        "judge_metrics.technical_appropriateness_score",
        "programmatic_metrics.word_count",
        "programmatic_metrics.hallucination_present",
        "programmatic_metrics.causal_overclaim_present",
        "programmatic_metrics.feature_grounding_score",
        "programmatic_metrics.key_feature_coverage",
        "programmatic_metrics.prediction_consistency",
    ]
].copy()

comparison_df["generation.condition"] = comparison_df["generation.condition"].map(
    CONDITION_LABELS
)
comparison_df["generation.audience"] = comparison_df["generation.audience"].map(
    AUDIENCE_LABELS
)
comparison_df["programmatic_metrics.hallucination_present"] = (
    comparison_df["programmatic_metrics.hallucination_present"].map(_as_bool)
)
comparison_df["programmatic_metrics.causal_overclaim_present"] = (
    comparison_df["programmatic_metrics.causal_overclaim_present"].map(_as_bool)
)
comparison_df = comparison_df.rename(
    columns={
        "generation.condition": "condition",
        "generation.audience": "audience",
        "judge_metrics.clarity_score": "clarity",
        "judge_metrics.technical_appropriateness_score": "tas",
        "programmatic_metrics.word_count": "word_count",
        "programmatic_metrics.hallucination_present": "hallucination",
        "programmatic_metrics.causal_overclaim_present": "causal_overclaim",
        "programmatic_metrics.feature_grounding_score": "fgs",
        "programmatic_metrics.key_feature_coverage": "kfc",
        "programmatic_metrics.prediction_consistency": "pc",
    }
)

comparison_df["condition"] = pd.Categorical(
    comparison_df["condition"],
    categories=[CONDITION_LABELS[key] for key in CONDITION_ORDER],
    ordered=True,
)
comparison_df["audience"] = pd.Categorical(
    comparison_df["audience"],
    categories=[AUDIENCE_LABELS[key] for key in AUDIENCE_ORDER],
    ordered=True,
)
comparison_df = comparison_df.sort_values(["condition", "audience"]).reset_index(drop=True)
display(comparison_df)
