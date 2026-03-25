"""Unit tests for audience and condition prompt rendering."""

from __future__ import annotations

import unittest

from evaluation.experiments.heros_llm.config import PromptConfig
from evaluation.experiments.heros_llm.data_models import (
    ActiveRule,
    FeatureGlossaryEntry,
    InstanceExplanationPacket,
    ModelContextSummary,
    RuleCondition,
    RuleMetadata,
)
from evaluation.experiments.heros_llm.prompt_builder import build_prompt_bundle


class PromptBuilderTests(unittest.TestCase):
    def _packet(self) -> InstanceExplanationPacket:
        rule = ActiveRule(
            rule_id="R1",
            action=1,
            supports_prediction=True,
            contradicts_prediction=False,
            conditions=[
                RuleCondition(
                    feature_index=0,
                    feature_name="A_0",
                    operator="=",
                    value=1,
                    is_categorical=True,
                )
            ],
            if_then_text="Rule R1",
            metadata=RuleMetadata(numerosity=2, accuracy=1.0),
        )
        return InstanceExplanationPacket(
            dataset_name="MUX6",
            split="test",
            instance_id=8,
            feature_values={"A_0": 1, "R_0": 0},
            active_rules=[rule],
            model_context=ModelContextSummary(
                prediction=1,
                prediction_probabilities={"0": 0.0, "1": 1.0},
                covered=True,
                num_matching_rules=1,
                num_supporting_rules=1,
                num_contradictory_rules=0,
                agreement_status="single_rule",
                conflict_present=False,
                prediction_margin=1.0,
                selection_reason=None,
                evidence_strength_label="strong",
            ),
            heros_description="desc",
            glossary=[
                FeatureGlossaryEntry(
                    feature_name="A_0",
                    short_label="Binary feature A_0",
                    one_sentence_definition="Binary input feature A_0 in the multiplexer dataset.",
                )
            ],
        )

    def test_condition_b_omits_glossary(self) -> None:
        prompt = build_prompt_bundle(self._packet(), "condition_b", "layman", PromptConfig())
        self.assertFalse(prompt.glossary_included)
        self.assertIn("No glossary is available in this condition.", prompt.user_prompt)

    def test_condition_c_includes_glossary(self) -> None:
        prompt = build_prompt_bundle(self._packet(), "condition_c", "expert", PromptConfig())
        self.assertTrue(prompt.glossary_included)
        self.assertIn("Feature glossary", prompt.user_prompt)
        self.assertIn("Binary input feature A_0 in the multiplexer dataset.", prompt.user_prompt)
        self.assertIn("do not explain the underlying multiplexer mechanism", prompt.user_prompt)


if __name__ == "__main__":
    unittest.main()
