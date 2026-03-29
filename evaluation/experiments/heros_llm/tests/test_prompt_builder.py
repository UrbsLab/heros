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
            metadata=RuleMetadata(
                numerosity=2,
                accuracy=1.0,
                match_cover=10,
                correct_cover=10,
                predicted_class_share_given_match=1.0,
            ),
        )
        return InstanceExplanationPacket(
            dataset_name="MUX6",
            split="test",
            instance_id=8,
            feature_values={"A_0": 1, "R_0": 0, "UnusedFeature": 1},
            active_rules=[rule],
            model_context=ModelContextSummary(
                prediction=1,
                prediction_probabilities={"0": 0.0, "1": 1.0},
                predicted_class_probability=1.0,
                confidence_bin="strong",
                covered=True,
                num_matching_rules=1,
                num_supporting_rules=1,
                num_contradictory_rules=0,
                agreement_status="single_rule",
                conflict_present=False,
                prediction_margin=1.0,
                selection_reason=None,
                evidence_strength_label="strong",
                train_instance_count=450,
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

    def test_condition_b_omits_glossary_in_v2(self) -> None:
        prompt = build_prompt_bundle(self._packet(), "condition_b", "layman", PromptConfig())
        self.assertFalse(prompt.glossary_included)
        self.assertIn("No feature definitions are available in this condition.", prompt.user_prompt)

    def test_condition_c_includes_only_relevant_glossary(self) -> None:
        prompt = build_prompt_bundle(self._packet(), "condition_c", "expert", PromptConfig())
        self.assertTrue(prompt.glossary_included)
        self.assertIn("Feature definitions", prompt.user_prompt)
        self.assertIn("Binary input feature A_0 in the multiplexer dataset.", prompt.user_prompt)
        self.assertNotIn("UnusedFeature", prompt.user_prompt)

    def test_v2_prompt_excludes_heros_and_feature_indices(self) -> None:
        prompt = build_prompt_bundle(self._packet(), "condition_b", "clinician", PromptConfig())
        self.assertNotIn("HEROS", prompt.system_prompt)
        self.assertNotIn("feature_index", prompt.user_prompt)
        self.assertNotIn("Selection reason", prompt.user_prompt)

    def test_layman_excludes_support_counts_but_clinician_includes_them(self) -> None:
        layman_prompt = build_prompt_bundle(self._packet(), "condition_b", "layman", PromptConfig())
        clinician_prompt = build_prompt_bundle(
            self._packet(), "condition_b", "clinician", PromptConfig()
        )
        self.assertNotIn("matched 10 training instances", layman_prompt.user_prompt)
        self.assertIn("matched 10 training instances", clinician_prompt.user_prompt)

    def test_confidence_rendering_is_audience_specific(self) -> None:
        layman_prompt = build_prompt_bundle(self._packet(), "condition_b", "layman", PromptConfig())
        clinician_prompt = build_prompt_bundle(
            self._packet(), "condition_b", "clinician", PromptConfig()
        )
        expert_prompt = build_prompt_bundle(self._packet(), "condition_b", "expert", PromptConfig())
        self.assertIn("strongly leaned", layman_prompt.user_prompt)
        self.assertIn("about 100%", clinician_prompt.user_prompt)
        self.assertIn("full distribution", expert_prompt.user_prompt)


if __name__ == "__main__":
    unittest.main()
