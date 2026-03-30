"""Unit tests for programmatic explanation metrics."""

from __future__ import annotations

import unittest

from evaluation.experiments.heros_llm.data_models import (
    ActiveRule,
    InstanceExplanationPacket,
    ModelContextSummary,
    RuleCondition,
    RuleMetadata,
)
from evaluation.experiments.heros_llm.metrics_programmatic import compute_programmatic_metrics


class ProgrammaticMetricsTests(unittest.TestCase):
    def _packet(self) -> InstanceExplanationPacket:
        return InstanceExplanationPacket(
            dataset_name="MUX6",
            split="test",
            instance_id=8,
            feature_values={"A_0": 1, "A_1": 0, "R_0": 1},
            active_rules=[
                ActiveRule(
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
                        ),
                        RuleCondition(
                            feature_index=2,
                            feature_name="R_0",
                            operator="=",
                            value=1,
                            is_categorical=True,
                        ),
                    ],
                    if_then_text="",
                    metadata=RuleMetadata(
                        numerosity=2,
                        accuracy=1.0,
                        vote_contribution={"1": 2.0},
                    ),
                )
            ],
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
            glossary=[],
            audience="layman",
        )

    def test_evidence_precision_recall_f1_and_hallucination(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_0 and R_0 match the active rule.",
        )
        self.assertEqual(metrics.evidence_precision, 1.0)
        self.assertEqual(metrics.evidence_recall, 1.0)
        self.assertEqual(metrics.evidence_f1, 1.0)
        self.assertFalse(metrics.hallucination_present)

    def test_detects_unsupported_feature(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_9 appears important here.",
        )
        self.assertTrue(metrics.hallucination_present)
        self.assertIn("A_9", metrics.unsupported_feature_mentions)

    def test_reserved_meta_labels_do_not_count_as_unsupported_features(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            (
                "The model leaned toward class 1. "
                "Predicted class confidence was about 100%, and this was a model-based explanation."
            ),
        )
        self.assertFalse(metrics.hallucination_present)
        self.assertEqual(metrics.unsupported_feature_mentions, [])

    def test_generic_feature_names_are_grounded(self) -> None:
        packet = self._packet()
        packet.feature_values = {"Age": 54, "BMI": 31.2}
        packet.active_rules[0].conditions = [
            RuleCondition(
                feature_index=0,
                feature_name="Age",
                operator="=",
                value=54,
                is_categorical=True,
            )
        ]
        metrics = compute_programmatic_metrics(
            packet,
            "The model leaned toward class 1 because Age matched the active rule.",
        )
        self.assertEqual(metrics.evidence_precision, 1.0)
        self.assertFalse(metrics.hallucination_present)

    def test_partial_evidence_recall_and_f1(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_0 matched the active rule.",
        )
        self.assertEqual(metrics.evidence_precision, 1.0)
        self.assertEqual(metrics.evidence_recall, 0.5)
        self.assertAlmostEqual(metrics.evidence_f1 or 0.0, 2.0 / 3.0, places=6)

    def test_comprehensiveness_and_sufficiency_scaffolds_default_to_none(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_0 and R_0 match the active rule.",
        )
        self.assertIsNone(metrics.comprehensiveness)
        self.assertIsNone(metrics.sufficiency)
        self.assertIn("comprehensiveness_scaffold", metrics.raw_flags)
        self.assertIn("sufficiency_scaffold", metrics.raw_flags)

    def test_readability_only_for_layman(self) -> None:
        layman_metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_0 matched the rule.",
        )
        self.assertIsNotNone(layman_metrics.flesch_reading_ease)
        packet = self._packet()
        packet.audience = "expert"
        expert_metrics = compute_programmatic_metrics(
            packet,
            "The model leaned toward class 1 because A_0 matched the rule.",
        )
        self.assertIsNone(expert_metrics.flesch_reading_ease)


if __name__ == "__main__":
    unittest.main()
