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
            glossary=[],
        )

    def test_feature_grounding_and_hallucination(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_0 and R_0 match the active rule.",
        )
        self.assertEqual(metrics.feature_grounding_score, 1.0)
        self.assertFalse(metrics.hallucination_present)

    def test_detects_unsupported_feature(self) -> None:
        metrics = compute_programmatic_metrics(
            self._packet(),
            "The model leaned toward class 1 because A_9 appears important here.",
        )
        self.assertTrue(metrics.hallucination_present)
        self.assertIn("A_9", metrics.unsupported_feature_mentions)


if __name__ == "__main__":
    unittest.main()
