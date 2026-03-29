"""Unit tests for packet-construction helpers."""

from __future__ import annotations

import unittest

from evaluation.experiments.heros_llm.config import PromptConfig
from evaluation.experiments.heros_llm.heros_adapter import _confidence_bin, _convert_rule


class HerosAdapterTests(unittest.TestCase):
    def test_convert_rule_computes_train_support_fields(self) -> None:
        active_rule = _convert_rule(
            rule_payload={
                "rule_id": "17",
                "action": 0,
                "conditions": [
                    {
                        "feature_index": 2,
                        "feature_name": "R_2",
                        "operator": "=",
                        "type": "categorical",
                        "value": 0,
                        "human_readable": "R_2 = 0",
                    }
                ],
                "match_cover": 229,
                "correct_cover": 144,
                "accuracy": 144 / 229,
            },
            fallback_id="R1",
            prediction=0,
            train_instance_count=450,
        )

        self.assertAlmostEqual(active_rule.metadata.match_fraction_train, 229 / 450)
        self.assertAlmostEqual(active_rule.metadata.correct_fraction_train, 144 / 450)
        self.assertAlmostEqual(
            active_rule.metadata.predicted_class_share_given_match,
            144 / 229,
        )

    def test_confidence_bins_respect_boundaries(self) -> None:
        config = PromptConfig()
        self.assertEqual(_confidence_bin(0.80, config), "strong")
        self.assertEqual(_confidence_bin(0.65, config), "moderate")
        self.assertEqual(_confidence_bin(0.55, config), "slight_lean")
        self.assertEqual(_confidence_bin(0.54, config), "uncertain")


if __name__ == "__main__":
    unittest.main()
