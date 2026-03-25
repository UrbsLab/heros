"""Unit tests for rule and instance rendering."""

from __future__ import annotations

import unittest

from evaluation.experiments.heros_llm.data_models import ActiveRule, RuleCondition, RuleMetadata
from evaluation.experiments.heros_llm.rule_text import render_condition, render_rule_line


class RuleTextTests(unittest.TestCase):
    def test_render_categorical_condition(self) -> None:
        condition = RuleCondition(
            feature_index=0,
            feature_name="A_0",
            operator="=",
            value=1,
            is_categorical=True,
        )
        self.assertEqual(render_condition(condition), "A_0 = 1")

    def test_render_rule_line_with_metadata(self) -> None:
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
            if_then_text="",
            metadata=RuleMetadata(numerosity=3, accuracy=1.0, match_cover=12),
        )
        text = render_rule_line(rule, include_metadata=True)
        self.assertIn("Rule R1 [supports prediction]", text)
        self.assertIn("A_0 = 1", text)
        self.assertIn("numerosity=3", text)


if __name__ == "__main__":
    unittest.main()
