"""Tests for experiment runner sampling helpers."""

from __future__ import annotations

import unittest

from evaluation.experiments.heros_llm.data_models import (
    InstanceExplanationPacket,
    ModelContextSummary,
)
from evaluation.experiments.heros_llm.runner import _select_packets


def _packet(instance_id: int) -> InstanceExplanationPacket:
    return InstanceExplanationPacket(
        dataset_name="MUX6",
        split="test",
        instance_id=instance_id,
        feature_values={"A_0": 0},
        active_rules=[],
        model_context=ModelContextSummary(
            prediction=0,
            prediction_probabilities={"0": 0.7, "1": 0.3},
            predicted_class_probability=0.7,
            confidence_bin="moderate",
            covered=True,
            num_matching_rules=1,
            num_supporting_rules=1,
            num_contradictory_rules=0,
            agreement_status="single_rule",
            conflict_present=False,
            prediction_margin=0.4,
            selection_reason="test",
            evidence_strength_label="strong",
            train_instance_count=450,
        ),
        heros_description="demo",
    )


class SelectPacketsTests(unittest.TestCase):
    def test_select_packets_filters_explicit_instance_ids(self) -> None:
        packets = [_packet(101), _packet(165), _packet(303)]

        selected = _select_packets(
            packets=packets,
            sample_size=1,
            seed=42,
            use_full_test_set=False,
            instance_ids=["165"],
        )

        self.assertEqual([packet.instance_id for packet in selected], [165])

    def test_select_packets_raises_for_missing_instance_id(self) -> None:
        packets = [_packet(101), _packet(165)]

        with self.assertRaises(ValueError):
            _select_packets(
                packets=packets,
                sample_size=1,
                seed=42,
                use_full_test_set=False,
                instance_ids=["999"],
            )


if __name__ == "__main__":
    unittest.main()
