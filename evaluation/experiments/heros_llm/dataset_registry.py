"""Dataset registry for MUX experiment splits."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    train_path: Path
    test_path: Path
    outcome_label: str = "Class"
    instance_id_label: str = "InstanceID"
    excluded_columns: List[str] = field(default_factory=lambda: ["Group"])
    expected_train_instances: int = 0
    expected_test_instances: int = 0
    total_instances: int = 0


DATASET_REGISTRY: Dict[str, DatasetDefinition] = {
    "MUX6": DatasetDefinition(
        name="MUX6",
        train_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/A_multiplexer_6_bit_500_inst_CV_Train_1.txt",
        test_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/A_multiplexer_6_bit_500_inst_CV_Test_1.txt",
        expected_train_instances=450,
        expected_test_instances=50,
        total_instances=500,
    ),
    "MUX11": DatasetDefinition(
        name="MUX11",
        train_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Train_1.txt",
        test_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/B_multiplexer_11_bit_5000_inst_CV_Test_1.txt",
        expected_train_instances=4500,
        expected_test_instances=500,
        total_instances=5000,
    ),
    "MUX20": DatasetDefinition(
        name="MUX20",
        train_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Train_1.txt",
        test_path=REPO_ROOT / "evaluation/datasets/partitioned/multiplexer/C_multiplexer_20_bit_10000_inst_CV_Test_1.txt",
        expected_train_instances=9000,
        expected_test_instances=1000,
        total_instances=10000,
    ),
}


def get_dataset_definition(name: str) -> DatasetDefinition:
    """Return a dataset definition by registry name."""
    if name not in DATASET_REGISTRY:
        raise KeyError("Unknown dataset: {0}".format(name))
    return DATASET_REGISTRY[name]


def list_dataset_names() -> List[str]:
    """Return available dataset names."""
    return sorted(DATASET_REGISTRY.keys())
