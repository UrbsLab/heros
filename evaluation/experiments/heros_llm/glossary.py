"""Glossary builders for MUX feature names."""

from __future__ import annotations

import re
from typing import Iterable, List

from .data_models import FeatureGlossaryEntry


ADDRESS_PATTERN = re.compile(r"^A_(\d+)$")
REGISTER_PATTERN = re.compile(r"^R_(\d+)$")


def build_mux_glossary(feature_names: Iterable[str]) -> List[FeatureGlossaryEntry]:
    """Create one-sentence glossary entries for MUX feature names."""
    glossary: List[FeatureGlossaryEntry] = []
    for feature_name in feature_names:
        address_match = ADDRESS_PATTERN.match(feature_name)
        register_match = REGISTER_PATTERN.match(feature_name)
        if address_match:
            index = address_match.group(1)
            glossary.append(
                FeatureGlossaryEntry(
                    feature_name=feature_name,
                    short_label="Binary feature A_{0}".format(index),
                    one_sentence_definition=(
                        "Binary input feature A_{0} in the multiplexer dataset."
                    ).format(index),
                )
            )
        elif register_match:
            index = register_match.group(1)
            glossary.append(
                FeatureGlossaryEntry(
                    feature_name=feature_name,
                    short_label="Binary feature R_{0}".format(index),
                    one_sentence_definition=(
                        "Binary input feature R_{0} in the multiplexer dataset."
                    ).format(index),
                )
            )
    return glossary


def select_glossary_entries(
    glossary: Iterable[FeatureGlossaryEntry], feature_names: Iterable[str]
) -> List[FeatureGlossaryEntry]:
    """Filter glossary entries down to the requested feature names."""
    wanted = set(feature_names)
    return [entry for entry in glossary if entry.feature_name in wanted]
