"""Classify source-versus-RDKit structure identity evidence, without changing structures."""

from __future__ import annotations

from typing import Literal

IdentityClass = Literal[
    "exact_match",
    "source_unspecified_stereochemical_block",
    "same_connectivity_other_inchi_layer_difference",
    "different_connectivity_block",
    "rdkit_unavailable",
    "invalid_inchikey_format",
]


def _parts(key: str) -> tuple[str, str, str] | None:
    """Return the three InChIKey components when the format is recognizable."""
    parts = str(key).strip().split("-")
    if len(parts) != 3 or [len(part) for part in parts] != [14, 10, 1]:
        return None
    if not all(part.isalpha() and part.isupper() for part in parts):
        return None
    return parts[0], parts[1], parts[2]


def classify_identity(
    rdkit_status: str,
    source_key: str,
    rdkit_key: str,
) -> IdentityClass:
    """Assign a conservative QA status; never infer whether a source is erroneous."""
    if rdkit_status != "parsed" or not rdkit_key:
        return "rdkit_unavailable"
    source_parts = _parts(source_key)
    rdkit_parts = _parts(rdkit_key)
    if source_parts is None or rdkit_parts is None:
        return "invalid_inchikey_format"
    if source_key == rdkit_key:
        return "exact_match"
    if source_parts[0] != rdkit_parts[0]:
        return "different_connectivity_block"
    if source_parts[1] == "UHFFFAOYSA":
        return "source_unspecified_stereochemical_block"
    return "same_connectivity_other_inchi_layer_difference"


def review_priority(identity_class: IdentityClass, fragment_count: int | None) -> str:
    """Prioritize manual review without automatically dropping any observation."""
    if identity_class in {
        "different_connectivity_block", "rdkit_unavailable", "invalid_inchikey_format"
    }:
        return "high"
    if identity_class == "same_connectivity_other_inchi_layer_difference":
        return "medium"
    if fragment_count is not None and fragment_count > 1:
        return "medium"
    if identity_class == "source_unspecified_stereochemical_block":
        return "review_by_policy"
    return "routine"
