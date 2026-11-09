"""M1.8c: conservative, non-destructive chemical-identity policies.

These helpers NEVER aggregate experimental labels or change submitted SMILES.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from rdkit import Chem

StructureStatus = Literal[
    "provisional_exact_structure",
    "hold_rdkit_parse_failure",
    "hold_inchikey_connectivity_disagreement",
    "hold_multifragment",
]


def structure_id(canonical_isomeric_smiles: str) -> str:
    """Opaque identity token keyed to the full RDKit isomeric canonical SMILES.

    The policy/version are part of the hash preimage; do not assume hash IDs
    remain identical after future changes to chemical-standardization rules.
    """
    if not canonical_isomeric_smiles:
        raise ValueError("Cannot assign an ID without canonical isomeric SMILES")
    digest = hashlib.sha256(
        b"rdkit-isomeric-smiles-policy-v1\0"
        + canonical_isomeric_smiles.encode("utf-8")
    ).hexdigest()
    return f"RDKISO1-{digest}"


def classify_structure(qa: dict[str, str], flagged_raw_smiles: set[str]) -> StructureStatus:
    """Only promote parseable single-fragment, non-discrepant structures."""
    if qa["rdkit_status"] != "parsed":
        return "hold_rdkit_parse_failure"
    if (
        qa["smiles"] in flagged_raw_smiles
        or qa["source_key_comparison"] == "different_connectivity_block"
    ):
        return "hold_inchikey_connectivity_disagreement"
    fragment_count = qa["rdkit_fragment_count"].strip()
    if not fragment_count:
        raise ValueError("Parsed structure lacks RDKit fragment count")
    try:
        fragments = float(fragment_count)
    except ValueError as exc:
        raise ValueError(f"Invalid fragment count {fragment_count!r}") from exc
    if not fragments.is_integer() or fragments < 1:
        raise ValueError(f"Invalid fragment count {fragment_count!r}")
    if fragments != 1:
        return "hold_multifragment"
    if not qa["rdkit_canonical_isomeric_smiles"]:
        raise ValueError("Parsed structure lacks canonical isomeric SMILES")
    return "provisional_exact_structure"


def unspecified_stereogenic_elements(canonical_isomeric_smiles: str) -> int | None:
    """Count potentially stereogenic atoms/bonds without assigned stereo.

    Diagnostic only. Potential stereochemistry is not proof that experimental
    material contained a single specified stereoisomer.
    """
    if not canonical_isomeric_smiles:
        return None
    molecule = Chem.MolFromSmiles(canonical_isomeric_smiles)
    if molecule is None:
        raise ValueError("Previously parsed canonical SMILES no longer parses")
    return sum(
        item.specified == Chem.StereoSpecified.Unspecified
        for item in Chem.FindPotentialStereo(molecule)
    )
