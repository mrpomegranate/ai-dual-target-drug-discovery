"""M1.8a: non-destructive RDKit molecular-identity QA.

Keep every observation; do not strip salts, neutralize, collapse stereochemistry,
reconcile potency labels, or allocate train/test folds in this stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger, rdBase

RDLogger.DisableLog("rdApp.error")

REQUIRED = {
    "reactant_set_id",
    "target",
    "smiles",
    "inchi_key",
    "m17_partition",
    "measurement_status",
    "source",
}


@dataclass(frozen=True)
class StructureQA:
    status: str
    canonical_isomeric_smiles: str | None
    rdkit_inchi_key: str | None
    fragment_count: int | None
    heavy_atom_count: int | None


def profile_structure(smiles: str | None) -> StructureQA:
    """Parse one *original* structure; do not edit its chemistry."""
    if not isinstance(smiles, str) or not smiles.strip():
        return StructureQA("missing_smiles", None, None, None, None)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return StructureQA("rdkit_parse_failure", None, None, None, None)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        try:
            key = Chem.MolToInchiKey(mol)
        except (RuntimeError, ValueError):
            key = None
        return StructureQA(
            "parsed" if key else "inchi_generation_failure",
            canonical,
            key,
            len(Chem.GetMolFrags(mol)),
            mol.GetNumHeavyAtoms(),
        )
    except (RuntimeError, ValueError, TypeError):
        return StructureQA("rdkit_parse_failure", None, None, None, None)


def compare_inchikey(source: str | None, computed: str | None) -> str:
    """Distinguish whole-key agreement from connectivity-block agreement."""
    if not computed:
        return "rdkit_key_unavailable"
    if not isinstance(source, str) or not source.strip():
        return "source_key_missing"
    reported = source.strip().upper()
    if reported == computed:
        return "exact_match"
    if reported.split("-")[0] == computed.split("-")[0]:
        return "same_connectivity_block_only"
    return "different_connectivity_block"


def sha256_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def summarize(df: pd.DataFrame, identity: pd.DataFrame) -> dict:
    """Count raw identity QA; never infer independent compounds from record counts."""
    def counts(series: pd.Series) -> dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    valid = identity[identity["rdkit_canonical_isomeric_smiles"].notna()]
    by_partition = {}
    for (target, role), group in identity.groupby(["target", "m17_partition"], dropna=False):
        by_partition[f"{target}/{role}"] = {
            "observations": len(group),
            "distinct_reported_inchikeys": int(group["inchi_key"].nunique()),
            "distinct_rdkit_inchikeys": int(group["rdkit_inchi_key"].nunique()),
            "rdkit_parse_failures": int((group["rdkit_status"] == "rdkit_parse_failure").sum()),
            "rdkit_key_different_connectivity": int(
                (group["source_key_comparison"] == "different_connectivity_block").sum()
            ),
        }
    source_key_to_smiles = valid.groupby("inchi_key")[
        "rdkit_canonical_isomeric_smiles"
    ].nunique()
    rdkit_key_to_smiles = valid.groupby("rdkit_inchi_key")[
        "rdkit_canonical_isomeric_smiles"
    ].nunique()
    unique_structure = identity.drop_duplicates("smiles")
    failed = identity[identity["rdkit_status"] == "rdkit_parse_failure"]
    connectivity = identity[
        identity["source_key_comparison"] == "different_connectivity_block"
    ]
    return {
        "stage": "M1.8a_RDKit_identity_QA_NOT_curated_molecules_or_labels",
        "generated_utc": datetime.now(UTC).isoformat(),
        "rdkit_version": rdBase.rdkitVersion,
        "observations": len(identity),
        "unique_raw_smiles": int(df["smiles"].nunique()),
        "unique_reported_inchikeys": int(df["inchi_key"].nunique()),
        "unique_rdkit_canonical_isomeric_smiles": int(
            identity["rdkit_canonical_isomeric_smiles"].nunique()
        ),
        "unique_rdkit_inchikeys": int(identity["rdkit_inchi_key"].nunique()),
        "unique_structure_parse_status": counts(unique_structure["rdkit_status"]),
        "unique_structure_source_key_comparison": counts(
            unique_structure["source_key_comparison"]
        ),
        "observation_source_key_comparison": counts(identity["source_key_comparison"]),
        "unique_raw_smiles_with_multiple_fragments": int(
            (unique_structure["rdkit_fragment_count"] > 1).sum()
        ),
        "reported_inchikeys_with_multiple_isomeric_canonical_smiles": int(
            (source_key_to_smiles > 1).sum()
        ),
        "rdkit_inchikeys_with_multiple_isomeric_canonical_smiles": int(
            (rdkit_key_to_smiles > 1).sum()
        ),
        "parse_failure_record_ids": sorted(failed["reactant_set_id"].astype(str).unique().tolist())[:30],
        "different_connectivity_sample_record_ids": sorted(
            connectivity["reactant_set_id"].astype(str).unique().tolist()
        )[:30],
        "by_target_and_m17_partition": by_partition,
        "limitations": [
            "Source-reported InChIKeys may omit stereochemistry or reflect a different normalization convention.",
            "Agreement of InChIKey first blocks indicates matching InChI connectivity layer, not necessarily identical stereochemistry, protonation or assay material.",
            "Canonical isomeric SMILES are calculated from supplied SMILES only; no salt stripping, tautomer normalization, uncharging or stereochemical completion occurs.",
            "A reported-InChIKey mismatch is a QA flag, not automatic evidence of erroneous data.",
            "No observations or target-construct groups are removed, aggregated or assigned to train/test partitions.",
        ],
    }


def audit(input_path: Path, output_path: Path, summary_path: Path) -> dict:
    df = pd.read_csv(
        input_path,
        low_memory=False,
        dtype={"reactant_set_id": "string", "inchi_key": "string", "smiles": "string"},
    )
    absent = sorted(REQUIRED - set(df.columns))
    if absent:
        raise ValueError(f"Missing required columns: {absent}")
    if df["reactant_set_id"].isna().any() or df["reactant_set_id"].duplicated().any():
        raise ValueError("Observation reactant_set_id must be present and unique")

    unique_smiles = df["smiles"].drop_duplicates()
    mapping = {smiles: profile_structure(smiles) for smiles in unique_smiles.tolist()}
    keys = [asdict(mapping[smiles]) for smiles in df["smiles"].tolist()]
    profiles = pd.DataFrame(keys).rename(
        columns={
            "status": "rdkit_status",
            "canonical_isomeric_smiles": "rdkit_canonical_isomeric_smiles",
            "fragment_count": "rdkit_fragment_count",
            "heavy_atom_count": "rdkit_heavy_atom_count",
        }
    )
    outcols = [
        "reactant_set_id",
        "monomer_id",
        "target",
        "m17_partition",
        "measurement_status",
        "source",
        "smiles",
        "inchi_key",
    ]
    out = pd.concat(
        [df[[c for c in outcols if c in df.columns]].reset_index(drop=True), profiles],
        axis=1,
    )
    out["source_key_comparison"] = [
        compare_inchikey(source, computed)
        for source, computed in zip(out["inchi_key"], out["rdkit_inchi_key"], strict=True)
    ]
    if len(out) != len(df):
        raise AssertionError("M1.8a output row count changed")
    result = summarize(df, out)
    result["input_filename"] = input_path.name
    result["input_sha256"] = sha256_file(input_path)
    result["output_filename"] = output_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_path.with_name(output_path.name + ".partial")
    tmp_summary = summary_path.with_name(summary_path.name + ".partial")
    out.to_csv(tmp_csv, index=False)
    tmp_summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp_csv.replace(output_path)
    tmp_summary.replace(summary_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.input, args.output, args.summary)
    print(f"Observations retained: {result['observations']:,}")
    print(f"Unique raw SMILES: {result['unique_raw_smiles']:,}")
    print(f"RDKit status (unique SMILES): {result['unique_structure_parse_status']}")
    print(f"InChIKey comparison (unique SMILES): {result['unique_structure_source_key_comparison']}")
    print(f"Summary: {args.summary}")
    print(f"Observation-level QA: {args.output}")


if __name__ == "__main__":
    main()
