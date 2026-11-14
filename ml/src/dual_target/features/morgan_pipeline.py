
"""Cache Morgan fingerprints once and safely select rows by molecule identity.

This module is independent of Random Forest, Optuna, and MLflow. It never
performs label-based feature fitting or accesses the frozen test partition.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from ..data.modeling_dataset import MoleculeRecord
from ..morgan_features import make_morgan_fingerprints


@dataclass(frozen=True)
class MorganFeatureBank:
    """In-memory fingerprint matrix with verified molecule-to-row mapping."""

    molecule_ids: tuple[str, ...]
    smiles: tuple[str, ...]
    scaffold_ids: tuple[str, ...]
    features: np.ndarray
    radius: int
    n_bits: int
    include_chirality: bool
    fingerprint_sha256: str
    _index: Mapping[str, int] = field(repr=False, compare=False)

    def select_features(self, records: Sequence[MoleculeRecord]) -> np.ndarray:
        """Return fingerprint rows in exactly the requested record order.

        The returned matrix is a copy, so downstream estimators cannot change
        the cached features. Missing, duplicated, or mismatched IDs fail fast.
        """
        positions: list[int] = []
        seen: set[str] = set()

        for record in records:
            structure_id = record.structure_id
            if structure_id in seen:
                raise ValueError(f"Duplicate requested molecule ID: {structure_id}")
            seen.add(structure_id)

            position = self._index.get(structure_id)
            if position is None:
                raise ValueError(f"Molecule is not in fingerprint bank: {structure_id}")
            if record.smiles != self.smiles[position]:
                raise ValueError(f"SMILES mismatch for molecule: {structure_id}")
            if record.scaffold_group_id != self.scaffold_ids[position]:
                raise ValueError(f"Scaffold mismatch for molecule: {structure_id}")
            positions.append(position)

        return self.features[positions].copy()


def build_morgan_feature_bank(
    records: Sequence[MoleculeRecord],
    *,
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> MorganFeatureBank:
    """Fingerprint a fixed record set once for reuse across folds and trials.

    Call with pipeline.data.train for internal CV; include validation only
    when evaluating a model after training. Do not supply test records.
    """
    records = tuple(records)
    if not records:
        raise ValueError("Cannot build a fingerprint bank with no molecules")
    if radius < 0 or n_bits <= 0:
        raise ValueError("Invalid Morgan radius or fingerprint bit count")

    molecule_ids = tuple(record.structure_id for record in records)
    smiles = tuple(record.smiles for record in records)
    scaffold_ids = tuple(record.scaffold_group_id for record in records)

    if any(not structure_id for structure_id in molecule_ids):
        raise ValueError("Missing molecule ID")
    if len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("Duplicate molecule IDs in fingerprint bank")
    if any(not scaffold_id for scaffold_id in scaffold_ids):
        raise ValueError("Missing scaffold ID")

    matrix = make_morgan_fingerprints(
        smiles,
        radius=radius,
        n_bits=n_bits,
        include_chirality=include_chirality,
    )
    matrix.setflags(write=False)

    provenance = {
        "radius": radius,
        "n_bits": n_bits,
        "include_chirality": include_chirality,
        "molecule_ids": molecule_ids,
        "smiles": smiles,
        "scaffold_ids": scaffold_ids,
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(matrix.tobytes(order="C"))

    return MorganFeatureBank(
        molecule_ids=molecule_ids,
        smiles=smiles,
        scaffold_ids=scaffold_ids,
        features=matrix,
        radius=radius,
        n_bits=n_bits,
        include_chirality=include_chirality,
        fingerprint_sha256=digest.hexdigest(),
        _index=MappingProxyType(
            {structure_id: index for index, structure_id in enumerate(molecule_ids)}
        ),
    )
