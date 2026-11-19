"""Deterministic, label-safe RDKit -> PyG graphs for the frozen M2 dataset.

Each feature column is an integer category, not a continuous measurement.
The future GNN must embed these categories before message passing.
"""

from __future__ import annotations

import math

import torch
from rdkit import Chem
from torch_geometric.data import Data

from ..data.modeling_dataset import MoleculeRecord

GRAPH_SCHEMA_VERSION = "m3_molecular_graph_v1"

ATOM_FEATURE_NAMES = (
    "atomic_number",
    "degree",
    "formal_charge",
    "total_hydrogens",
    "hybridization",
    "aromatic",
    "in_ring",
    "chiral_tag",
    "cip_code",
)
BOND_FEATURE_NAMES = (
    "bond_type",
    "conjugated",
    "in_ring",
    "stereo",
)

# Last value is a distinct fallback bucket for uncommon, valid chemistry.
_DEGREES = (*range(9), "OTHER")
_CHARGES = (-3, -2, -1, 0, 1, 2, 3, "OTHER")
_HYDROGENS = (*range(7), "OTHER")
_HYBRIDIZATIONS = (
    "UNSPECIFIED", "S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"
)
_CHIRAL_TAGS = (
    "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
    "CHI_OTHER", "OTHER",
)
_CIP_CODES = ("NONE", "R", "S", "OTHER")
_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "OTHER")
_BOND_STEREO = (
    "STEREONONE", "STEREOANY", "STEREOE", "STEREOZ",
    "STEREOCIS", "STEREOTRANS", "OTHER",
)

ATOM_CARDINALITIES = (
    119, len(_DEGREES), len(_CHARGES), len(_HYDROGENS),
    len(_HYBRIDIZATIONS), 2, 2, len(_CHIRAL_TAGS), len(_CIP_CODES),
)
BOND_CARDINALITIES = (
    len(_BOND_TYPES), 2, 2, len(_BOND_STEREO),
)


def _category(value: int | str, choices: tuple) -> int:
    """Return a fixed-vocabulary category; final index is OTHER."""
    try:
        return choices.index(value)
    except ValueError:
        return len(choices) - 1


def _atom_features(atom: Chem.Atom) -> list[int]:
    atomic_number = atom.GetAtomicNum()
    if not 0 <= atomic_number <= 118:
        raise ValueError(f"Unsupported atomic number: {atomic_number}")
    cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "NONE"
    return [
        atomic_number,
        _category(atom.GetDegree(), _DEGREES),
        _category(atom.GetFormalCharge(), _CHARGES),
        _category(atom.GetTotalNumHs(includeNeighbors=True), _HYDROGENS),
        _category(str(atom.GetHybridization()), _HYBRIDIZATIONS),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
        _category(str(atom.GetChiralTag()), _CHIRAL_TAGS),
        _category(cip, _CIP_CODES),
    ]


def _bond_features(bond: Chem.Bond) -> list[int]:
    return [
        _category(str(bond.GetBondType()), _BOND_TYPES),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        _category(str(bond.GetStereo()), _BOND_STEREO),
    ]


def _target_tensors(record: MoleculeRecord) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    masks = []
    for target in ("egfr", "her2"):
        mask = getattr(record, f"{target}_mask")
        label = getattr(record, f"{target}_pic50")
        if mask not in (0, 1):
            raise ValueError(f"{record.structure_id}: invalid {target} mask {mask!r}")
        if mask == 0 and label is not None:
            raise ValueError(f"{record.structure_id}: masked {target} has a label")
        if mask == 1 and (label is None or not math.isfinite(label)):
            raise ValueError(f"{record.structure_id}: measured {target} label invalid")
        values.append(float(label) if mask else 0.0)
        masks.append(bool(mask))
    if not any(masks):
        raise ValueError(f"{record.structure_id}: no accepted target labels")
    return (
        torch.tensor([values], dtype=torch.float32),
        torch.tensor([masks], dtype=torch.bool),
    )


def molecule_to_graph(record: MoleculeRecord) -> Data:
    """Build one molecular graph; never infer missing labels or alter identities.

    x: [atoms, 9] categorical atom features
    edge_index: [2, directed bonds]
    edge_attr: [directed bonds, 4] categorical bond features
    y, y_mask: [1, 2] so PyG batches them into [batch_size, 2]
    """
    if not record.structure_id or not record.scaffold_group_id:
        raise ValueError("Molecule identity and scaffold group must be nonempty")
    if not isinstance(record.smiles, str) or not record.smiles.strip():
        raise ValueError(f"{record.structure_id}: SMILES is empty")

    y, y_mask = _target_tensors(record)
    molecule = Chem.MolFromSmiles(record.smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"{record.structure_id}: cannot parse SMILES {record.smiles!r}")
    # Populate stereochemical tags, including CIP assignments where available.
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)

    x = torch.tensor(
        [_atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.long
    )
    ends: list[list[int]] = []
    attributes: list[list[int]] = []
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_features = _bond_features(bond)
        ends.extend(([i, j], [j, i]))
        attributes.extend((bond_features, bond_features))

    edge_index = (
        torch.tensor(ends, dtype=torch.long).t().contiguous()
        if ends else torch.empty((2, 0), dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(attributes, dtype=torch.long)
        if attributes else torch.empty((0, len(BOND_FEATURE_NAMES)), dtype=torch.long)
    )
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        y_mask=y_mask,
        num_nodes=molecule.GetNumAtoms(),
        structure_id=record.structure_id,
        scaffold_group_id=record.scaffold_group_id,
        smiles=record.smiles,
        graph_schema_version=GRAPH_SCHEMA_VERSION,
    )
