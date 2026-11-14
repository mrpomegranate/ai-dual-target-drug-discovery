
"""Morgan fingerprint features for the EGFR/HER2 benchmark."""

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def make_morgan_fingerprints(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    include_chirality: bool = True,
) -> np.ndarray:
    """Convert SMILES into a binary fingerprint matrix.

    Each input molecule produces one row with n_bits features.
    Input order is preserved. Invalid SMILES raise an error.
    """
    if radius < 0:
        raise ValueError("Fingerprint radius must be nonnegative.")
    if n_bits <= 0:
        raise ValueError("Fingerprint size must be positive.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=include_chirality,
    )

    fingerprints = []

    for index, smiles in enumerate(smiles_list):
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Missing SMILES at row {index}.")

        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            raise ValueError(f"Invalid SMILES at row {index}.")

        fingerprint = generator.GetFingerprintAsNumPy(molecule)
        fingerprints.append(fingerprint.astype(np.uint8, copy=False))

    if not fingerprints:
        return np.empty((0, n_bits), dtype=np.uint8)

    return np.vstack(fingerprints)
