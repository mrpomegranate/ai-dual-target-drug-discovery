"""M0 infrastructure smoke tests (not scientific-validation tests)."""

import sys

from dual_target import __version__
from dual_target.cli import doctor
from rdkit import Chem


def test_python_minor_version() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_rdkit_parses_simple_smiles() -> None:
    molecule = Chem.MolFromSmiles("CCO")
    assert molecule is not None
    assert molecule.GetNumAtoms() == 3


def test_doctor_reports_dependencies() -> None:
    info = doctor()
    assert info["status"] == "ok"
    assert info["package_version"] == __version__
    for package in ("numpy", "pandas", "pyarrow", "rdkit"):
        assert package in info
