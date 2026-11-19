"""GATv2 CV driver metadata, alignment, and checkpoint tests (synthetic only)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from dual_target.data.modeling_dataset import MoleculeRecord
from dual_target.evaluation.gnn_cv_metrics import score_oof_rows
from dual_target.features.molecular_graph import GRAPH_SCHEMA_VERSION, molecule_to_graph
from dual_target.models.gat import MolecularGATv2
from dual_target.training.gnn_training import GNNTrainingConfig

from scripts.run_gatv2_5fold_reference import (
    epoch_rows,
    output_paths,
    predict_oof,
    read_reference_fold_membership,
    save_checkpoint,
    verify_reference_fold_membership,
    write_csv,
)


def _folds():
    return SimpleNamespace(
        folds=tuple(SimpleNamespace(
            number=i,
            holdout=(SimpleNamespace(structure_id=f"mol_{i}"),),
        ) for i in range(5))
    )


def test_reference_fold_membership_matches_and_rejects_mismatch(tmp_path):
    file = tmp_path / "gine_oof.csv"
    write_csv(file, [{"structure_id": f"mol_{i}", "fold": i} for i in range(5)])
    mapping = read_reference_fold_membership(file)
    verify_reference_fold_membership(_folds(), mapping)
    with pytest.raises(ValueError, match="membership differs"):
        verify_reference_fold_membership(_folds(), {**mapping, "mol_2": 3})


def test_repeated_reference_ids_rejected(tmp_path):
    file = tmp_path / "gine_oof.csv"
    write_csv(file, [{"structure_id": "same", "fold": 0},
                     {"structure_id": "same", "fold": 1}])
    with pytest.raises(ValueError, match="repeated"):
        read_reference_fold_membership(file)


def test_missing_reference_columns_rejected(tmp_path):
    file = tmp_path / "bad.csv"
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["structure_id", "not_fold"])
        writer.writerow(["mol_0", 0])
    with pytest.raises(ValueError, match="lacks"):
        read_reference_fold_membership(file)


def test_gat_output_paths_are_separate_from_gine(tmp_path):
    paths = output_paths(tmp_path / "reports", tmp_path / "models")
    assert len(paths) == 13
    assert len(set(paths.values())) == 13
    assert all("GATv2" in path.name for path in paths.values())
    assert paths["oof"].name.endswith("oof_predictions.csv")


@dataclass
class _Metrics:
    egfr_mae: float
    her2_mae: float

    @property
    def balanced_mae(self) -> float:
        return (self.egfr_mae + self.her2_mae) / 2


def test_epoch_rows_preserve_training_and_stopping_metrics():
    rows = epoch_rows([
        SimpleNamespace(epoch=1, training_batch_loss=0.7,
                        stopping_metrics=_Metrics(0.5, 0.9))
    ])
    assert rows == [{
        "epoch": 1,
        "inner_training_batch_loss": 0.7,
        "inner_stop_egfr_mae": 0.5,
        "inner_stop_her2_mae": 0.9,
        "inner_stop_balanced_mae": pytest.approx(0.7),
    }]


def test_checkpoint_roundtrip_with_architecture_metadata(tmp_path):
    config = GNNTrainingConfig(architecture="gatv2", hidden_dim=16,
                               num_layers=1, attention_heads=4)
    model = MolecularGATv2(hidden_dim=16, num_layers=1, heads=4)
    bank = SimpleNamespace(dataset_sha256="dataset", manifest_sha256="manifest",
                           stage_c_assignment_sha256="assignment")
    path = tmp_path / "gat.pt"
    save_checkpoint(model, fold=0, epochs=3, config=config, bank=bank, path=path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    assert restored["model_config"]["heads"] == 4
    assert restored["training_config"]["architecture"] == "gatv2"
    assert restored["graph_schema_version"] == GRAPH_SCHEMA_VERSION
    assert restored["refit_epoch_count"] == 3
    assert path.is_file()


def test_predict_oof_preserves_masks_and_two_target_shape():
    records = (
        MoleculeRecord(
            structure_id="one", smiles="CCO", scaffold_group_id="s1",
            egfr_mask=1, her2_mask=0, egfr_pic50=7.2, her2_pic50=None,
        ),
        MoleculeRecord(
            structure_id="two", smiles="CCN", scaffold_group_id="s2",
            egfr_mask=1, her2_mask=1, egfr_pic50=6.5, her2_pic50=7.5,
        ),
    )
    graphs = tuple(molecule_to_graph(record) for record in records)
    model = MolecularGATv2(hidden_dim=16, num_layers=1, heads=4).eval()
    rows = predict_oof(model, graphs, fold=2, batch_size=2, device="cpu")
    assert [row["structure_id"] for row in rows] == ["one", "two"]
    assert all(row["fold"] == 2 for row in rows)
    assert rows[0]["her2_mask"] is False
    assert rows[0]["her2_pic50"] is None
    assert rows[1]["her2_mask"] is True
    assert rows[1]["her2_pic50"] == pytest.approx(7.5)
    metrics = score_oof_rows(rows)
    assert metrics["egfr_n"] == 2
    assert metrics["her2_n"] == 1
    assert metrics["paired_n"] == 1
