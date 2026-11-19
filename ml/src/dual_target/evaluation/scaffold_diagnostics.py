"""Fit-only out-of-fold scaffold difficulty diagnostics; no calibration fitting.

Scaffold-level MAE can be very noisy for small groups. All scaffold groups
are retained; only >= ``minimum_n`` are eligible for the main ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("egfr", "her2")
MODEL_NAMES = ("rf", "gine", "gatv2", "rf75_gine25")
REQUIRED = {
    "structure_id", "scaffold_group_id", "fold",
    "egfr_mask", "her2_mask", "egfr_pic50", "her2_pic50",
} | {f"{t}_{m}_pred" for t in TARGETS for m in MODEL_NAMES}


def validate_fit_only_oof(frame: pd.DataFrame, *, expected_n: int | None = None) -> None:
    """Fail before aggregation if target masks, folds, or model predictions fail."""
    if missing := REQUIRED - set(frame.columns):
        raise ValueError(f"OOF missing required columns: {sorted(missing)}")
    if frame.empty or (expected_n is not None and len(frame) != expected_n):
        raise ValueError("OOF molecule count does not match expected fit-only population")
    if frame.structure_id.isna().any() or frame.scaffold_group_id.isna().any():
        raise ValueError("Missing molecular or scaffold identity")
    if frame.structure_id.duplicated().any():
        raise ValueError("Duplicate OOF structure_id")
    if not frame.fold.isin(range(5)).all():
        raise ValueError("Expected scaffold-CV fold IDs 0 to 4")
    if frame.groupby("scaffold_group_id").fold.nunique().max() != 1:
        raise ValueError("Scaffold has OOF molecules in multiple folds")
    for t in TARGETS:
        masks = frame[f"{t}_mask"]
        if not masks.isin([True, False, 0, 1]).all():
            raise ValueError(f"Non-boolean {t} mask")
        active = masks.astype(bool).to_numpy()
        labels = pd.to_numeric(frame[f"{t}_pic50"], errors="coerce").to_numpy()
        if not np.isfinite(labels[active]).all() or not np.isnan(labels[~active]).all():
            raise ValueError(f"{t} observed labels disagree with accepted masks")
        for model in MODEL_NAMES:
            estimated = pd.to_numeric(frame[f"{t}_{model}_pred"], errors="coerce").to_numpy()
            if not np.isfinite(estimated[active]).all():
                raise ValueError(f"Nonfinite {t} {model} prediction on measured label")
        expected_ensemble = (
            .75 * frame[f"{t}_rf_pred"].to_numpy(dtype=float)
            + .25 * frame[f"{t}_gine_pred"].to_numpy(dtype=float)
        )
        actual_ensemble = frame[f"{t}_rf75_gine25_pred"].to_numpy(dtype=float)
        if not np.allclose(
            actual_ensemble[active], expected_ensemble[active], rtol=0, atol=1e-6
        ):
            raise ValueError("75/25 ensemble does not match frozen formula")
    if not (frame.egfr_mask.astype(bool) | frame.her2_mask.astype(bool)).all():
        raise ValueError("OOF includes molecule with neither accepted potency label")


def scaffold_error_table(
    frame: pd.DataFrame, *, minimum_n: int = 10,
) -> pd.DataFrame:
    """Group errors by the ORIGINAL scaffold ID and target-specific masks."""
    if isinstance(minimum_n, bool) or not isinstance(minimum_n, int) or minimum_n < 1:
        raise ValueError("minimum_n must be a positive integer")
    validate_fit_only_oof(frame)
    all_rows: list[dict] = []
    for target in TARGETS:
        measured = frame.loc[frame[f"{target}_mask"].astype(bool)].copy()
        label_col = f"{target}_pic50"
        for model in MODEL_NAMES:
            measured[f"{model}_error"] = measured[f"{target}_{model}_pred"] - measured[label_col]
            measured[f"{model}_abs_error"] = measured[f"{model}_error"].abs()
        for scaffold_id, group in measured.groupby("scaffold_group_id", sort=True):
            n = len(group)
            row = {
                "target": target,
                "scaffold_group_id": scaffold_id,
                "fold": int(group.fold.iloc[0]),
                "n": n,
                "meets_minimum_n": n >= minimum_n,
                "observed_mean_pic50": float(group[label_col].mean()),
                "observed_min_pic50": float(group[label_col].min()),
                "observed_max_pic50": float(group[label_col].max()),
            }
            for model in MODEL_NAMES:
                row[f"{model}_mae"] = float(group[f"{model}_abs_error"].mean())
                row[f"{model}_median_abs_error"] = float(group[f"{model}_abs_error"].median())
                row[f"{model}_mean_signed_error"] = float(group[f"{model}_error"].mean())
            row["gine_minus_rf_mae"] = row["gine_mae"] - row["rf_mae"]
            row["gatv2_minus_rf_mae"] = row["gatv2_mae"] - row["rf_mae"]
            row["ensemble_minus_rf_mae"] = row["rf75_gine25_mae"] - row["rf_mae"]
            all_rows.append(row)
    return pd.DataFrame(all_rows).sort_values(
        ["target", "scaffold_group_id"]
    ).reset_index(drop=True)


def rank_difficult_scaffolds(
    groups: pd.DataFrame, *, top_k: int = 12, rank_by: str = "rf_mae"
) -> pd.DataFrame:
    """Top RF errors separately per target, with explicit support threshold."""
    if rank_by not in {f"{name}_mae" for name in MODEL_NAMES}:
        raise ValueError("rank_by must be one of the supported model MAEs")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be positive")
    eligible = groups.loc[groups.meets_minimum_n].copy()
    if eligible.empty:
        raise ValueError("No scaffold has enough target labels for a stable ranking")
    ranked: list[pd.DataFrame] = []
    for target in TARGETS:
        subset = eligible.loc[eligible.target == target]
        if subset.empty:
            raise ValueError(f"No reportable {target.upper()} scaffolds")
        ranked.append(
            subset.sort_values([rank_by, "n", "scaffold_group_id"],
                               ascending=[False, False, True]).head(top_k)
        )
    return pd.concat(ranked, ignore_index=True)


def summarize_scaffold_diagnostics(frame: pd.DataFrame, groups: pd.DataFrame) -> dict:
    summary = {
        "purpose": "fit_only_oof_scaffold_diagnostics_no_calibration_labels",
        "molecules": len(frame),
        "scaffold_groups": int(frame.scaffold_group_id.nunique()),
        "calibration_scored": False,
        "threshold_scored": False,
        "development_validation_scored": False,
        "locked_test_scored": False,
        "ranking_is_exploratory": True,
        "targets": {},
    }
    for target in TARGETS:
        subset = frame.loc[frame[f"{target}_mask"].astype(bool)]
        groups_target = groups.loc[groups.target == target]
        summary["targets"][target] = {
            "n_labeled": len(subset),
            "scaffold_groups_with_labels": len(groups_target),
            "reportable_scaffolds": int(groups_target.meets_minimum_n.sum()),
            "reportable_molecules": int(groups_target.loc[groups_target.meets_minimum_n, "n"].sum()),
            "minimum_scaffold_n": int(groups_target.loc[groups_target.meets_minimum_n, "n"].min()) if groups_target.meets_minimum_n.any() else None,
            "pooled_mae": {
                model: float(np.abs(subset[f"{target}_{model}_pred"] - subset[f"{target}_pic50"]).mean())
                for model in MODEL_NAMES
            },
        }
    return summary
