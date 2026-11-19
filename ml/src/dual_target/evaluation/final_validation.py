"""Read-only, molecule-aligned development-validation scoring for frozen models.

No model selection, training, calibration fitting or test loading takes place here.
Scaffold and potency subgroup summaries are descriptive, not coverage guarantees.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

TARGETS = ("egfr", "her2")
MODELS = ("rf", "gine", "rf75_gine25")
BINS = (("<6", -math.inf, 6.0), ("6–<7", 6.0, 7.0),
        ("7–<8", 7.0, 8.0), ("≥8", 8.0, math.inf))


def validate_prediction_table(frame: pd.DataFrame, *, expected_n: int = 1087) -> None:
    """Check exact coverage, masks, labels, predictions and frozen 75/25 formula."""
    needed = {"structure_id", "scaffold_group_id", "egfr_mask", "her2_mask",
              "egfr_pic50", "her2_pic50"}
    needed |= {f"{t}_{m}_pred" for t in TARGETS for m in MODELS}
    if missing := needed - set(frame.columns):
        raise ValueError(f"Missing validation columns: {sorted(missing)}")
    if len(frame) != expected_n or frame.structure_id.duplicated().any():
        raise ValueError("Development-validation identity count or uniqueness mismatch")
    if frame.structure_id.isna().any() or frame.scaffold_group_id.isna().any():
        raise ValueError("Missing molecule or scaffold identity")
    for target in TARGETS:
        mask = frame[f"{target}_mask"]
        if not mask.isin([0, 1, True, False]).all():
            raise ValueError(f"Invalid {target} mask")
        active = mask.astype(bool).to_numpy()
        labels = frame[f"{target}_pic50"].to_numpy(dtype=float)
        if not np.isfinite(labels[active]).all() or not np.isnan(labels[~active]).all():
            raise ValueError(f"Measured {target} labels violate masks")
        for model in MODELS:
            values = frame[f"{target}_{model}_pred"].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {target} {model} predictions")
        blend = (.75 * frame[f"{target}_rf_pred"].to_numpy(dtype=float)
                 + .25 * frame[f"{target}_gine_pred"].to_numpy(dtype=float))
        if not np.allclose(blend, frame[f"{target}_rf75_gine25_pred"],
                           atol=1e-6, rtol=0):
            raise ValueError("Ensemble differs from frozen 75/25 rule")
    if not (frame.egfr_mask.astype(bool) | frame.her2_mask.astype(bool)).all():
        raise ValueError("Validation row has no accepted target label")


def score_target(
    frame: pd.DataFrame, *, target: str, model: str, quantile: float,
) -> tuple[dict, pd.DataFrame]:
    """Point MAE/rank, fixed interval coverage and per-molecule error details."""
    if target not in TARGETS or model not in MODELS:
        raise ValueError("Unsupported target or model")
    if not math.isfinite(quantile) or quantile < 0:
        raise ValueError("Invalid frozen conformal quantile")
    sub = frame.loc[frame[f"{target}_mask"].astype(bool), [
        "structure_id", "scaffold_group_id", f"{target}_pic50", f"{target}_{model}_pred",
    ]].copy()
    if sub.empty:
        raise ValueError("No labeled validation observations")
    observed = sub[f"{target}_pic50"].to_numpy(dtype=float)
    predicted = sub[f"{target}_{model}_pred"].to_numpy(dtype=float)
    residual = predicted - observed
    sub["target"] = target
    sub["model"] = model
    sub["observed"] = observed
    sub["predicted"] = predicted
    sub["signed_error"] = residual
    sub["abs_error"] = np.abs(residual)
    sub["lower"] = predicted - quantile
    sub["upper"] = predicted + quantile
    sub["covered"] = sub.abs_error.to_numpy() <= quantile
    rho = None
    if len(observed) >= 2 and np.ptp(observed) > 0 and np.ptp(predicted) > 0:
        value = float(spearmanr(observed, predicted).statistic)
        if math.isfinite(value):
            rho = value
    return {
        "target": target, "model": model,
        "n_labeled": len(sub),
        "n_scaffolds": int(sub.scaffold_group_id.nunique()),
        "mae": float(sub.abs_error.mean()),
        "spearman": rho,
        "nominal_coverage": 0.90,
        "empirical_coverage": float(sub.covered.mean()),
        "interval_half_width_pic50": float(quantile),
        "mean_interval_width_pic50": float(2 * quantile),
    }, sub


def score_scaffolds(details: pd.DataFrame, *, minimum_n: int = 10) -> pd.DataFrame:
    """Retain small scaffolds, but flag them as not eligible for stable ranking."""
    if isinstance(minimum_n, bool) or minimum_n < 1:
        raise ValueError("minimum_n must be positive")
    grouped = details.groupby(["target", "model", "scaffold_group_id"], sort=True)
    rows = []
    for (target, model, scaffold), group in grouped:
        rows.append({
            "target": target, "model": model, "scaffold_group_id": scaffold,
            "n": len(group), "meets_minimum_n": len(group) >= minimum_n,
            "observed_mean_pic50": float(group.observed.mean()),
            "mae": float(group.abs_error.mean()),
            "mean_signed_error": float(group.signed_error.mean()),
            "coverage90": float(group.covered.mean()),
            "mean_interval_width_pic50": float((group.upper - group.lower).mean()),
        })
    return pd.DataFrame(rows)


def score_potency_bins(details: pd.DataFrame) -> pd.DataFrame:
    """Evaluate conditional error and coverage descriptively by measured potency."""
    rows = []
    for (target, model), group in details.groupby(["target", "model"], sort=True):
        for label, low, high in BINS:
            part = group.loc[(group.observed >= low) & (group.observed < high)]
            rows.append({
                "target": target, "model": model, "observed_pic50_bin": label,
                "n": len(part),
                "mae": float(part.abs_error.mean()) if len(part) else None,
                "coverage90": float(part.covered.mean()) if len(part) else None,
                "mean_signed_error": float(part.signed_error.mean()) if len(part) else None,
            })
    return pd.DataFrame(rows)


def score_dual_activity(frame: pd.DataFrame) -> dict[str, dict]:
    """AP on measured paired labels using exactly the same regression ranking."""
    paired = frame.loc[frame.egfr_mask.astype(bool) & frame.her2_mask.astype(bool)]
    if paired.empty:
        raise ValueError("No paired EGFR/HER2 validation labels")
    labels = ((paired.egfr_pic50 > 6) & (paired.her2_pic50 > 6)).astype(int)
    if labels.nunique() < 2:
        raise ValueError("Paired validation lacks one dual-activity class")
    return {
        model: {
            "paired_n": len(paired),
            "dual_active_n": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "regression_min_pic50_ap": float(average_precision_score(
                labels, np.minimum(paired[f"egfr_{model}_pred"].to_numpy(),
                                   paired[f"her2_{model}_pred"].to_numpy()),
            )),
        }
        for model in MODELS
    }


def make_detail_rows(details: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine all labeled target/model predictions, without imputing missing labels."""
    combined = pd.concat(list(details), ignore_index=True)
    if combined.empty:
        raise ValueError("No validation predictions")
    return combined[["structure_id", "scaffold_group_id", "target", "model",
                     "observed", "predicted", "signed_error", "abs_error",
                     "lower", "upper", "covered"]]
