"""M1.10: conservative, reversible, provisional pIC50 label proposal.

Preserves exact and censored observation evidence; never edits the M1.8c input.
No protein construct, endpoint or provenance group is independently verified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")

KEY = ["target", "m18c_structure_id"]
ACCEPT = {"provisional_single_exact", "provisional_tight_consensus"}
REQUIRED = {
    "reactant_set_id", "target", "m18c_structure_id", "m18c_status",
    "m17_partition", "ic50_nm", "pic50", "activity_at_1000_nm",
    "provenance_key", "source", "m18c_candidate_isomeric_smiles",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_upstream(input_path: Path, m18d_path: Path, m19a_path: Path) -> dict:
    actual = digest(input_path)
    summaries = {}
    for name, path in (("m18d", m18d_path), ("m19a", m19a_path)):
        obj = json.loads(path.read_text(encoding="utf-8"))
        if obj.get("input_sha256") != actual:
            raise ValueError(f"{name}: input SHA-256 mismatch; regenerate or select matched files")
        summaries[name] = obj
    return summaries


def reconcile(
    observations: pd.DataFrame,
    tight_max_pic50_range: float = 0.30,
    relaxed_max_pic50_range: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Return label candidates, sensitivity, row trace and descriptive counts.

    An 'accepted' output is only *provisional* and not sequence/assay verified.
    Censored evidence can veto a provisional exact regression label when the
    activity class contradicts the exact measurement(s).
    """
    if not (0 < tight_max_pic50_range < relaxed_max_pic50_range <= 1):
        raise ValueError("Require 0 < tight cutoff < relaxed cutoff <= 1 pIC50")
    missing = REQUIRED - set(observations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if observations.reactant_set_id.isna().any() or observations.reactant_set_id.duplicated().any():
        raise ValueError("reactant_set_id must be present and unique across observations")
    approved = observations["m18c_status"].eq("provisional_exact_structure")
    exact = observations.loc[
        approved & observations["m17_partition"].eq("primary_exact_candidate")
    ].copy()
    censored = observations.loc[
        approved & observations["m17_partition"].eq("primary_classification_candidate")
    ].copy()
    reserve = observations.loc[
        approved & observations["m17_partition"].eq("secondary_construct_reserve")
    ]
    if exact["m18c_structure_id"].isna().any():
        raise ValueError("Exact records with approved status must have a structure ID")
    exact["pic50"] = pd.to_numeric(exact["pic50"], errors="raise")
    exact["ic50_nm"] = pd.to_numeric(exact["ic50_nm"], errors="raise")
    if not np.isfinite(exact["pic50"]).all() or not (exact["ic50_nm"] > 0).all():
        raise ValueError("Exact measurements must be finite and strictly positive")
    if not np.allclose(exact["pic50"], 9 - np.log10(exact["ic50_nm"]), atol=1e-6):
        raise ValueError("pIC50 values are inconsistent with exact IC50 (nM)")
    expected_activity = np.where(exact.ic50_nm < 1000, "active", "inactive")
    if not (expected_activity == exact.activity_at_1000_nm.to_numpy()).all():
        raise ValueError("An exact activity label conflicts with the 1000 nM cutoff")
    if censored["m18c_structure_id"].isna().any():
        raise ValueError("Censored records require a structure ID to compare with exact data")
    if not censored["activity_at_1000_nm"].isin(["active", "inactive"]).all():
        raise ValueError("Censored observations here must have determinate activity")

    censored_map = {}
    for key, group in censored.groupby(KEY, sort=False):
        censored_map[key] = (set(group["activity_at_1000_nm"]), len(group))
    reserve_pairs = set(zip(reserve["target"], reserve["m18c_structure_id"], strict=True))

    rows = []
    trace_status = {}
    for (target, structure_id), group in exact.groupby(KEY, sort=True):
        n = len(group)
        minimum = float(group["pic50"].min())
        maximum = float(group["pic50"].max())
        spread = max(0.0, maximum - minimum)
        observed_class = set(group["activity_at_1000_nm"])
        exact_conflict = len(observed_class) > 1
        single_class = next(iter(observed_class)) if not exact_conflict else "conflicting"
        cset, cn = censored_map.get((target, structure_id), (set(), 0))
        censored_conflict = bool(cset - observed_class) if not exact_conflict else bool(cset)
        if exact_conflict:
            reason = "hold_exact_activity_contradiction"
        elif censored_conflict:
            reason = "hold_censored_activity_contradiction"
        elif n == 1:
            reason = "provisional_single_exact"
        elif spread <= tight_max_pic50_range + 1e-12:
            reason = "provisional_tight_consensus"
        elif spread > 1.0 + 1e-12:
            reason = "hold_major_disagreement_gt1_pic50"
        else:
            reason = "hold_moderate_disagreement"
        strict_accept = reason in ACCEPT
        relaxed_accept = (
            not exact_conflict and not censored_conflict
            and (n == 1 or spread <= relaxed_max_pic50_range + 1e-12)
        )
        no_censor_check_accept = (
            not exact_conflict and (n == 1 or spread <= tight_max_pic50_range + 1e-12)
        )
        # One study/patent group receives one vote, regardless of duplicated rows.
        # This is a descriptive robust estimator, not proof of independence.
        per_provenance = group.groupby("provenance_key")["pic50"].median().to_numpy()
        center = float(np.median(per_provenance))
        raw_median = float(group["pic50"].median())
        canonical_set = set(group["m18c_candidate_isomeric_smiles"].dropna())
        if len(canonical_set) != 1:
            raise ValueError(f"A provisional exact structure ID has inconsistent SMILES: {structure_id}")
        unique_contexts = group[["provenance_key", "pic50"]].drop_duplicates().shape[0]
        rows.append({
            "target": target,
            "m18c_structure_id": structure_id,
            "canonical_isomeric_smiles": next(iter(canonical_set)),
            "n_exact_observations": n,
            "n_provenance_keys": int(group.provenance_key.nunique()),
            "n_source_categories": int(group.source.nunique()),
            "n_potential_repeat_signatures": n - unique_contexts,
            "n_censored_classification_observations": cn,
            "censored_activity_evidence": ";".join(sorted(cset)) or "none",
            "has_censored_activity_conflict": censored_conflict,
            "has_exact_activity_conflict": exact_conflict,
            "has_annotated_construct_for_same_structure_target": (target, structure_id) in reserve_pairs,
            "min_exact_pic50": minimum,
            "max_exact_pic50": maximum,
            "exact_pic50_range": spread,
            "unadjusted_observation_median_pic50": raw_median,
            "provenance_median_of_medians_pic50": center,
            "m10_policy_status": reason,
            "m10_provisional_pic50": center if strict_accept else np.nan,
            "m10_provisional_activity": single_class if strict_accept else "",
            "m10_label_support": ("unreplicated_unverified" if n == 1 else "repeated_unverified") if strict_accept else "held_no_label",
            "m10_target_construct_verification": "NOT_sequence_or_assay_verified",
            "m10_exact_observation_ids": ";".join(group.reactant_set_id.astype(str)),
            "sensitivity_relaxed_0_50_accepted": bool(relaxed_accept),
            "sensitivity_ignore_censored_accepted": bool(no_censor_check_accept),
        })
        for reactant in group.reactant_set_id.astype(str):
            trace_status[reactant] = reason

    labels = pd.DataFrame.from_records(rows)
    sensitivity = labels[[
        "target", "m18c_structure_id", "n_exact_observations",
        "exact_pic50_range", "has_exact_activity_conflict",
        "has_censored_activity_conflict", "m10_policy_status",
        "m10_provisional_pic50", "unadjusted_observation_median_pic50",
        "provenance_median_of_medians_pic50",
        "sensitivity_relaxed_0_50_accepted", "sensitivity_ignore_censored_accepted",
    ]].copy()
    sensitivity["sensitivity_strict_0_30_accepted"] = labels.m10_policy_status.isin(ACCEPT)
    trace = observations.loc[
        approved & observations.m17_partition.isin(
            ["primary_exact_candidate", "primary_classification_candidate"]
        ), [
            "reactant_set_id", "target", "m18c_structure_id", "m17_partition",
            "provenance_key", "source", "ic50_raw", "ic50_nm", "pic50",
            "activity_at_1000_nm",
        ]
    ].copy()
    trace["m10_exact_pair_decision"] = trace.reactant_set_id.astype(str).map(trace_status)
    label_by_pair = dict(zip(
        zip(labels.target, labels.m18c_structure_id, strict=True),
        labels.m10_policy_status,
        strict=True,
    ))
    censored_mask = trace.m17_partition.eq("primary_classification_candidate")
    trace.loc[censored_mask, "m10_exact_pair_decision"] = [
        label_by_pair.get((r.target, r.m18c_structure_id), "no_exact_label_to_reconcile")
        for r in trace.loc[censored_mask, ["target", "m18c_structure_id"]].itertuples(index=False)
    ]
    by_target = {}
    for target, group in labels.groupby("target"):
        by_target[target] = {
            "pairs": len(group),
            "provisional_labels": int(group.m10_policy_status.isin(ACCEPT).sum()),
            "policy_status_counts": dict(sorted(Counter(group.m10_policy_status).items())),
            "pairs_with_censored_conflict": int(group.has_censored_activity_conflict.sum()),
        }
    status = dict(sorted(Counter(labels.m10_policy_status).items()))
    summary = {
        "input_observations": len(observations),
        "eligible_exact_observations": len(exact),
        "eligible_determinate_censored_observations": len(censored),
        "candidate_molecule_target_pairs": len(labels),
        "policy_status_counts": status,
        "provisional_regression_labels": int(labels.m10_policy_status.isin(ACCEPT).sum()),
        "held_regression_pairs": int((~labels.m10_policy_status.isin(ACCEPT)).sum()),
        "pairs_with_censored_evidence": int((labels.n_censored_classification_observations > 0).sum()),
        "pairs_with_any_censored_activity_conflict": int(labels.has_censored_activity_conflict.sum()),
        "pairs_with_exact_activity_conflict": int(labels.has_exact_activity_conflict.sum()),
        "pairs_accepted_if_censored_evidence_ignored": int(sensitivity.sensitivity_ignore_censored_accepted.sum()),
        "pairs_accepted_relaxed_cutoff_0_50": int(sensitivity.sensitivity_relaxed_0_50_accepted.sum()),
        "strict_accepted_pairs_also_seen_in_annotated_construct_reserve": int(
            (labels.m10_policy_status.isin(ACCEPT) & labels.has_annotated_construct_for_same_structure_target).sum()
        ),
        "per_target": by_target,
    }
    return labels, sensitivity, trace, summary


def plot_summary(labels: pd.DataFrame, path: Path) -> None:
    order = [
        "provisional_single_exact", "provisional_tight_consensus",
        "hold_exact_activity_contradiction", "hold_censored_activity_contradiction",
        "hold_moderate_disagreement", "hold_major_disagreement_gt1_pic50",
    ]
    pretty = ["Single exact", "Tight repeats", "Exact activity conflict", "Censored conflict", "Moderate disagreement", "Major disagreement"]
    targets = [x for x in ("EGFR", "HER2") if x in set(labels.target)]
    counts = pd.crosstab(labels.m10_policy_status, labels.target).reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(order))
    width = 0.36
    for j, target in enumerate(targets):
        heights = counts[target] if target in counts.columns else np.zeros(len(order))
        bars = ax.bar(x + (j - (len(targets) - 1) / 2) * width, heights, width, label=target)
        for b in bars:
            if b.get_height():
                ax.annotate(f"{int(b.get_height()):,}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, pretty, rotation=22, ha="right")
    ax.set_ylabel("Molecule–target pairs")
    ax.set_title("M1.10 provisional regression label decisions\nNot sequence/assay verified; no model training")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--m18d-summary", required=True, type=Path)
    p.add_argument("--m19a-summary", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--sensitivity", required=True, type=Path)
    p.add_argument("--observation-trace", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--figure", required=True, type=Path)
    args = p.parse_args()
    upstream = verify_upstream(args.input, args.m18d_summary, args.m19a_summary)
    data = pd.read_csv(args.input, low_memory=False)
    labels, sensitivity, trace, summary = reconcile(data)
    expected = upstream["m19a"]["population"]
    checks = {
        "input_observations": expected["input_observations"],
        "eligible_exact_observations": expected["primary_exact_with_provisional_structure_id"],
        "candidate_molecule_target_pairs": expected["distinct_molecule_target_pairs"],
    }
    for key, value in checks.items():
        if summary[key] != value:
            raise ValueError(f"M1.9a provenance mismatch: {key}: {summary[key]} != {value}")
    for path in (args.labels, args.sensitivity, args.observation_trace, args.summary, args.figure):
        path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.labels, index=False)
    sensitivity.to_csv(args.sensitivity, index=False)
    trace.to_csv(args.observation_trace, index=False)
    plot_summary(labels, args.figure)
    report = {
        "stage": "M1.10_provisional_exact_potency_label_reconciliation_NOT_validated_assay_labels",
        "generated_utc": datetime.now(UTC).isoformat(),
        "policy": {
            "candidate_input": "M1.7 primary exact, M1.8c approved provisional chemical structure IDs",
            "exact_repeat_consensus_max_range_pic50": 0.30,
            "relaxed_sensitivity_max_range_pic50": 0.50,
            "activity_cutoff": "strict IC50 < 1000 nM is active",
            "one_exact_observation": "provisional unreplicated label; assay not confirmed",
            "repeat_estimator": "median of provenance-key medians, not proof of independent evidence",
            "censored_conflicts": "contradictory determinate censored activity vetoes exact regression label",
            "construct_assays": "plain target names only; no sequence or endpoint confirmation",
            "ambiguous_censored_records": "not used as exact regression labels",
            "duplicate_sources": "all records retained; apparent duplicate signatures flagged",
        },
        "inputs": {
            "input_filename": args.input.name,
            "input_sha256": digest(args.input),
            "m18d_summary_sha256": digest(args.m18d_summary),
            "m19a_summary_sha256": digest(args.m19a_summary),
        },
        "analysis": summary,
        "outputs": {
            "labels": str(args.labels),
            "sensitivity": str(args.sensitivity),
            "observation_trace": str(args.observation_trace),
            "figure": str(args.figure),
        },
        "limitations": [
            "Every regression label is provisional: neither reference EGFR/HER2 sequence nor biochemical assay endpoint is verified at record level.",
            "Single-observation labels are unreplicated, not necessarily reliable ground truth.",
            "A provenance_key is a publication/patent grouping, not an independent lab or assay ID.",
            "Accepted repeats only agree within an empirical 0.30 pIC50 range; this is a policy choice, not a universally established measurement error threshold.",
            "Censored evidence is used only to flag contradictory activity; censored values are never converted to exact regression pIC50.",
            "No salts, tautomers, protein constructs or source-linked chemistry are silently merged.",
            "No train/test split, baseline model, GNN, or external validation has occurred.",
        ],
    }
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Input observations: {summary['input_observations']:,}")
    print(f"Exact records: {summary['eligible_exact_observations']:,}")
    print(f"Censored records considered as contradiction evidence: {summary['eligible_determinate_censored_observations']:,}")
    print(f"Molecule–target pairs: {summary['candidate_molecule_target_pairs']:,}")
    print(f"Provisional regression labels: {summary['provisional_regression_labels']:,}")
    print(f"Held pairs: {summary['held_regression_pairs']:,}")
    print(f"Pairs with censored activity conflict: {summary['pairs_with_any_censored_activity_conflict']:,}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
