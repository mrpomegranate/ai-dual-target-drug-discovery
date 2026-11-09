# M1.10 — Interpretation of provisional label decisions

M1.10 proposes **12,221 provisional pIC50 molecule–target labels** from the M1.8c/M1.9a evidence, while withholding **2,002 molecule–target pairs** for conflicting exact or censored evidence or excessive repeated-measurement dispersion.

## What changed relative to an exact-only policy?

The existing M1.9a analysis identified **354** pairs whose *exact* IC50 values imply opposite active/inactive labels at the strict 1000 nM cutoff. M1.10 also compares the exact evidence against 4,844 determinate censored observations (which cannot be converted into exact pIC50). This adds **217 additional vetoed pairs** that would have passed the strict exact-only rule. The 285 pairs exclusively classified as `hold_censored_activity_contradiction` include these and other pairs that would have failed the exact repeat-dispersion criterion.

## Key caution

A `provisional_single_exact` row is **one unreplicated experimental observation**, not a high-confidence confirmed target response. The repeated-agreement rule of 0.30 pIC50 is a transparent heuristic. Treat `m10_provisional_pic50` as a candidate label to be further audited, **not final biochemical WT EGFR/HER2 ground truth**.

A publication/patent provenance group is not equivalent to an assay. Source-associated effects cannot be attributed causally to laboratories on these data alone.

## Downstream

M1.11 must preserve missing target masks, ensure the annotated-construct challenge set remains separate, and implement scaffold-/identity-disjoint splitting. The estimated accepted-label counts do **not** equal the number of usable paired dual-target molecules.
