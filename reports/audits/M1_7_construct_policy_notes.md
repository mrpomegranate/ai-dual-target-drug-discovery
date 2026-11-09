# M1.7 scientific interpretation — provisional inclusion and construct shift

The dataset contains **35,210** single-chain EGFR/HER2 source observations from an October 2026 BindingDB archive. M1.7 assigns *observation-level* roles, **not curated molecule labels**.

## Core counts

- Primary exact-label candidates: **16,517 EGFR + 4,148 HER2 = 20,665**.
- Primary classifiable censored-label candidates: **4,267 EGFR + 578 HER2 = 4,845**. Exact pIC50 cannot be assigned.
- Primary measurement holds: **213 EGFR + 10 HER2 = 223**.
- Annotated-construct reserve: **8,218 EGFR + 1,259 HER2 = 9,477** observations, spanning exact, censored and metadata-limited assays.
- Annotated exact-IC50 assay candidates after preliminary eligibility checks: **6,261 EGFR + 793 HER2 = 7,054**; these need substantial curation and splitting before any challenge test.

**Data-version caveat:** The summary included here was generated from an earlier uploaded CSV in which `organism_flag` misidentified `Human` as unexpected; the raw `organism` field was used to normalize human metadata. The script reports 3,409 mismatches in that earlier snapshot. Regenerate using your local corrected M1.4 CSV; expect zero mismatches if its flags were corrected.

## Scientific caution

- The **plain target-name** group is *reference-like by naming convention*, not verified wild-type protein. Mutation or truncated-construct status may be incompletely documented.
- Name-annotated proteins are not guaranteed to be isolated single substitution mutants: brackets can encode insertions, deletions and other engineered constructs.
- A ligand-only GNN cannot condition its output on protein variant. Different labels on the same ligand for two constructs can be real biological effects or assay discrepancies, not ordinary random label noise.
- **No scaffold train/test allocation is performed in M1.7.** The 9,477 held observations are a *challenge reserve*, not yet an eligible independent test set. A scaffold-disjoint construct-shift test and a same-ligand paired contrast answer fundamentally different questions.
- Publication/patent provenance and construct are associated; source imbalance can confound comparisons of potency and apparent prediction error. Do not attribute differences to labs or infer causality from aggregate source plots.

## Two complementary future tests

**A. Scaffold-disjoint construct-shift challenge:** After structure/assay curation, compare error of models trained on provisional reference-like constructs when applied to annotated constructs with disjoint ligand scaffolds. Compare to an in-domain scaffold-disjoint reference-like benchmark and report source and chemistry differences. Any performance difference is associative, not an isolated mutation effect.

**B. Same-molecule contrast:** For an identical standardized ligand with independent measurements against reference-like and named mutant constructs, estimate observed ΔpIC50 across matched conditions where possible. This directly exposes construct-associated potency changes but is **not** a fair scaffold-held-out generalization test. The existing ligand-only models cannot produce two different predictions when all supplied inputs are identical, so a target-aware extension would be required to learn such differences.

## Outputs and source of truth

- `M1_7_construct_policy_summary.json` records input checksum, per-target partition counts, source composition, provisional overlap, and assumptions.
- `09_provisional_inclusion_decisions.png` depicts observation counts by policy role.
- `10_source_composition_after_policy.png` depicts source shifts and provides a warning against naïve domain comparisons.
- `data/interim/M1_7_observation_decisions.csv` is a row-level audit artifact, not committed and not a training table.
