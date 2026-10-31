# M1.5 — Source and Construct Confounding: Descriptive Data Audit

**Project:** AI Dual-Target Drug Discovery — EGFR/HER2
**Data stage:** M1.4 observation-level extract from `BindingDB_All_202610_tsv.zip`
**Important:** This is a descriptive audit **before** molecular canonicalization, assay reconciliation, and train/test splitting. No model has been evaluated.

## Purpose

Identify source-dependent imbalances that may confound EGFR/HER2 inhibitory-potency prediction. The `source` column contains **BindingDB curation-source categories**, such as ChEMBL and US Patent; it does **not** establish the individual laboratory responsible for each measurement. Publications, patents, institutions, and experimental protocols need further linkage to assess laboratory or study effects.

## Findings from 35,210 source observations

- The extract contains **29,215 EGFR** and **5,995 HER2** observations (35,210 total). These are *records*, not independent compounds.
- Source composition is highly unequal: EGFR **14,274 US Patent** and **13,316 ChEMBL** records; HER2 **3,364 US Patent** and **2,479 ChEMBL** records, with smaller literature-curation groups.
- Target-name construct annotations are unevenly distributed. **7,722/14,274 (54.1%) of EGFR US Patent records** have bracket annotations, versus **0/13,316 EGFR ChEMBL** records. For HER2 US Patent: **1,257/3,364 (37.4%)**; HER2 ChEMBL: **2/2,479 (~0.1%)**. Bracket annotation is an imperfect construct flag; lack of annotation does not establish wild-type protein.
- Among **exact numerical IC50** values, unadjusted median EGFR pIC50 is **7.84** for US Patent versus **7.04** for ChEMBL. For HER2, medians are **7.39** versus **6.89**. These differences may reflect different molecules, construct variants, study design, selection, or repeated records. They do **not** demonstrate source-related measurement bias by themselves.
- The population of `Article DOI`/`PMID`/`Patent number` differs systematically across source categories. The extracted data has **limited pH and temperature coverage**, so those fields cannot reliably correct for all assay conditions.
- Multiple observations share reported InChIKeys. They must not automatically be treated as independent molecules or as duplicate errors; some may represent valid repeat assays, alternate contexts, or overlapping publications.

## Figure index

| PNG under `reports/figures/source_bias/` | Interpretation |
|---|---|
| `01_observations_by_source.png` | Absolute source-category imbalance by target |
| `02_construct_annotations_by_source.png` | Source-associated difference in target-name construct annotations |
| `03_exact_pic50_by_source_egfr.png` | EGFR exact pIC50 distribution by source (not adjusted) |
| `04_exact_pic50_by_source_her2.png` | HER2 exact pIC50 distribution by source (not adjusted) |
| `05_activity_by_source.png` | Provisional activity evidence at strictly IC50 < 1000 nM; includes classified censored bounds |
| `06_metadata_completeness.png` | Provenance and assay-metadata availability |
| `07_observations_vs_distinct_inchikeys.png` | Observation count versus distinct source InChIKeys, **not** RDKit standardized molecules |
| `00_visual_index.png` | At-a-glance contact sheet for the seven analytical figures |

An additional local gallery is `reports/figures/source_bias/index.html`.

## Reproduce plots in the capstone repository

1. Copy `scripts/plot_source_bias.py` into the repository, and copy the `reports/` directory as supplied in the ZIP.
2. Add plotting dependencies once from PowerShell in the project root:

   ```powershell
   uv add --dev matplotlib
   ```

   This updates `pyproject.toml` and `uv.lock`. Matplotlib is a development/reporting dependency; the baseline scientific pipeline does not require it for inference.
3. Re-run the plots against **your current locally corrected M1.4 CSV**:

   ```powershell
   uv run --locked python .\scripts\plot_source_bias.py `
       --input .\data\interim\bindingdb_egfr_her2_observations.csv `
       --outdir .\reports\figures\source_bias `
       --summary .\reports\audits\M1_5_source_bias_profile.json
   ```

4. Inspect `reports/figures/source_bias/index.html` and the generated JSON profile. The script records the input CSV SHA-256 so runs with slightly different source CSVs remain identifiable.

The figures included in this bundle were generated from the M1.4 CSV supplied in the conversation **before the corrected organism-flag rerun**. They do not rely on that flag; nevertheless regenerate them locally using the final corrected CSV for a consistent evidence trail. No raw experimental observation CSV is included in the bundle.

## Required next scientific checks

1. **Construct-aware dataset:** classify EGFR and HER2 wild-type, point mutants, insertions, deletions, and ambiguous constructs using target name *and available protein-sequence evidence*; choose a defensible primary analysis cohort.
2. **Study/source identifier audit:** recover institution when possible; use Article DOI/PMID, patent numbers, and study/assay identifiers to identify repeated and overlapping evidence. `BindingDB Entry DOI` is generally an entry/source identifier and not a substitute for a particular lab or a standardized assay protocol.
3. **Comparable molecule audit:** standardize SMILES with RDKit; reconcile exact/censored IC50 observations and preserve source evidence, conflicting labels, and measurement variability.
4. **Scaffold/source intersection:** determine whether chemical scaffolds are highly concentrated within individual studies. A scaffold-only split may not cover source shifts; a source-held-out test is a *secondary* analysis if sufficiently supported.
5. **Post-model robustness:** plot per-source residual distributions and MAE (with group counts and uncertainty), and compare to scaffold-held-out metrics. A residual difference does not establish causal lab bias without chemistry/assay/construct adjustment.

## Interpretation safeguards

- **Descriptive associations are not causal bias claims.** Source, ligand series, protein construct, research era, and publication selection are all potential confounders.
- The 1000 nM threshold is a **research operating choice**, not a validated clinical efficacy threshold.
- Censored measurements must not be converted to exact pIC50 labels. Figures 03–04 use exact labels only.
- Ratios of observations to InChIKeys are **not independent replicate counts** or deduplicated molecule counts.
- Clinical anticancer efficacy, toxicity, selectivity, ADMET, and de novo generated compound activity cannot be established from these plots.
