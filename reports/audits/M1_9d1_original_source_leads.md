# M1.9d.1 — Original-source review leads

**Status:** Manually verified *publication descriptions* and observed BindingDB record links; **assay-row verification is outstanding**. No label decisions made.

The M1.9c prioritized evidence includes **1,001 observations across 656 provenance groups**, with 62 candidate repeated-value/context signature rows. In the source-review queue, 115 groups have a >10-fold within-provenance IC50 spread and 28 cross the active/inactive threshold. This is an *extreme purposively selected review sample*, not a prevalence estimate for BindingDB.

## Specific publication checks

| Source | Rows in our 20-case sample | Observed BindingDB connection | What the publication describes | Next verification |
|---|---:|---|---|---|
| [PMID 32931277](https://pubmed.ncbi.nlm.nih.gov/32931277/) (DOI 10.1021/acs.jmedchem.0c00870) | 20 (19 EGFR, 1 HER2) | Plain EGFR/HER2 target-name observations | Article describes wild-type HER2 and HER2 A775_G776insYVMA variant screening; it may include additional family-member selectivity assays | Identify each selected EGFR IC50's assay/construct in its supplementary tables; distinguish HER2 mutant from EGFR counterscreen |
| [PMID 37197473](https://pubmed.ncbi.nlm.nih.gov/37197473/) (DOI 10.1021/acsmedchemlett.2c00514) | 7 EGFR | All seven have plain EGFR target-name strings | Publication focuses on EGFR L858R/C797S and L858R/T790M/C797S reversible inhibitors and cell-based activity | Verify whether each IC50 is WT EGFR or mutant EGFR and whether biochemical versus cellular |
| [PMID 36375337](https://pubmed.ncbi.nlm.nih.gov/36375337/) (DOI 10.1016/j.ejmech.2022.114889) | 2 EGFR | Includes record `51573947` reported IC50 **83,000,000 nM** | Article describes mutant EGFR enzyme assays and anticancer cell-line viability assays | Verify source table and units for record 51573947 (83 mM equivalent); do not assume typographical error |
| [PMID 20797871](https://pubmed.ncbi.nlm.nih.gov/20797871/) (DOI 10.1016/j.bmc.2010.08.004) | 2 EGFR | Two exact IC50 observations, **0.24 nM** and **8.02 nM** | EGFR imaging-agent study reports kinase affinity and cellular potency as distinct outcomes | Match both records to their compound identity and endpoint; different biological endpoints cannot be automatically pooled |

The linked PubMed records substantiate the **topics** of the publications. They do not establish whether an individual BindingDB observation is correct or which original table its number came from. In particular, a plain BindingDB target name may conceal variant-dependent assays; independent verification is required before treating it as an unmodified-protein label.

## Before potency-label reconciliation

1. Reconstruct record-to-table provenance, actual target construct, endpoint, and units for selected priority cases.
2. Preserve exact source record IDs, measured IC50, experimental context and disagreements.
3. Assign statuses such as `verified_comparable`, `verified_different_endpoint`, `verified_different_construct`, `suspected_duplicate_reporting`, and `unresolved` only with evidence.
4. Decide on assay inclusion and conflict-handling policy **before** creating any averaged pIC50 target.

## Counts and caveats

- 1,001 records with 1,001 distinct `reactant_set_id` values; all `primary_exact_candidate` and `provisional_exact_structure`.
- 938 ChEMBL, 48 US Patent, 11 BindingDB literature curation, 4 WIPO; source mix is strongly skewed by selecting the 20 most extreme cases.
- Metadata reported in source CSV: PMID 946/1,001, DOI 655/1,001, PubChem AID 58/1,001, pH 12/1,001, temperature 12/1,001.
- `provenance_key` groups are publications/patents, **not verified laboratories or assays**.
- The queue is explicitly conservative: **no IC50 correction, exclusion, or label aggregation is performed**.
