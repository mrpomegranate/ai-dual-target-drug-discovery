# M1.8d — Preliminary interpretation of representation collisions

Based on a local analysis of the uploaded M1.8c preliminary output:

- **4 canonical-SMILES groups** arise from multiple raw SMILES strings, affecting **154 observations**. These are already assigned the same exact-structure ID under M1.8c. They do not imply identical experimental conditions.
- **8 recalculated full InChIKeys** each correspond to two distinct canonical isomeric SMILES and separate M1.8c structure IDs, affecting **41 observations**. Of these rows, **38** are EGFR primary-exact, **2** EGFR primary-classification and **1** HER2 primary-exact. The repeated InChIKey alone is not an adequate label-aggregation rule.
- Example same-key differences include `CS(=O)` versus `C[S+]([O-])` sulfoxide representations, alternate `[nH]` positions in heteroaromatic rings, and other tautomer-like forms. These are **structure-representation hypotheses**, not independent confirmation of assay-material equivalence.
- Do not treat the affected row counts as statistically independent measurements or as verified molecule counts.
- Preserve the original structure/IC50/provenance, keep separate canonical-structure IDs, and apply a leak-resistant grouping rule when preparing held-out splits. Scientific curation of assays, constructs and provenance remains outstanding.

**No model has been trained or evaluated on these data.**
