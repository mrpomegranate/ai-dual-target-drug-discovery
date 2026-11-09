# M1.9d.2 — Original-publication leads and evidence status

**Status:** Publication-level research only. No individual BindingDB measurement has been verified against an original assay table. **No curation decisions made.**

## Lead A: PMID 32931277 — high priority

- [PubMed: *Targeting Her2-insYVMA with Covalent Inhibitors—A Focused Compound Screening and Structure-Based Design Approach*](https://pubmed.ncbi.nlm.nih.gov/32931277/)
- DOI: `10.1021/acs.jmedchem.0c00870`, 2020, *Journal of Medicinal Chemistry*.
- The publication abstract reports a screening campaign applied against wild-type HER2 and the HER2 `A775_G776insYVMA` mutant and subsequent structure-based design.
- In the **uploaded BindingDB evidence**, monomer ID `50029668` appears in this source with plain EGFR target names and IC50 values `911`, `22`, `0.20`, `0.06`, and `0.02` nM (reactant-set IDs `51352922`, `51352915`, `51352946`, `51352938`, `51352930`). That is a 45,550-fold span.
- **NOT established:** Whether the five EGFR-assigned records represent the same EGFR construct, different kinase mutants, a combination of biochemical and cellular assays, or an import/mapping issue. Match these five records to tables and supporting information before interpreting them.
- ACS article page describes supporting information including activity screening results and IC50/EC50 tables: https://pubs.acs.org/doi/10.1021/acs.jmedchem.0c00870

## Lead B: PMID 37197473 — high priority

- [PubMed: *Addressing the Osimertinib Resistance Mutation EGFR-L858R/C797S with Reversible Aminopyrimidines*](https://pubmed.ncbi.nlm.nih.gov/37197473/)
- [Open full-text article (PMC10184309)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10184309/)
- DOI: `10.1021/acsmedchemlett.2c00514`, 2023, *ACS Medicinal Chemistry Letters*.
- **Publication-level verification:** This paper reports biochemical EGFR **WT**, **L858R**, and **L858R/C797S** measured with an HTRF assay, while mutant EGFR-dependent Ba/F3 cellular viability was measured with a CTG assay (Table 1 note). Its text reports an osimertinib EGFR-L858R/C797S biochemical IC50 of **354 nM**.
- The uploaded BindingDB evidence includes an EGFR observation of **354 nM**, reactant-set `51528540`, monomer ID `50029668`, under this PMID. **A matching numerical value is not sufficient to prove** the exact compound/construct/assay mapping. Record-to-table verification remains open.
- This paper provides a concrete example of how **protein genotype** and **endpoint format** can change within one publication, even if a database's generic target name stays unchanged.

## Lead C: PMID 36375337 — high-priority extreme value

- [PubMed: *Design, synthesis, and biological evaluation of novel quinoline derivatives as small molecule mutant EGFR inhibitors targeting resistance in NSCLC*](https://pubmed.ncbi.nlm.nih.gov/36375337/)
- DOI: `10.1016/j.ejmech.2022.114889`, 2023, *European Journal of Medicinal Chemistry*.
- The source concerns mutant EGFR inhibition and in-vitro testing. An uploaded BindingDB record linked to this PMID reports **83,000,000 nM** for a plain-name EGFR observation (reactant-set `51573947`, monomer `31090`).
- **NOT established:** Whether that number came from a biochemical inhibition assay, a cellular endpoint, conversion of units, a reported bound, or a transcription/normalization error. It should be checked, not automatically deleted.

## What constitutes actual completion of row-level source verification?

A completed worksheet row needs a source table/figure/supplement location and a verified compound identity, exact endpoint, protein construct, units, and duplicate/reporting context. No such claim has been made here.
