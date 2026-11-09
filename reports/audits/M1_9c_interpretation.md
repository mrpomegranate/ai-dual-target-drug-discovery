# M1.9c — Preliminary evidence findings

The M1.9c preview selected the ten highest-priority *EGFR* and ten highest-priority *HER2* cases. They produced **1,001 exact IC50 observation rows** across **656 publication/patent provenance groups**. The use of `target + m18c_structure_id` prevents chemical-identity mistakes; restricting to `primary_exact_candidate` prevents mutant-construct observations from silently entering reference-target assessments.

**Metadata coverage across the selected evidence:** 1,001 rows have a source category; 655 have an article DOI; 946 have a PMID; 52 have a patent number; 58 have a PubChem AID; 12 have pH; 12 have temperature. There are 62 rows participating in candidate repeated-value/context signatures. These are not established duplicate experiments. The provenance group check finds 115 groups with a within-group pIC50 range >1.0 (greater than tenfold IC50 spread). These metrics describe an **extreme, purposively selected review sample**, not population-level rates.

This dataset can support a priority list for original-source investigation, not a causal lab-batch-effect estimate or a curated potency label. Assay media, ATP concentration, protein sequence and additional experimental conditions remain unverified. No observations were altered or aggregated.
