# Milestone M1 — Independent EGFR/HER2 IC50 Re-audit: Initial Plan

**Status:** Proposed scientific protocol; source archive not yet examined in this milestone. **Do not report sample counts yet.**

## Scientific objective

Independently re-audit measured inhibitory potency (IC50) for human EGFR (UniProt `P00533`) and HER2/ERBB2 (UniProt `P04626`) from the BindingDB October 2026 release. Document inclusion/exclusion decisions, source provenance, compound identity, measurement operator/units, comparable assay context, duplicate and discordant observations. ChEMBL can be added as an explicitly identified secondary source for cross-checking overlap, not a blind concatenation.

## Gate sequence

1. **M1.0: Source inventory:** Confirm exact file path, archive contents, internal table name, available column headers, release metadata and SHA-256. Do not decompress the entire multi-GB archive simply to inspect columns. Save the machine-readable source manifest.
2. **M1.1: Column/target mapping:** Build explicit mappings from real input headers to stable internal field names; recognize EGFR/HER2 only through documented, verified identifiers, organism and assay/construct rules. Keep all original source fields needed for audit.
3. **M1.2: Observation normalization:** Stream rows/chunks. Parse SMILES with RDKit, retain raw IDs and original SMILES, apply versioned canonicalization, preserve stereochemistry according to policy, flag salts/multicomponent/invalid structures. Parse IC50 values, units and qualifiers, including absent, malformed, censored and interval values; never convert an inequality to an exact value.
4. **M1.3: Evidence reconciliation:** Examine replicate measures, patents/articles, shared source IDs, BindingDB–ChEMBL overlap (if included), assay comparability and conflict severity. Record explicit reasons for exclusions and quarantines. Do not silently average incompatible assays.
5. **M1.4: Molecule-level curation:** Export observation-level table plus a molecule-centric table with curated target pIC50 values and missing-label masks. Compute dual-activity status only where evidence permits a definitive conclusion for both targets; uncertain cases remain unknown.
6. **M1.5: Independent verification:** Check deterministic manifests/hashes, unit tests for edge cases, inclusion/exclusion waterfalls, actual per-target/paired denominators, and reconciliation against the *preliminary* audit counts. Produce evidence report before M2 scaffold splitting.

## Initial molecule and measurement policy (subject to real-data inspection)

- Primary targets: human EGFR `P00533` and HER2 `P04626`, validated against source identifier semantics. Complex or ambiguous target constructs are quarantined until scientifically justified.
- Primary potency task: exact, strictly positive IC50 only; `pIC50 = 9 - log10(IC50_nM)`.
- Dual-active definition: credible evidence that both targets have IC50 strictly below 1,000 nM. An exact 1,000 nM value is **not active**.
- Example bounded observations: `< 1000 nM` can support active threshold classification (quality permitting) but **cannot** supply exact regression pIC50; `> 1000 nM` is definitively non-active for that target; `< 2000 nM` is indeterminate for the 1,000 nM threshold.
- Absent target assays do **not** imply inactivity. Preserve both per-target label masks and evidence-status fields.
- Chemical identity: preserve raw IDs/SMILES; version RDKit canonicalization, structure standardization, stereochemistry and salt policy; do not merge stereoisomers silently.
- Replicate disagreements and non-comparable assay conditions must remain traceable. The actual label-reconciliation policy will be finalized only after inspecting observed contexts and conflicts.

## Proposed output schema

**`observations.parquet` (long form)**: `observation_id`, `source_database`, `source_release`, `source_record_id`, `source_reference_id`, `raw_target_name`, `target_uniprot`, `organism`, `assay_id`, `assay_type`, `assay_context`, `raw_smiles`, `raw_molecule_id`, `canonical_smiles`, `molecule_id`, `ic50_raw`, `ic50_unit`, `ic50_operator`, `ic50_nm`, `pic50_exact`, `censoring_status`, `include_status`, `exclusion_reason`.

**`molecules.parquet` (wide form)**: `molecule_id`, `canonical_smiles`, `egfr_pic50`, `egfr_mask`, `egfr_evidence_status`, `her2_pic50`, `her2_mask`, `her2_evidence_status`, `paired_status`, `dual_active`, `provenance_ids`. Scaffold IDs are added or finalized in M2 using a documented empty/acyclic scaffold policy.

**`manifest.json`**: source archive SHA-256/release, parser/curation/config versions and hash, RDKit version, accepted source columns, exact command, target mapping, count waterfall, exclusion and conflict breakdown, final artifact paths and checksums.

## M1 first local commands (PowerShell, from repository root)

```powershell
Get-ChildItem .\data\raw -File | Select-Object Name, Length, FullName
```

If the BindingDB archive is somewhere else, identify its exact path and configure a read-only bind mount rather than committing it or embedding it into a Docker image. After verifying the archive path, inspect the ZIP central directory without extracting it:

```powershell
$path = 'REPLACE_WITH_ACTUAL_ARCHIVE_PATH.zip'
Add-Type -AssemblyName System.IO.Compression
$archive = [System.IO.Compression.ZipFile]::OpenRead($path)
try {
    $archive.Entries | Select-Object FullName, Length, CompressedLength
} finally {
    $archive.Dispose()
}
```

After confirming the exact source file, calculate its hash for the immutable input manifest:

```powershell
Get-FileHash -Algorithm SHA256 -Path $path
```

**Stop here for M1.0 review:** the actual file inventory and internal archive entries will determine our ingestion code and field mapping. Do not assume preliminary counts or fabricate column names.
