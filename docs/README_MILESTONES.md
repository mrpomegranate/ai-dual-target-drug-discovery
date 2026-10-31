# Capstone milestone READMEs — index

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Last updated:** 2026-10-31

These milestone files document work actually undertaken in the repository and retain scientific caveats. All paths in the documents are relative to the `docs/` directory unless a command is explicitly run from the repository root.

| Stage | Documentation | Completion / scope |
|---|---|---|
| M0 | [README_M0.md](README_M0.md) | Environment, GitHub CI, Docker and MLflow persistence foundation |
| M1.1 | [README_M1_1.md](README_M1_1.md) | October 2026 BindingDB ZIP/source schema inspection |
| M1.2 | [README_M1_2.md](README_M1_2.md) | Six actual EGFR/HER2 IC50 records previewed |
| M1.3 | [README_M1_3.md](README_M1_3.md) | Conservative IC50 parser; 23 unit tests passed |
| M1.4 | [README_M1_4.md](README_M1_4.md) | 3,243,660 source rows scanned; 35,210 observation records extracted |
| M1.5 | [README_M1_5.md](README_M1_5.md) | Source/construct/potency diagnostics and seven PNG figures |
| M1.6 | [README_M1_6.md](README_M1_6.md) | Construct-name and provenance diagnostic; script + 44-test overall suite validated locally |

**About M1.6:** You already have `docs/README_M1_6.md`. This ZIP intentionally **does not overwrite it**. Its existing documentation and `reports/audits/M1_6_interpretation.md` remain the M1.6 source of record.

## Sequence of scientific maturity

```text
M0: environment and evidence tooling
  -> M1.1: schema and source provenance
  -> M1.2: inspect real observations
  -> M1.3: IC50 qualifier parsing
  -> M1.4: whole-archive observation extraction
  -> M1.5: source-associated descriptive diagnostics
  -> M1.6: name-based construct and publication/patent grouping
  -> NEXT: sequence/construct verification + molecule standardization
  -> NEXT: assay reconciliation + model-ready label masks
  -> LATER: scaffold-held-out RF vs GNN evaluation
```

**Not yet established:** a curated 2,400/2,529-molecule dual-target training cohort, source-independent potency effects, final label counts, trained model performance, experimental anticancer efficacy, or de novo molecules. M1.6's reported 2,400 cross-target source InChIKeys are an **overlap diagnostic only**, not confirmed paired labels.

## Reproducibility checklist

- Raw BindingDB archive and `data/interim/` outputs should remain outside version control where appropriate; commit source code, documentation, selected aggregate JSON reports and figures after licensing/privacy review.
- Use `uv run --locked` and track `uv.lock` changes when adding packages.
- Log the **input CSV SHA-256** alongside each generated figure/report; regenerate M1.5 reports if their input hash differs from your final corrected M1.4 CSV.
- Retain original IC50 text and all censoring/construct/provenance uncertainty. Do not average measurements across incompatible assay settings or treat missing assays as inactive.
- Review `git status` and `git diff --check` before each research commit.
