# M1.3 — Conservative IC50 parsing and activity evidence

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Date:** 2026-10-31
**Status:** Completed — **23 IC50-specific unit tests passed** and Ruff passed.
**Scope:** Measurement parsing and threshold logic; not molecule curation.

## Objective

Interpret measured IC50 strings consistently while keeping censored inequalities separate from exact numerical potency labels. Avoid accidentally using `<`, `>`, `<=`, or `>=` values as exact pIC50 regression targets.

## Operating rule

The provisional research threshold is **IC50 strictly less than 1,000 nM** for an active target-specific observation. This is not a clinical efficacy threshold.

| Raw IC50 (nM) | Classification at 1000 nM | Exact pIC50 available? | Why |
|---|---|---|---|
| `800` | active | Yes | 800 < 1000 |
| `1000` | inactive | Yes | strict inequality |
| `<1000` | active | No | true value is below 1000 |
| `<=1000` | indeterminate | No | could be exactly 1000 |
| `>1000` | inactive | No | true value exceeds threshold |
| `>=1000` | inactive | No | true value is at least threshold |
| `<2000` | indeterminate | No | actual value could fall on either side |
| `0.000` | parse error | No | nonpositive IC50 invalid for logarithmic transform |
| empty | missing | No | no measured evidence |

For exact, strictly positive IC50 only:

```text
pIC50 = 9 - log10(IC50_nM)
```

The parser rejects malformed, nonpositive and extreme values rather than silently converting them. It returns no exact `pic50` for censored observations; classification is determined from the inequality and threshold where logically possible.

## Implementation

- `../ml/src/dual_target/ic50.py` — `parse_ic50` and `ParsedIC50`.
- `../ml/tests/test_ic50.py` — test coverage for boundaries, censored notation, missing/invalid strings, pIC50 and six real preview values.

## Reproduce (PowerShell)

```powershell
uv run --locked pytest -q ml/tests/test_ic50.py
uv run --locked ruff check ml/src/dual_target/ic50.py ml/tests/test_ic50.py
```

**Observed at M1.3:** `23 passed in 0.03s`, `All checks passed!`. Later overall suite totals include additional milestone tests; they do not change the M1.3 result.

## Scientific decision

Retain the original raw IC50 string, qualifier, parsed numerical bound, activity-evidence classification, and a missing/invalid status. Use **only exact comparable potency values** in conventional pIC50 regression. Censored bounds may still contribute to a separate, carefully specified binary-evidence analysis where logically determinate. Missing evidence is never presumed inactive.

## Limitations

This parser alone does not establish assay quality, compound identity, target construct, independent measurements, replicates, or data splitting. It must operate within the complete source audit (M1.4) and later scientific curation.
