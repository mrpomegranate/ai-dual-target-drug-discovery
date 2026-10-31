# Milestone M0 — Reproducible Research Foundation: Evidence Report

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Date of evidence:** 2026-10-31
**Status:** **M0 acceptance checks passed**, based on the user's local command output and GitHub Actions screenshot. This report was prepared **after** the original M0 commit and must be separately added to version control.
**Scope:** Engineering foundation only. No real BindingDB/ChEMBL audit, model training, molecular predictions, or performance metrics were executed.

## 1. Source control provenance

| Field | Actual observed value |
|---|---|
| Repository | `https://github.com/mrpomegranate/ai-dual-target-drug-discovery` |
| Branch | `main` |
| Foundation commit (full SHA) | `ad37ae198e3acd627e3c15bf6a53957fc0a1965b` |
| Commit subject | `chore(m0): establish reproducible research foundation` |
| Commit changes | 17 added files; 743 insertions |
| Working tree immediately after commit | Clean |
| Remote push | `main -> main` successful; upstream `origin/main` established |
| Lockfile | `uv.lock` included in foundation commit |

**Committed file classes:** root `pyproject.toml`, `.python-version`, `uv.lock`, Python package and foundation tests, Docker Compose definitions, audit and MLflow Dockerfiles, MLflow smoke script, CI workflow, example environment configuration, ignore rules, and M0 readme/template. The course-restricted professor guidance and raw data were not listed among the committed files.

## 2. Local checks actually observed

| Command/check | Observed outcome | Evidence / qualification |
|---|---|---|
| `docker compose version` | Passed — Compose `v5.1.3` | Local terminal transcript |
| `docker compose config --quiet` | Passed — no error output | Local terminal transcript |
| `docker compose up -d --build mlflow` | Passed on retry | First attempt failed while Docker Desktop Linux engine was unavailable; second built and started |
| `docker compose ps` | MLflow `Up ... (healthy)` | `127.0.0.1:5000->5000/tcp` |
| `docker compose run --rm --build audit doctor` | Passed — JSON status `ok` | Built a non-development Python environment inside Docker |
| `uv run --locked pytest -q` | **4 passed in 0.06 s** | Local user terminal output |
| `uv run --locked ruff check ml/src ml/tests` | **All checks passed** | Local user terminal output |
| `git diff --cached --check` | Passed — no output | Local user terminal output |
| `git commit ...` | Passed | Foundation SHA above |
| `git push -u origin main` | Passed | GitHub `main` branch created and tracking established |
| GitHub Actions `M0 Python checks` | **Success** | Screenshot: `python-checks` job, 11 s; total run duration 14 s |

**Exit codes:** The supplied transcript did not print explicit numeric `$LASTEXITCODE` values for each command. Success statuses are based on the observed command output and the green GitHub Actions status; do not represent these as separately captured exit codes.

**Software versions actually observed inside the audit container:**

| Component | Version |
|---|---|
| Python | 3.12.16 |
| Project package | 0.1.0 |
| NumPy | 2.5.4 |
| pandas | 2.3.3 |
| PyArrow | 25.0.1 |
| RDKit | 2026.9.1 |
| MLflow server image direct dependency | 3.16.1 |
| Docker Compose | 5.1.3 |

`uv --version`, Docker Engine version, and a standalone `uv sync --locked` command output were not supplied. The successful `uv run --locked` checks and Docker image build support the environment's usability, but are not a transcript of those separate commands.

## 3. MLflow persistence proof (observed)

| Field | Evidence |
|---|---|
| Experiment | `m0-infrastructure-smoke` |
| Run name | `persistence-check` |
| Created run ID | `b64c75785c0744d9b575a54d8dd26737` |
| Stored artifact | `checks/persistence.txt` |
| Restart | `docker compose restart mlflow` followed by `docker compose up -d --wait mlflow` |
| Post-restart status | `Healthy` |
| Verified run ID | `b64c75785c0744d9b575a54d8dd26737` |
| Verification output | `{"verified_run_id":"b64c75785c0744d9b575a54d8dd26737","artifact":"checks/persistence.txt"}` |

**Interpretation:** The same run and its expected artifact were retrieved after restarting the MLflow container, supporting persistence across a **container restart**. This is not a disaster-recovery or cross-host restoration test.

The smoke run stores infrastructure evidence only; it is **not** a scientific model evaluation.

## 4. CI provenance

- Workflow: `.github/workflows/ci.yaml`, `M0 Python checks`.
- Trigger: GitHub push to `main`.
- Screenshot: overall **Success**, `python-checks` passed, total duration **14 seconds**.
- Reported warning: some installed GitHub Actions depended on the deprecated Node.js 20 runtime and were automatically executed under Node.js 24. This did **not** fail the run; update action major versions during a separate maintenance change.
- Informational notice: `ubuntu-latest` image migration scheduled by GitHub; no failure observed.
- The GitHub workflow is a clean-environment code check, **not** an end-to-end production Docker or scientific-data validation.

## 5. Open issues and limitations

1. **Git metadata inside the MLflow container:** MLflow emitted a warning because the Git executable was unavailable. This did not prevent the smoke run. Later pipelines should explicitly log `git_commit` and source/data/split hashes from the caller rather than assuming the tracking server can infer them.
2. **MLflow dependency resolution:** Its Dockerfile pins `mlflow==3.16.1`, but the transitive server environment is not yet fully controlled by a dedicated lockfile. Harden before final reproducibility claims.
3. **Raw-data access:** No BindingDB archive was supplied to or processed by the M0 audit job; `doctor` verified the installed environment only.
4. **Science gate:** Source overlap, IC50 censoring/units, canonical molecule identity, assay harmonization, final sample sizes and masks must be independently re-audited in M1.
5. **Optional components:** No RF/GNN metrics, GPU training, FastAPI inference, web frontend, or molecular generation exist yet.
6. **Repository privacy review:** Do not commit course-restricted guidance, credentials or datasets without verifying permission/licenses.

## 6. Exit decision

**M0 infrastructure exit gate: PASS, with documented caveats.** Proceed to **M1 source inventory and independent EGFR/HER2 IC50 audit**. The M0 evidence report itself needs its own Git commit; its inclusion in the original M0 foundation SHA must not be implied.
