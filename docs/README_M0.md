# M0 — Reproducible research foundation

**Project:** AI Dual-Target Drug Discovery (EGFR/HER2)
**Evidence date:** 2026-10-31
**Status:** Completed — engineering acceptance checks passed.
**Milestone type:** Repository, runtime, experiment-tracking, and CI setup. **No molecular model was trained.**

## Objective

Establish a reproducible development foundation for the capstone before processing experimental drug-target data. The research question is whether a molecular graph neural network (GNN) improves inhibitory-potency prediction over a Morgan-fingerprint + random forest baseline for EGFR and HER2 under scaffold-held-out evaluation. Molecule generation is a possible later extension, **not** part of M0.

## Implemented components

| Component | Evidence / implementation |
|---|---|
| Git repository | `https://github.com/mrpomegranate/ai-dual-target-drug-discovery` |
| Python environment | Python 3.12, `uv`, `pyproject.toml`, `uv.lock` |
| Installable ML package | `ml/src/dual_target/`, with `dual-target doctor` CLI |
| Code checks | `pytest`, Ruff, GitHub Actions workflow `.github/workflows/ci.yaml` |
| Containers | `compose.yaml`, optional `compose.dev.yaml`, audit and MLflow images |
| Experiment tracking | Local MLflow server with persisted SQLite metadata and artifact storage |
| Source hygiene | Ignore raw data, generated outputs as appropriate, credentials, `.venv` and course-restricted material |

## Recorded local and CI evidence

- Git foundation commit: `ad37ae198e3acd627e3c15bf6a53957fc0a1965b` on `main`, pushed successfully.
- Foundation Python check: **4 tests passed**; Ruff checks passed.
- GitHub Actions **M0 Python checks** workflow passed (overall successful run).
- Docker Compose validation passed; MLflow launched in a healthy state at `127.0.0.1:5000`.
- Dockerized audit environment returned `status: ok`; it was an **environment check, not a data audit**.
- MLflow experiment `m0-infrastructure-smoke` created run `b64c75785c0744d9b575a54d8dd26737`, logged artifact `checks/persistence.txt`, and retrieved that same run/artifact after a container restart.
- MLflow smoke testing established restart persistence only, not disaster recovery or scientific correctness.

At the time of M0 evidence collection, the audit container reported Python `3.12.16`, NumPy `2.5.4`, pandas `2.3.3`, PyArrow `25.0.1`, and RDKit `2026.9.1`; MLflow's directly pinned server dependency was `3.16.1`. These are historical observations, not guaranteed versions for future rebuilds.

## Repeat the baseline checks (PowerShell, from repository root)

```powershell
uv run --locked dual-target doctor
uv run --locked pytest -q
uv run --locked ruff check ml/src ml/tests
docker compose config --quiet
docker compose ps
```

To test the MLflow environment and a new persistence run where appropriate:

```powershell
docker compose up -d --build mlflow
docker compose run --rm --build audit doctor
docker compose exec -T mlflow python /opt/scripts/mlflow_smoke.py create
docker compose restart mlflow
docker compose up -d --wait mlflow
docker compose exec -T mlflow python /opt/scripts/mlflow_smoke.py verify
```

**Warning:** Do not run `docker compose down -v` if you want to preserve the MLflow Docker volumes.

## Artifacts and references

- `../reports/milestones/M0_evidence.md` — detailed evidence with observed commands, run ID, commit, caveats; generated separately from the original M0 foundation commit.
- `milestones/M1_audit_plan.md` — initial M1 scientific plan; later stage names evolved, so use the concrete M1.1–M1.6 README series for the executed steps.
- `../.github/workflows/ci.yaml`, `../compose.yaml`, `../pyproject.toml`, `../uv.lock`.

## Limitations and exit criteria

M0 did **not** audit BindingDB, produce training labels, evaluate RF or GNN metrics, build a web application, or generate new molecules. The MLflow server's transitive dependencies were not yet fully locked in a dedicated service lockfile. M0 completed the engineering gate and enabled the independent data work documented in M1.
