# Milestone M0: installation and verification

This is a PROPOSED overlay for an **existing** repository; inspect differences before copying files.
No dataset was analyzed and no model metrics have been generated.

## Requirements

- Run commands from repository root in PowerShell (or WSL2 shell with equivalent syntax).
- `uv` installed; Docker Engine with Compose V2 running. Python 3.12 is installed by `uv` if absent.
- Do not copy professor-only files or BindingDB archives into Git.

## Install / verify the Python project

```powershell
uv --version
uv python install 3.12
uv lock
uv sync --locked
uv run --locked dual-target doctor
uv run --locked pytest
uv run --locked ruff check ml/src ml/tests
```

Expect the CLI to print installed Python and scientific-package versions. Tests should pass;
actual counts and software versions must be recorded by the researcher.

Commit `uv.lock` along with pyproject.toml, but never `.venv/`.

## Docker / MLflow

```powershell
docker compose version
docker compose config --quiet
docker compose -f compose.yaml -f compose.dev.yaml config --quiet
docker compose up -d --build mlflow
docker compose ps
docker compose run --rm --build audit doctor
```

MLflow UI: http://127.0.0.1:5000 (unless `MLFLOW_PORT` changes).

## Test that experiment runs AND artifacts persist

```powershell
docker compose exec -T mlflow python /opt/scripts/mlflow_smoke.py create
docker compose restart mlflow
docker compose up -d --wait mlflow
docker compose exec -T mlflow python /opt/scripts/mlflow_smoke.py verify
```

`verify` must print an existing run ID and validate the artifact contents. `restart` may
briefly interrupt connections; if verification sees a connection error, wait for the
container to become healthy and re-run `verify` (do not recreate the run).

The MLflow server is restricted to localhost via the Compose published-port binding.
SQLite metadata and artifacts use distinct Docker named volumes. **Do not run
`docker compose down -v`**, which removes those volumes.

## Evidence to capture for M0

```powershell
git rev-parse HEAD
uv --version
uv run --locked python --version
uv run --locked dual-target doctor
uv run --locked pytest -q
uv run --locked ruff check ml/src ml/tests
docker compose config --quiet
docker compose ps
```

Record the exact test outcomes, Docker versions, image IDs, Git SHA, and MLflow run ID.
The MLflow smoke test logs only infrastructure evidence, never invented potency scores.

## Notes / limitations

- Docker Engine and MLflow runtime cannot be tested from a static source archive.
- The MLflow image pins the direct package version, but its transitive dependencies
  are **not yet fully locked**; add a dedicated service lockfile during hardening.
- The Dockerized audit container currently supplies an environment check, not a
  BindingDB parser. The independent data audit is milestone M1.
- Web services, GPU training and generation are not included in M0.
