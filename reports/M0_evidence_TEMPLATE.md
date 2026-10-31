# M0 Evidence Report — TEMPLATE (fill after actual local execution)

Status: NOT YET EXECUTED / NOT VERIFIED

## Repository
- Commit SHA: _pending_
- UV version: _pending_
- Python version: _pending_
- Docker and Compose versions: _pending_
- `uv.lock` committed: _pending_

## Checks (copy actual commands and exit codes)
| Command | Actual exit code | Evidence/log location |
|---|---|---|
| `uv sync --locked` | pending | pending |
| `uv run --locked pytest` | pending | pending |
| `uv run --locked ruff check ml/src ml/tests` | pending | pending |
| `docker compose config --quiet` | pending | pending |
| `docker compose run --rm --build audit doctor` | pending | pending |
| MLflow create / restart / verify | pending | pending |

## MLflow persistence proof
- Experiment name: m0-infrastructure-smoke
- Created run ID: _pending_
- Verified run ID after restart: _pending_
- Artifact: checks/persistence.txt

## Open issues
- MLflow transitive dependencies are not fully locked in the M0 prototype.
- M1 dataset audit, GPU training, and web app are not implemented.
- Additional local observations: _pending_
