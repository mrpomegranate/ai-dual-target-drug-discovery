"""M0 persistence check: create a real tracking run, restart, verify artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "m0-infrastructure-smoke"
ARTIFACT_PATH = "checks/persistence.txt"
ARTIFACT_TEXT = "M0 persistence smoke test - no scientific measurements.\n"


def create() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="persistence-check") as run:
        mlflow.set_tag("milestone", "M0")
        mlflow.log_param("check", "persist_across_restart")
        mlflow.log_text(ARTIFACT_TEXT, ARTIFACT_PATH)
        print(json.dumps({"created_run_id": run.info.run_id, "artifact": ARTIFACT_PATH}))


def verify() -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Smoke experiment not found after restart")
    runs = client.search_runs([experiment.experiment_id], max_results=100)
    for run in runs:
        if run.data.tags.get("milestone") != "M0":
            continue
        destination = Path(
            mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path=ARTIFACT_PATH)
        )
        if destination.read_text(encoding="utf-8") != ARTIFACT_TEXT:
            raise RuntimeError("Artifact content changed or disappeared")
        print(json.dumps({"verified_run_id": run.info.run_id, "artifact": ARTIFACT_PATH}))
        return
    raise RuntimeError("No M0 run found after restart")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    if args.action == "create":
        create()
    else:
        verify()


if __name__ == "__main__":
    main()
