"""Shared MLflow experiment/tracking setup.

One sqlite-backed tracking store for the whole project (`sqlite:///mlruns/mlflow.db`) rather than
the plain bare-filesystem `./mlruns` store MLflow 3.x deprecated -- see CLAUDE.md Tech Stack.
Every stage that logs a real experiment (geometry validation, later biomass/activity runs) should
call `set_experiment()` once before its own `mlflow.start_run()`, so all runs land in the same
store and are comparable via `mlflow.search_runs()`.
"""
import mlflow

TRACKING_URI = "sqlite:///mlruns/mlflow.db"


def set_experiment(name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(name)
