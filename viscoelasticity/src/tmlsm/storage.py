"""Storage utilities for saving/loading models and results."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import equinox as eqx
import numpy as np

if TYPE_CHECKING:
    from .configs import ExperimentConfig, ModelType
    from .experiments import ExperimentResult, ModelResult


# =============================================================================
# Naming Schema
# =============================================================================

def get_model_filename(
    model_type: str,
    experiment_name: str,
    train_steps: int,
    n_timesteps: int,
    timestamp: str | None = None,
) -> str:
    """Generate a descriptive filename for a model.

    Format: {model_type}__{experiment}__{steps}steps__{n}ts__{timestamp}.eqx

    Example: gsm__baseline__100000steps__100ts__20260130_143022.eqx
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{model_type}__{experiment_name}__{train_steps}steps__{n_timesteps}ts__{timestamp}.eqx"


def parse_model_filename(filename: str) -> dict[str, Any]:
    """Parse a model filename back into its components."""
    # Remove extension
    name = filename.replace(".eqx", "")
    parts = name.split("__")

    if len(parts) != 5:
        raise ValueError(f"Invalid filename format: {filename}")

    return {
        "model_type": parts[0],
        "experiment_name": parts[1],
        "train_steps": int(parts[2].replace("steps", "")),
        "n_timesteps": int(parts[3].replace("ts", "")),
        "timestamp": parts[4],
    }


def get_results_filename(experiment_name: str, timestamp: str | None = None) -> str:
    """Generate filename for experiment results JSON."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"results__{experiment_name}__{timestamp}.json"


# =============================================================================
# Model Saving/Loading
# =============================================================================

def save_model(
    model: Any,
    model_type: str,
    experiment_name: str,
    train_steps: int,
    n_timesteps: int,
    artifacts_dir: str | Path = "artifacts",
    timestamp: str | None = None,
) -> Path:
    """Save an Equinox model to disk.

    Args:
        model: The trained Equinox model
        model_type: Type of model (e.g., "gsm", "simple_rnn")
        experiment_name: Name of the experiment
        train_steps: Number of training steps
        n_timesteps: Number of timesteps in training data
        artifacts_dir: Directory to save to
        timestamp: Optional timestamp (auto-generated if None)

    Returns:
        Path to the saved model file
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = get_model_filename(
        model_type, experiment_name, train_steps, n_timesteps, timestamp
    )
    filepath = artifacts_dir / filename

    eqx.tree_serialise_leaves(str(filepath), model)

    return filepath


def load_model(
    filepath: str | Path,
    model_template: Any,
) -> Any:
    """Load an Equinox model from disk.

    Args:
        filepath: Path to the .eqx file
        model_template: An uninitialized model with the same structure
                       (needed because eqx needs to know the tree structure)

    Returns:
        The loaded model
    """
    return eqx.tree_deserialise_leaves(str(filepath), model_template)


# =============================================================================
# Results Saving/Loading (JSON)
# =============================================================================

def _make_serializable(obj: Any) -> Any:
    """Convert numpy arrays and other non-JSON types to serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif hasattr(obj, "__dataclass_fields__"):
        return _make_serializable(asdict(obj))
    else:
        return obj


def save_results_json(
    result: ExperimentResult,
    artifacts_dir: str | Path = "artifacts",
    timestamp: str | None = None,
) -> Path:
    """Save experiment results (metrics only, no models) to JSON.

    Args:
        result: ExperimentResult to save
        artifacts_dir: Directory to save to
        timestamp: Optional timestamp

    Returns:
        Path to the saved JSON file
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = get_results_filename(result.config.name, timestamp)
    filepath = artifacts_dir / filename

    # Build serializable dict (exclude model objects and predictions)
    data = {
        "config": {
            "name": result.config.name,
            "description": result.config.description,
            "train_loadcases": result.config.train_loadcases,
            "test_loadcases": result.config.test_loadcases,
            "n_timesteps": result.config.n_timesteps,
            "n_periods": result.config.n_periods,
            "train_steps": result.config.train_steps,
            "models": list(result.config.models),
            "test_relaxation": result.config.test_relaxation,
        },
        "model_results": {},
        "timestamp": timestamp,
    }

    for model_type, model_result in result.model_results.items():
        data["model_results"][model_type] = {
            "model_type": model_result.model_type,
            "experiment_name": model_result.experiment_name,
            "train_time": model_result.train_time,
            "train_steps": model_result.train_steps,
            "final_loss": model_result.final_loss,
            "harmonic_metrics": _make_serializable(model_result.harmonic_metrics),
            "relaxation_metrics": _make_serializable(model_result.relaxation_metrics),
            # Note: predictions and model are NOT saved (too large / not serializable)
        }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def load_results_json(filepath: str | Path) -> dict:
    """Load experiment results from JSON.

    Returns a dict (not ExperimentResult) since we don't have models/predictions.
    """
    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# Convenience Functions
# =============================================================================

def save_experiment(
    result: ExperimentResult,
    artifacts_dir: str | Path = "artifacts",
    save_models: bool = True,
    save_json: bool = True,
) -> dict[str, Path]:
    """Save a complete experiment (models + results JSON).

    Args:
        result: ExperimentResult to save
        artifacts_dir: Directory to save to
        save_models: Whether to save model weights
        save_json: Whether to save metrics as JSON

    Returns:
        Dict mapping model_type/json -> filepath
    """
    artifacts_dir = Path(artifacts_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_files = {}

    # Save JSON results
    if save_json:
        json_path = save_results_json(result, artifacts_dir, timestamp)
        saved_files["json"] = json_path

    # Save models
    if save_models:
        for model_type, model_result in result.model_results.items():
            if model_result.model is not None:
                # Skip analytical maxwell (no trainable params)
                if model_type == "maxwell":
                    continue

                model_path = save_model(
                    model_result.model,
                    model_type,
                    result.config.name,
                    result.config.train_steps,
                    result.config.n_timesteps,
                    artifacts_dir,
                    timestamp,
                )
                saved_files[model_type] = model_path

    return saved_files


def list_artifacts(artifacts_dir: str | Path = "artifacts") -> dict[str, list[Path]]:
    """List all saved artifacts.

    Returns:
        Dict with "models" and "results" keys containing lists of paths
    """
    artifacts_dir = Path(artifacts_dir)

    if not artifacts_dir.exists():
        return {"models": [], "results": []}

    models = sorted(artifacts_dir.glob("*.eqx"))
    results = sorted(artifacts_dir.glob("results__*.json"))

    return {"models": models, "results": results}


def find_latest_model(
    model_type: str,
    experiment_name: str | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> Path | None:
    """Find the most recent model of a given type.

    Args:
        model_type: Type of model to find
        experiment_name: Optional filter by experiment
        artifacts_dir: Directory to search

    Returns:
        Path to the most recent matching model, or None
    """
    artifacts_dir = Path(artifacts_dir)

    pattern = f"{model_type}__"
    if experiment_name:
        pattern += f"{experiment_name}__"
    pattern += "*.eqx"

    matches = sorted(artifacts_dir.glob(pattern))

    return matches[-1] if matches else None
