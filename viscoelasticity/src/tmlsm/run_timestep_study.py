"""
Timestep Study Experiment
=========================
Tests the influence of temporal resolution (n_timesteps) on model quality.

Uses only the best-performing loadpath config: mixed_4 (corners)
  w={1,1,4,4}, A={1,4,1,4}

Trains GSM, Maxwell NN, and Simple RNN for each timestep setting.
n_timesteps in {50, 100, 200, 400}.

Each (model_type, n_timesteps) combination is trained with 5 seeds.
250k training steps per run. Only the final model is saved.

Usage:
    cd viscoelasticity
    python -m tmlsm.run_timestep_study

Or from Jupyter:
    from tmlsm.run_timestep_study import run_all_experiments
    run_all_experiments()
"""

import json
import time
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jrandom
import klax
import numpy as np

from tmlsm import data as td
from tmlsm import models as tm
from tmlsm.configs import MATERIAL_PARAMS

# Try to register HistoryCallback as PyTree
try:
    jax.tree_util.register_pytree_node(
        klax.HistoryCallback,
        lambda x: ((), x),
        lambda x, _: x,
    )
except Exception:
    pass


# =============================================================================
# Configuration
# =============================================================================

# Fixed loadpath: mixed_4 (corners) — best generalisation across all experiments
TRAIN_OMEGAS = [1, 1, 4, 4]
TRAIN_AS = [1, 4, 1, 4]
CONFIG_NAME = "mixed_4"

# Timesteps to test
TIMESTEP_VALUES = [50, 100, 200, 400]

# Model types to train
MODEL_TYPES = ["gsm", "maxwell_nn", "simple_rnn"]

# Training settings
N_SEEDS = 3
TOTAL_STEPS = 100_000
LOG_EVERY = 500

# Material parameters
E_INFTY = MATERIAL_PARAMS["E_infty"]
E_VAL = MATERIAL_PARAMS["E"]
ETA = MATERIAL_PARAMS["eta"]
G = 1.0 / ETA


# =============================================================================
# Helper Functions
# =============================================================================

def build_model(model_type, key):
    """Build a fresh model of the given type."""
    if model_type == "gsm":
        return tm.build_gsm(key=key, g=G)
    elif model_type == "maxwell_nn":
        return tm.build_maxwell_nn(key=key)
    elif model_type == "simple_rnn":
        return tm.build(key=key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model(model, history_losses, model_type, n_ts, seed, steps, artifacts_dir):
    """Save model .eqx and history .json."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = (
        f"{model_type}__{CONFIG_NAME}__seed_{seed}"
        f"__{steps}steps__{n_ts}ts__{timestamp}.eqx"
    )
    model_path = artifacts_dir / model_filename

    eqx.tree_serialise_leaves(str(model_path), model)

    history_filename = model_filename.replace(".eqx", "_history.json")
    history_path = artifacts_dir / history_filename

    history_data = {
        "model_type": model_type,
        "config_name": CONFIG_NAME,
        "n_timesteps": n_ts,
        "seed": seed,
        "steps": steps,
        "timestamp": timestamp,
        "train_omegas": TRAIN_OMEGAS,
        "train_As": TRAIN_AS,
        "losses": [float(l) for l in history_losses],
        "final_loss": float(history_losses[-1]) if len(history_losses) > 0 else None,
    }

    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    return model_path, history_path


def generate_data(n_timesteps):
    """Generate harmonic training data with the given temporal resolution."""
    return td.generate_data_harmonic(
        E_INFTY, E_VAL, ETA, n_timesteps, TRAIN_OMEGAS, TRAIN_AS
    )


# =============================================================================
# Main Training Loop
# =============================================================================

def run_all_experiments():
    """Run the full timestep study."""

    artifacts_dir = Path("artifacts") / "timestep_study"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    n_ts_values = len(TIMESTEP_VALUES)
    n_model_types = len(MODEL_TYPES)
    total_runs = n_ts_values * n_model_types * N_SEEDS

    print("=" * 70)
    print("TIMESTEP STUDY")
    print("=" * 70)
    print(f"Loadpath:           {CONFIG_NAME}  w={TRAIN_OMEGAS}, A={TRAIN_AS}")
    print(f"Timestep values:    {TIMESTEP_VALUES}")
    print(f"Model types:        {MODEL_TYPES}")
    print(f"Seeds per combo:    {N_SEEDS}")
    print(f"Total training runs: {total_runs}")
    print(f"Steps per run:      {TOTAL_STEPS:,}")
    print(f"Artifacts dir:      {artifacts_dir}")
    print("=" * 70)

    run_counter = 0
    global_start = time.time()

    # Save experiment overview
    overview = {
        "study": "timestep_influence",
        "config_name": CONFIG_NAME,
        "train_omegas": TRAIN_OMEGAS,
        "train_As": TRAIN_AS,
        "timestep_values": TIMESTEP_VALUES,
        "model_types": MODEL_TYPES,
        "n_seeds": N_SEEDS,
        "total_steps": TOTAL_STEPS,
        "material_params": MATERIAL_PARAMS,
        "note": "All timestep values trained fresh in this folder",
        "start_time": datetime.now().isoformat(),
    }
    with open(artifacts_dir / "experiment_overview.json", "w") as f:
        json.dump(overview, f, indent=2)

    for n_ts in TIMESTEP_VALUES:
        print(f"\n{'='*70}")
        print(f"TIMESTEPS: {n_ts}")
        print(f"{'='*70}")

        # Generate training data for this resolution
        eps_train, _, sig_train, dts_train = generate_data(n_ts)
        print(f"  Data shape: eps={eps_train.shape}, sig={sig_train.shape}, dts={dts_train.shape}")

        for model_type in MODEL_TYPES:
            print(f"\n  --- Model: {model_type} ---")

            for seed_idx in range(N_SEEDS):
                run_counter += 1
                run_start = time.time()

                print(f"    Seed {seed_idx} (Run {run_counter}/{total_runs}) ... ", end="", flush=True)

                # Create reproducible key
                master_key = jrandom.PRNGKey(
                    seed_idx * 1000 + hash((CONFIG_NAME, model_type, n_ts)) % 10000
                )
                key_init, key_train = jrandom.split(master_key, 2)

                # Build fresh model
                model = build_model(model_type, key_init)

                # Train
                model, history = klax.fit(
                    model,
                    ((eps_train, dts_train), sig_train),
                    batch_axis=0,
                    steps=TOTAL_STEPS,
                    history=klax.HistoryCallback(log_every=LOG_EVERY),
                    key=key_train,
                )

                # Collect losses
                all_losses = list(history.loss) if hasattr(history, "loss") else []
                final_loss = float(all_losses[-1]) if all_losses else float("nan")

                run_time = time.time() - run_start

                # Save
                model_path, _ = save_model(
                    model=model,
                    history_losses=all_losses,
                    model_type=model_type,
                    n_ts=n_ts,
                    seed=seed_idx,
                    steps=TOTAL_STEPS,
                    artifacts_dir=artifacts_dir,
                )

                print(
                    f"loss={final_loss:.6e}, "
                    f"time={run_time:.1f}s, "
                    f"saved: {model_path.name}"
                )

    total_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"TIMESTEP STUDY COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total models saved: {total_runs}")
    print(f"Artifacts in: {artifacts_dir}")
    print(f"{'='*70}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_all_experiments()
