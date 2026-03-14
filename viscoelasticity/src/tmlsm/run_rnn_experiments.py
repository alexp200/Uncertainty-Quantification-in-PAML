"""
Simple RNN Experiment Training Script
=====================================
Trains Simple RNN models across multiple configurations.

Same experiment groups as GSM sweep:
  1. Omega sweep:   increasing number of load cases varying omega (A=1 fixed)
  2. Amplitude sweep: increasing number of load cases varying A (omega=1 fixed)
  3. Mixed configs:  2 mixed load case sets (corners + moderate)

Each config is trained with 5 different seeds.
Only the final model (250k steps) is saved.

Usage:
    cd viscoelasticity
    python -m tmlsm.run_rnn_experiments

Or from Jupyter:
    from tmlsm.run_rnn_experiments import run_all_experiments
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

# Try to register HistoryCallback as PyTree (same patch as experiments.py)
try:
    jax.tree_util.register_pytree_node(
        klax.HistoryCallback,
        lambda x: ((), x),
        lambda x, _: x,
    )
except Exception:
    pass


# =============================================================================
# Experiment Configurations (identical to GSM sweep)
# =============================================================================

EXPERIMENTS = {
    # --- Omega Sweep (A=1 fixed, adding frequencies) ---
    "omega_1": {
        "omegas": [1],
        "As": [1],
        "description": "Omega sweep: w={1}, A={1}",
    },
    "omega_2": {
        "omegas": [1, 2],
        "As": [1, 1],
        "description": "Omega sweep: w={1,2}, A={1,1}",
    },
    "omega_3": {
        "omegas": [1, 2, 3],
        "As": [1, 1, 1],
        "description": "Omega sweep: w={1,2,3}, A={1,1,1}",
    },
    "omega_4": {
        "omegas": [1, 2, 3, 4],
        "As": [1, 1, 1, 1],
        "description": "Omega sweep: w={1,2,3,4}, A={1,1,1,1}",
    },
    # --- Amplitude Sweep (omega=1 fixed, adding amplitudes) ---
    "amp_2": {
        "omegas": [1, 1],
        "As": [1, 2],
        "description": "Amplitude sweep: w={1,1}, A={1,2}",
    },
    "amp_3": {
        "omegas": [1, 1, 1],
        "As": [1, 2, 3],
        "description": "Amplitude sweep: w={1,1,1}, A={1,2,3}",
    },
    "amp_4": {
        "omegas": [1, 1, 1, 1],
        "As": [1, 2, 3, 4],
        "description": "Amplitude sweep: w={1,1,1,1}, A={1,2,3,4}",
    },
    # --- Mixed Configs ---
    "mixed_4": {
        "omegas": [1, 1, 4, 4],
        "As": [1, 4, 1, 4],
        "description": "Mixed 4 (corners): w={1,1,4,4}, A={1,4,1,4}",
    },
    "mixed_2": {
        "omegas": [1, 1, 2, 2],
        "As": [1, 2, 1, 2],
        "description": "Mixed 2 (moderate): w={1,1,2,2}, A={1,2,1,2}",
    },
}

# Training settings
N_SEEDS = 5
TOTAL_STEPS = 250_000
N_TIMESTEPS = 100
LOG_EVERY = 500

# Material parameters (for data generation only - RNN has no physics)
E_INFTY = MATERIAL_PARAMS["E_infty"]
E_VAL = MATERIAL_PARAMS["E"]
ETA = MATERIAL_PARAMS["eta"]


# =============================================================================
# Helper Functions
# =============================================================================

def save_model(model, history_losses, config_name, seed, steps, n_ts, artifacts_dir):
    """Save model .eqx and history .json."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"simple_rnn__{config_name}__seed_{seed}__{steps}steps__{n_ts}ts__{timestamp}.eqx"
    model_path = artifacts_dir / model_filename

    eqx.tree_serialise_leaves(str(model_path), model)

    history_filename = model_filename.replace(".eqx", "_history.json")
    history_path = artifacts_dir / history_filename

    history_data = {
        "config_name": config_name,
        "seed": seed,
        "steps": steps,
        "n_timesteps": n_ts,
        "timestamp": timestamp,
        "losses": [float(l) for l in history_losses],
        "final_loss": float(history_losses[-1]) if len(history_losses) > 0 else None,
    }

    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

    return model_path, history_path


def generate_data(omegas, As):
    """Generate harmonic training data for given load cases."""
    return td.generate_data_harmonic(E_INFTY, E_VAL, ETA, N_TIMESTEPS, omegas, As)


# =============================================================================
# Main Training Loop
# =============================================================================

def run_all_experiments():
    """Run all Simple RNN experiments."""

    artifacts_dir = Path("artifacts") / "rnn_experiments"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    total_configs = len(EXPERIMENTS)
    total_runs = total_configs * N_SEEDS

    print("=" * 70)
    print("SIMPLE RNN EXPERIMENT TRAINING")
    print("=" * 70)
    print(f"Configurations:     {total_configs}")
    print(f"Seeds per config:   {N_SEEDS}")
    print(f"Total training runs: {total_runs}")
    print(f"Steps per run:      {TOTAL_STEPS:,}")
    print(f"Artifacts dir:      {artifacts_dir}")
    print("=" * 70)

    run_counter = 0
    global_start = time.time()

    # Save experiment overview
    overview = {
        "experiments": {k: v["description"] for k, v in EXPERIMENTS.items()},
        "n_seeds": N_SEEDS,
        "total_steps": TOTAL_STEPS,
        "n_timesteps": N_TIMESTEPS,
        "material_params": MATERIAL_PARAMS,
        "start_time": datetime.now().isoformat(),
    }
    with open(artifacts_dir / "experiment_overview.json", "w") as f:
        json.dump(overview, f, indent=2)

    for config_name, config in EXPERIMENTS.items():
        omegas = config["omegas"]
        As = config["As"]
        desc = config["description"]

        print(f"\n{'='*70}")
        print(f"CONFIG: {config_name}")
        print(f"  {desc}")
        print(f"  omegas={omegas}, As={As}")
        print(f"{'='*70}")

        # Generate training data (same for all seeds of this config)
        eps_train, _, sig_train, dts_train = generate_data(omegas, As)
        print(f"  Training data shape: eps={eps_train.shape}, sig={sig_train.shape}")

        for seed_idx in range(N_SEEDS):
            run_counter += 1
            run_start = time.time()

            print(f"\n  --- Seed {seed_idx} (Run {run_counter}/{total_runs}) ---")

            # Create reproducible key from seed
            master_key = jrandom.PRNGKey(seed_idx * 1000 + hash(config_name) % 10000)
            key_init, key_train = jrandom.split(master_key, 2)

            # Build fresh Simple RNN model
            model = tm.build(key=key_init)

            # Train for all steps at once
            model, history = klax.fit(
                model,
                ((eps_train, dts_train), sig_train),
                batch_axis=0,
                steps=TOTAL_STEPS,
                history=klax.HistoryCallback(log_every=LOG_EVERY),
                key=key_train,
            )

            # Collect losses
            all_losses = list(history.loss) if hasattr(history, 'loss') else []
            final_loss = float(all_losses[-1]) if all_losses else float('nan')

            run_time = time.time() - run_start

            # Save final model
            model_path, hist_path = save_model(
                model=model,
                history_losses=all_losses,
                config_name=config_name,
                seed=seed_idx,
                steps=TOTAL_STEPS,
                n_ts=N_TIMESTEPS,
                artifacts_dir=artifacts_dir,
            )

            print(
                f"  Seed {seed_idx} complete: "
                f"loss={final_loss:.6e}, "
                f"time={run_time:.1f}s, "
                f"saved: {model_path.name}"
            )

    total_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total models saved: {total_runs}")
    print(f"Artifacts in: {artifacts_dir}")
    print(f"{'='*70}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_all_experiments()
