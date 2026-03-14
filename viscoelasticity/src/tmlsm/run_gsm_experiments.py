"""
GSM Experiment Training Script
==============================
Trains GSM models across multiple configurations with checkpointing.

Experiment groups:
  1. Omega sweep:   increasing number of load cases varying omega (A=1 fixed)
  2. Amplitude sweep: increasing number of load cases varying A (omega=1 fixed)
  3. Mixed configs:  2 mixed load case sets (corners + moderate)

Each config is trained with 5 different seeds.
Checkpoints saved every 50k steps (50k, 100k, 150k, 200k, 250k).

Usage:
    cd viscoelasticity
    python -m tmlsm.run_gsm_experiments
"""

import json
import time
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
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
# Experiment Configurations
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
    # NOTE: amp_1 is identical to omega_1, so we skip it and reuse omega_1 results
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
CHECKPOINT_EVERY = 50_000
N_TIMESTEPS = 100
LOG_EVERY = 500

# Material parameters
E_INFTY = MATERIAL_PARAMS["E_infty"]
E_VAL = MATERIAL_PARAMS["E"]
ETA = MATERIAL_PARAMS["eta"]
G = 1.0 / ETA


# =============================================================================
# Helper Functions
# =============================================================================

def save_checkpoint(model, history_losses, config_name, seed, steps, n_ts, artifacts_dir):
    """Save model .eqx and history .json for a checkpoint."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model filename (matches existing naming schema)
    model_filename = f"gsm__{config_name}__seed_{seed}__{steps}steps__{n_ts}ts__{timestamp}.eqx"
    model_path = artifacts_dir / model_filename

    # Save model weights
    eqx.tree_serialise_leaves(str(model_path), model)

    # Save history as JSON
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
    """Run all GSM experiments with checkpointing."""

    # Setup artifacts directory
    artifacts_dir = Path("artifacts") / "gsm_experiments"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Count total runs
    total_configs = len(EXPERIMENTS)
    total_runs = total_configs * N_SEEDS
    checkpoints_per_run = TOTAL_STEPS // CHECKPOINT_EVERY
    total_checkpoints = total_runs * checkpoints_per_run

    print("=" * 70)
    print("GSM EXPERIMENT TRAINING")
    print("=" * 70)
    print(f"Configurations:     {total_configs}")
    print(f"Seeds per config:   {N_SEEDS}")
    print(f"Total training runs: {total_runs}")
    print(f"Steps per run:      {TOTAL_STEPS:,}")
    print(f"Checkpoint every:   {CHECKPOINT_EVERY:,} steps")
    print(f"Checkpoints per run: {checkpoints_per_run}")
    print(f"Total checkpoints:  {total_checkpoints}")
    print(f"Artifacts dir:      {artifacts_dir}")
    print("=" * 70)

    run_counter = 0
    global_start = time.time()

    # Save experiment overview
    overview = {
        "experiments": {k: v["description"] for k, v in EXPERIMENTS.items()},
        "n_seeds": N_SEEDS,
        "total_steps": TOTAL_STEPS,
        "checkpoint_every": CHECKPOINT_EVERY,
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

            # Build fresh GSM model
            model = tm.build_gsm(key=key_init, g=G)

            # Collect all losses across checkpoints
            all_losses = []

            # Train in checkpoint intervals
            for cp_idx in range(checkpoints_per_run):
                cp_steps = CHECKPOINT_EVERY
                current_total = (cp_idx + 1) * CHECKPOINT_EVERY

                cp_start = time.time()

                # Split key for this training segment
                key_train, key_segment = jrandom.split(key_train)

                # Train for checkpoint_every steps
                model, history = klax.fit(
                    model,
                    ((eps_train, dts_train), sig_train),
                    batch_axis=0,
                    steps=cp_steps,
                    history=klax.HistoryCallback(log_every=LOG_EVERY),
                    key=key_segment,
                )

                # Collect losses from this segment
                segment_losses = list(history.loss) if hasattr(history, 'loss') else []
                all_losses.extend(segment_losses)

                cp_time = time.time() - cp_start
                final_loss = float(segment_losses[-1]) if segment_losses else float('nan')

                # Save checkpoint
                model_path, hist_path = save_checkpoint(
                    model=model,
                    history_losses=all_losses,
                    config_name=config_name,
                    seed=seed_idx,
                    steps=current_total,
                    n_ts=N_TIMESTEPS,
                    artifacts_dir=artifacts_dir,
                )

                print(
                    f"    Checkpoint {current_total // 1000}k: "
                    f"loss={final_loss:.6e}, "
                    f"time={cp_time:.1f}s, "
                    f"saved: {model_path.name}"
                )

            run_time = time.time() - run_start
            print(f"  Seed {seed_idx} complete in {run_time:.1f}s")

    total_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total checkpoints saved: {total_checkpoints}")
    print(f"Artifacts in: {artifacts_dir}")
    print(f"{'='*70}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run_all_experiments()
