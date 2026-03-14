"""
Sobolev GSM Experiment Training Script
=====================================

Trains GSM models with partial 2nd-order Sobolev penalty (eps2) for several regimes.
Each regime is trained with multiple random seeds and saved with a seed-aware filename:

    gsm__<config>__seed_<seed>__<steps>steps__<T>ts__<timestamp>.eqx

This matches the search patterns used by plots.find_latest / plot_best_state_curvature.

Usage:
    python -m tmlsm.run_sobolev_gsm
"""

import json
import time
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax.random as jrandom

from tmlsm.configs import MATERIAL_PARAMS
from tmlsm import data as td
from tmlsm import models as tm
from tmlsm import experiments as ex


# =============================================================================
# Settings
# =============================================================================

ARTIFACTS_DIR = Path("artifacts") / "gsm_sobolev_experiments"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

T = 100
TRAIN_STEPS = 250_000
SEEDS = [0, 1]               # "2 random seeds" (deterministic). Change if you want.
LR = 1e-3                    # start safe; tune if needed
LAMBDA_EPS2 = 1e-2
LOG_EVERY = 2000


# =============================================================================
# Regimes
# =============================================================================

REGIMES = {
    "sobolev_r1_corners": {
        "As": [1, 4, 4, 1],
        "omegas": [1, 1, 4, 4],
        "description": "Corners: (A,w) = (1,1),(4,1),(4,4),(1,4)",
    },
    "sobolev_r2_omega_sweep": {
        "As": [1, 1, 1, 1],
        "omegas": [1, 2, 3, 4],
        "description": "Omega sweep @ A=1",
    },
    "sobolev_r3_amp_sweep": {
        "As": [1, 2, 3, 4],
        "omegas": [1, 1, 1, 1],
        "description": "Amplitude sweep @ w=1",
    },
    "sobolev_r4_custom": {
        "As": [1, 2, 4, 6],
        "omegas": [0.5, 1, 2, 5],
        "description": "Custom: As=[1,2,4,6], w=[0.5,1,2,5]",
    },
}


# =============================================================================
# Helpers
# =============================================================================

def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_sobolev_checkpoint(model, config_name: str, seed: int, steps: int, n_ts: int, meta: dict):
    ts = _timestamp()

    model_filename = f"gsm__{config_name}__seed_{seed}__{steps}steps__{n_ts}ts__{ts}.eqx"
    model_path = ARTIFACTS_DIR / model_filename
    eqx.tree_serialise_leaves(str(model_path), model)

    meta_filename = model_filename.replace(".eqx", "_meta.json")
    meta_path = ARTIFACTS_DIR / meta_filename
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[saved] {model_path}")
    return model_path


def train_one_regime(config_name: str, As, omegas, seed: int):
    E_inf = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    # ---- data ----
    eps_train, _, sig_train, dts_train = td.generate_data_harmonic(E_inf, E, eta, T, omegas, As)

    # ---- init model ----
    key_model = jrandom.PRNGKey(seed)
    model = tm.build_gsm(key=key_model, g=1.0 / eta)

    # ---- train ----
    key_train = jrandom.PRNGKey(10_000 + seed)
    t0 = time.time()
    model_trained, train_time, final_loss = ex.train_gsm_sobolev_eps2(
        model,
        (eps_train, dts_train),
        sig_train,
        train_steps=TRAIN_STEPS,
        key=key_train,
        lr=LR,
        lambda_eps2=LAMBDA_EPS2,
        log_every=LOG_EVERY,
    )
    wall = time.time() - t0

    meta = {
        "config_name": config_name,
        "seed": seed,
        "train_steps": TRAIN_STEPS,
        "n_timesteps": T,
        "As": list(map(float, As)),
        "omegas": list(map(float, omegas)),
        "lr": LR,
        "lambda_eps2": LAMBDA_EPS2,
        "log_every": LOG_EVERY,
        "final_loss": float(final_loss),
        "train_time_returned_s": float(train_time),
        "wall_time_s": float(wall),
        "timestamp": _timestamp(),
    }

    save_sobolev_checkpoint(model_trained, config_name=config_name, seed=seed, steps=TRAIN_STEPS, n_ts=T, meta=meta)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=== Sobolev GSM batch run ===")
    print(f"Artifacts dir: {ARTIFACTS_DIR.resolve()}")
    print(f"T={T}, steps={TRAIN_STEPS}, seeds={SEEDS}, lr={LR}, lambda_eps2={LAMBDA_EPS2}")
    print("Regimes:")
    for name, cfg in REGIMES.items():
        print(f"  - {name}: {cfg['description']}")

    for regime_name, cfg in REGIMES.items():
        As = cfg["As"]
        omegas = cfg["omegas"]

        for seed in SEEDS:
            print(f"\n--- Training {regime_name} | seed={seed} ---")
            train_one_regime(regime_name, As=As, omegas=omegas, seed=seed)

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
