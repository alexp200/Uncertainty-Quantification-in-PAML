"""Plotting utilities."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
import numpy as np
import jax.numpy as jnp

if TYPE_CHECKING:
    from .experiments import ExperimentResult

from . import data as td
from . import models as tm
from . import storage
from . import evaluation as ev
from .configs import MATERIAL_PARAMS
import equinox as eqx
import jax
import jax.random as jrandom
import klax
from matplotlib.colors import LogNorm, SymLogNorm


# Colors for loadcases
LOADCASE_COLORS = np.array(
    [
        [(194 / 255, 76 / 255, 76 / 255)],
        [(246 / 255, 163 / 255, 21 / 255)],
        [(67 / 255, 83 / 255, 132 / 255)],
        [(22 / 255, 164 / 255, 138 / 255)],
        [(104 / 255, 143 / 255, 198 / 255)],
    ]
)

# Colors for models
MODEL_COLORS = {
    "simple_rnn": "#c24c4c",  # red
    "maxwell": "#436384",  # blue
    "maxwell_nn": "#16a48a",  # teal
    "gsm": "#f6a315",  # orange
}

MODEL_LABELS = {
    "simple_rnn": "Simple RNN",
    "maxwell": "Maxwell (analytical)",
    "maxwell_nn": "Maxwell + NN",
    "gsm": "GSM",
}

# Keep old name for backwards compatibility
colors = LOADCASE_COLORS


def plot_stress_hysteresis(
    eps, sig, omegas, As,
    figsize=(12, 5),
    fontsize_title=18,
    fontsize_label=16,
    fontsize_tick=14,
    fontsize_legend=14,
):
    """Plot stress over time and hysteresis (ε vs σ) — presentation-ready.

    Args:
        eps: (N, T) strain array
        sig: (N, T) stress array
        omegas: list of frequencies
        As: list of amplitudes
    """
    n = len(eps[0])
    ns = np.linspace(0, 2 * np.pi, n)

    fig, (ax_time, ax_hyst) = plt.subplots(1, 2, figsize=figsize)

    for i in range(len(eps)):
        label = f"$\\omega={omegas[i]:.0f},\\; A={As[i]:.0f}$"
        ax_time.plot(ns, sig[i], label=label, color=colors[i], linewidth=2)
        ax_hyst.plot(eps[i], sig[i], color=colors[i], linewidth=2, label=label)

    # Left: σ(t)
    ax_time.set_xlim([0, 2 * np.pi])
    ax_time.set_xlabel("time $t$", fontsize=fontsize_label)
    ax_time.set_ylabel(r"stress $\sigma$", fontsize=fontsize_label)
    ax_time.tick_params(labelsize=fontsize_tick)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=fontsize_legend)

    # Right: Hysteresis
    ax_hyst.set_xlabel(r"strain $\varepsilon$", fontsize=fontsize_label)
    ax_hyst.set_ylabel(r"stress $\sigma$", fontsize=fontsize_label)
    ax_hyst.tick_params(labelsize=fontsize_tick)
    ax_hyst.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_data(eps, eps_dot, sig, omegas, As):
    n = len(eps[0])
    ns = np.linspace(0, 2 * np.pi, n)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Data")

    ax = axs[0, 0]
    for i in range(len(eps)):
        ax.plot(
            ns,
            sig[i],
            label="$\\omega$: %.2f, $A$: %.2f" % (omegas[i], As[i]),
            color=colors[i],
            linestyle="--",
        )
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylabel("stress $\\sigma$")
    ax.set_xlabel("time $t$")
    ax.legend()

    ax = axs[0, 1]
    for i in range(len(eps)):
        ax.plot(eps[i], sig[i], color=colors[i], linestyle="--")
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("stress $\\sigma$")

    ax = axs[1, 0]
    for i in range(len(eps)):
        ax.plot(ns, eps[i], color=colors[i], linestyle="--")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("strain $\\varepsilon$")

    ax = axs[1, 1]
    for i in range(len(eps)):
        ax.plot(ns, eps_dot[i], color=colors[i], linestyle="--")
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"strain rate $\.{\varepsilon}$")

    fig.tight_layout()
    plt.show()


def plot_model_pred(eps, sig, sig_m, omegas, As, title=None):
    n = len(eps[0])
    ns = np.linspace(0, 2 * np.pi, n)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    if title:
        fig.suptitle(title, fontsize=11)
    else:
        fig.suptitle("Data: dashed line, model prediction: continuous line")

    ax = axs[0]
    for i in range(len(eps)):
        ax.plot(
            ns,
            sig[i],
            label="$\\omega$: %.2f, $A$: %.2f" % (omegas[i], As[i]),
            linestyle="--",
            color=colors[i],
        )
        ax.plot(ns, sig_m[i], color=colors[i])
    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylabel("stress $\\sigma$")
    ax.set_xlabel("time $t$")
    ax.legend()

    ax = axs[1]
    for i in range(len(eps)):
        ax.plot(eps[i], sig[i], linestyle="--", color=colors[i])
        ax.plot(eps[i], sig_m[i], color=colors[i])
    ax.set_xlabel("strain $\\varepsilon$")
    ax.set_ylabel("stress $\\sigma$")

    fig.tight_layout()
    plt.show()


# =============================================================================
# Comparison Plots for Experiments
# =============================================================================


def plot_model_comparison(
    result: ExperimentResult,
    test_type: str = "harmonic",
    figsize: tuple[int, int] = (14, 5),
) -> None:
    """Plot all models side by side for an experiment.

    Args:
        result: ExperimentResult from run_experiment()
        test_type: "harmonic" or "relaxation"
        figsize: Figure size
    """
    config = result.config
    n_models = len(result.model_results)
    loadcases = config.test_loadcases

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    fig.suptitle(
        f"{config.name}: {config.description}\n({test_type} test)",
        fontsize=12,
    )

    # Get test data for plotting
    from . import data as td
    from .configs import MATERIAL_PARAMS

    omegas = [lc[1] for lc in loadcases]
    As = [lc[0] for lc in loadcases]

    if test_type == "harmonic":
        eps, _, sig, _ = td.generate_data_harmonic(
            MATERIAL_PARAMS["E_infty"],
            MATERIAL_PARAMS["E"],
            MATERIAL_PARAMS["eta"],
            config.n_timesteps,
            omegas,
            As,
        )
    else:
        eps, _, sig, _ = td.generate_data_relaxation(
            MATERIAL_PARAMS["E_infty"],
            MATERIAL_PARAMS["E"],
            MATERIAL_PARAMS["eta"],
            config.n_timesteps,
            omegas,
            As,
        )

    n_points = len(eps[0])
    ts = np.linspace(0, 2 * np.pi, n_points)

    for ax_idx, (model_type, model_result) in enumerate(result.model_results.items()):
        ax = axes[ax_idx]
        sig_pred = model_result.predictions.get(test_type)

        if sig_pred is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Plot each loadcase
        for i, (A, omega) in enumerate(loadcases):
            color = LOADCASE_COLORS[i % len(LOADCASE_COLORS)].flatten()
            ax.plot(ts, sig[i], "--", color=color, alpha=0.7, linewidth=1.5)
            ax.plot(
                ts,
                sig_pred[i],
                "-",
                color=color,
                linewidth=2,
                label=f"A={A}, ω={omega}",
            )

        # Get metrics for title
        metrics = model_result.harmonic_metrics if test_type == "harmonic" else model_result.relaxation_metrics
        avg_rmse = np.mean([m["rmse"] for m in metrics.values()])
        avg_r2 = np.mean([m["r_squared"] for m in metrics.values()])

        ax.set_title(f"{MODEL_LABELS.get(model_type, model_type)}\nRMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")
        ax.set_xlabel("time $t$")
        ax.set_ylabel("stress $\\sigma$")
        ax.set_xlim([0, 2 * np.pi])

        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_metrics_comparison(
    results: dict[str, ExperimentResult],
    metric: str = "rmse",
    test_type: str = "harmonic",
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """Bar chart comparing a metric across experiments and models.

    Args:
        results: Dict of experiment_name -> ExperimentResult
        metric: Which metric to plot ("rmse", "r_squared", "mae", etc.)
        test_type: "harmonic" or "relaxation"
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    experiment_names = list(results.keys())
    n_experiments = len(experiment_names)

    # Collect all model types across experiments
    all_models = set()
    for result in results.values():
        all_models.update(result.model_results.keys())
    all_models = sorted(all_models)
    n_models = len(all_models)

    # Bar positions
    bar_width = 0.8 / n_models
    x = np.arange(n_experiments)

    for i, model_type in enumerate(all_models):
        values = []
        for exp_name in experiment_names:
            result = results[exp_name]
            if model_type in result.model_results:
                model_result = result.model_results[model_type]
                metrics_dict = (
                    model_result.harmonic_metrics
                    if test_type == "harmonic"
                    else model_result.relaxation_metrics
                )
                # Average across loadcases
                avg_value = np.mean([m[metric] for m in metrics_dict.values()])
                values.append(avg_value)
            else:
                values.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=MODEL_LABELS.get(model_type, model_type),
            color=MODEL_COLORS.get(model_type, "gray"),
        )

    ax.set_xlabel("Experiment")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Comparison ({test_type})")
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_metrics_heatmap(
    result: ExperimentResult,
    metric: str = "rmse",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Heatmap of metric values: models x loadcases.

    Args:
        result: ExperimentResult from run_experiment()
        metric: Which metric to plot
        figsize: Figure size
    """
    models = list(result.model_results.keys())
    loadcases = list(result.config.test_loadcases)
    loadcase_strs = [f"A={A},w={w}" for A, w in loadcases]

    # Build matrix
    data_h = np.zeros((len(models), len(loadcases)))
    data_r = np.zeros((len(models), len(loadcases)))

    for i, model_type in enumerate(models):
        model_result = result.model_results[model_type]
        for j, lc_str in enumerate(loadcase_strs):
            if lc_str in model_result.harmonic_metrics:
                data_h[i, j] = model_result.harmonic_metrics[lc_str][metric]
            if lc_str in model_result.relaxation_metrics:
                data_r[i, j] = model_result.relaxation_metrics[lc_str][metric]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, data, title in zip(axes, [data_h, data_r], ["Harmonic", "Relaxation"]):
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(loadcase_strs)))
        ax.set_xticklabels(loadcase_strs, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in models])
        ax.set_title(f"{title} - {metric.upper()}")

        # Add text annotations
        for i in range(len(models)):
            for j in range(len(loadcase_strs)):
                text = ax.text(
                    j, i, f"{data[i, j]:.4f}",
                    ha="center", va="center", color="black", fontsize=9
                )

        fig.colorbar(im, ax=ax)

    fig.suptitle(f"{result.config.name}: {result.config.description}")
    fig.tight_layout()
    plt.show()


def print_results_table(
    results: dict[str, ExperimentResult],
    metric: str = "rmse",
    test_type: str = "harmonic",
) -> None:
    """Print a formatted table of results.

    Args:
        results: Dict of experiment_name -> ExperimentResult
        metric: Which metric to show
        test_type: "harmonic" or "relaxation"
    """
    # Collect all models
    all_models = set()
    for result in results.values():
        all_models.update(result.model_results.keys())
    all_models = sorted(all_models)

    # Header
    header = f"{'Experiment':<25}" + "".join([f"{MODEL_LABELS.get(m, m):<18}" for m in all_models])
    print("\n" + "=" * len(header))
    print(f"{metric.upper()} ({test_type})")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Rows
    for exp_name, result in results.items():
        row = f"{exp_name:<25}"
        for model_type in all_models:
            if model_type in result.model_results:
                model_result = result.model_results[model_type]
                metrics_dict = (
                    model_result.harmonic_metrics
                    if test_type == "harmonic"
                    else model_result.relaxation_metrics
                )
                avg_value = np.mean([m[metric] for m in metrics_dict.values()])
                row += f"{avg_value:<18.4f}"
            else:
                row += f"{'-':<18}"
        print(row)

    print("=" * len(header) + "\n")

# =============================================================================
# Data Generation Helper
# =============================================================================

def _generate_test_data(n_timesteps, omegas, As, test_type="harmonic", noise_std_rel=0.0):
    """Generate test data, optionally with noisy eps.

    Returns: (eps, sig_true, dts)
    """
    mp = MATERIAL_PARAMS
    if test_type == "harmonic":
        if noise_std_rel > 0:
            eps, _, sig, dts = td.generate_data_harmonic_noisy_eps(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False)
        else:
            eps, _, sig, dts = td.generate_data_harmonic(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As)
    else:  # relaxation
        if noise_std_rel > 0:
            eps, _, sig, dts = td.generate_data_relaxation_noisy_eps(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False)
        else:
            eps, _, sig, dts = td.generate_data_relaxation(
                mp["E_infty"], mp["E"], mp["eta"], n_timesteps, omegas, As)
    return eps, sig, dts


# =============================================================================
# Single Model Plotting
# =============================================================================

def find_latest(pattern: str, steps=None, search_dirs=None, n_timesteps=None) -> str:
    """Find the latest .eqx file matching a pattern.

    Args:
        pattern: Substring to match in filenames (e.g. "omega_3", "gsm__amp_4__seed_0")
        steps: Optional filter for training steps (e.g. 50000, 100000, 250000)
        search_dirs: List of directories to search. Default: all known artifact dirs
        n_timesteps: Optional filter for number of timesteps (e.g. 50, 100, 200, 400)

    Returns:
        Path to the latest matching .eqx file (sorted by timestamp in filename)
    """
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ["artifacts", "artifacts/gsm_experiments", "artifacts/rnn_experiments", "artifacts/maxwell_nn_experiments"]

    matches = []
    for d in search_dirs:
        p = Path(d)
        if p.exists():
            matches.extend([f for f in p.glob("*.eqx") if pattern in f.name])

    # Filter nach steps falls angegeben
    if steps is not None:
        steps_str = f"{steps}steps"
        matches = [f for f in matches if steps_str in f.name]

    # Filter nach n_timesteps falls angegeben
    if n_timesteps is not None:
        ts_str = f"__{n_timesteps}ts__"
        matches = [f for f in matches if ts_str in f.name]

    if not matches:
        info = f"Pattern='{pattern}'"
        if steps is not None:
            info += f", steps={steps}"
        if n_timesteps is not None:
            info += f", n_timesteps={n_timesteps}"
        print(f"Keine .eqx Dateien gefunden mit {info}")
        print(f"Durchsuchte Ordner: {search_dirs}")
        return None

    # Sortiere nach Dateiname (Timestamp ist am Ende → lexikographisch sortierbar)
    matches.sort(key=lambda f: f.name)
    latest = matches[-1]
    print(f"Gefunden: {latest}")
    return str(latest)


def plot_latest(pattern: str, steps=None, test_loadcases=None, search_dirs=None,
                seeds=None, noise_std_rel=0.0, n_timesteps=None):
    """Find the latest model matching a pattern and plot it.

    Args:
        pattern: Substring to match (e.g. "omega_3", "amp_4__seed_0", "maxwell_nn")
        steps: Optional filter for training steps (e.g. 50000, 150000, 250000)
        test_loadcases: List of (A, omega) tuples to test on. Default: [(1,1), (1,2), (1,3)]
        search_dirs: Optional list of directories to search
        seeds: List of seed indices to plot overlaid in one figure (e.g. [0,2,4])
               Use seeds=[0,1,2,3,4] for all 5 seeds.
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
        n_timesteps: Optional filter for number of timesteps (e.g. 50, 100, 200, 400)

    Examples:
        plot_latest("omega_3__seed_0")                              # single seed
        plot_latest("omega_3__seed_0", noise_std_rel=0.02)          # with 2% noise
        plot_latest("omega_3", seeds=[0,1,2,3,4])                   # all 5 seeds overlaid
        plot_latest("gsm__mixed_4", seeds=[0,1,2], n_timesteps=50)  # timestep study
    """
    if seeds is not None:
        _plot_all_seeds(pattern, steps=steps, seeds=seeds,
                        test_loadcases=test_loadcases, search_dirs=search_dirs,
                        noise_std_rel=noise_std_rel, n_timesteps=n_timesteps)
    else:
        filename = find_latest(pattern, steps=steps, search_dirs=search_dirs,
                               n_timesteps=n_timesteps)
        if filename is not None:
            plot_saved_model(filename, test_loadcases=test_loadcases,
                            noise_std_rel=noise_std_rel)


# Best seeds per config (from visual inspection of all_seeds plots)
BEST_SEEDS_GSM = {
    "omega_1": 0,
    "omega_2": 2,
    "omega_3": 0,
    "omega_4": 1,
    "amp_2":   2,
    "amp_3":   0,
    "amp_4":   0,
    "mixed_2": 4,
    "mixed_4": 1,
    "mixed_4_sobolev_eps2": 2,
}

BEST_SEEDS_RNN = {
    "omega_1": 3,
    "omega_2": 0,
    "omega_3": 2,
    "omega_4": 1,
    "amp_2":   1,  # alle seeds schlecht
    "amp_3":   0,  # nicht sehr gut
    "amp_4":   0,
    "mixed_2": 0,
    "mixed_4": 4,
}

BEST_SEEDS_MAXWELL_NN = {
    "omega_1": 1,
    "omega_2": 2,
    "omega_3": 0,
    "omega_4": 0,
    "amp_2":   3,
    "amp_3":   3,
    "amp_4":   0,
    "mixed_2": 2,
    "mixed_4": 3,
}

BEST_SEEDS_GSM_SOBOLEV = {
    "sobolev_r1_corners":    1,
    "sobolev_r2_omega_sweep": 1,
    "sobolev_r3_amp_sweep":  1,
    "sobolev_r4_custom":     0,
}

# Best seeds for timestep study (model_type -> n_timesteps -> seed)
BEST_SEEDS_TIMESTEP_STUDY = {
    "gsm": {50: 1, 100: 2, 200: 0, 400: 1},
    "maxwell_nn": {50: 0, 100: 2, 200: 1, 400: 1},
    "simple_rnn": {50: 2, 100: 0, 200: 0, 400: 2},
}

# Default search dirs per model type
_SEARCH_DIRS = {
    "gsm":           ["artifacts", "artifacts/gsm_experiments"],
    "simple_rnn":    ["artifacts", "artifacts/rnn_experiments"],
    "maxwell_nn":    ["artifacts", "artifacts/maxwell_nn_experiments"],
    "gsm_sobolev":   ["artifacts/gsm_sobolev_experiments"],
}

def _get_best_seeds(model_type="gsm"):
    if model_type == "simple_rnn":
        return BEST_SEEDS_RNN
    elif model_type == "maxwell_nn":
        return BEST_SEEDS_MAXWELL_NN
    elif model_type == "gsm_sobolev":
        return BEST_SEEDS_GSM_SOBOLEV
    return BEST_SEEDS_GSM

def _get_search_dirs(model_type="gsm", search_dirs=None):
    if search_dirs is not None:
        return search_dirs
    return _SEARCH_DIRS.get(model_type, ["artifacts"])


def plot_best(
    configs=None,
    steps=250000,
    test_loadcases=None,
    search_dirs=None,
    noise_std_rel=0.0,
    model_type="gsm",
    n_test_timesteps=None,
    # ------------------------------------------------------------------
    # NEW PARAMS (presentation customization)
    legend_labels=None,                 # dict: {config_name: "Pretty label"}
    legend_label_fn=None,               # fn(config, seed) -> str (overrides legend_labels if given)
    ground_truth_label="Ground Truth",  # legend label for GT when only one loadcase
    title=None,                         # explicit suptitle; overrides auto title
    title_fn=None,                      # fn(context_dict) -> str (overrides title if given)
    show_harmonic=True,                 # True -> create harmonic plot
    show_relaxation=True,               # True -> create relaxation plot
    # choose which subplot(s) to show
    harmonic_panels="both",             # "left" | "right" | "both"
    relaxation_panels="both",           # "left" | "right" | "both"
    # axis labels
    axis_labels_time=("time $t$", "stress $\\sigma$"),                 # (xlabel, ylabel) for time plot
    axis_labels_epssig=("strain $\\varepsilon$", "stress $\\sigma$"),  # (xlabel, ylabel) for eps-sig plot
    # fonts
    title_fontsize=11,
    legend_fontsize=7,
    axis_label_fontsize=10,
    title_bold=False,
    gt_label=None,
    # ------------------------------------------------------------------
    # NEW: legend controls
    show_legend=True,                   # False -> disable legend completely
    legend_position="auto",             # "auto" | "below"
    legend_ncol=None,                   # columns when legend_position="below" (None -> auto)
    legend_below_y=0.02,                # vertical anchor (smaller -> lower)
    legend_bottom_margin=0.18,          # space reserved at bottom if legend below
):
    """Plot best seed of each config overlaid in one figure for comparison."""
    # ---- existing selection logic ----
    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe...")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Parse model filename robustly using storage helper (instead of manual splitting)
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    try:
        meta = storage.parse_model_filename(name_only)
        file_model_type = meta["model_type"]
        n_timesteps = int(meta["n_timesteps"])
    except Exception:
        parts = name_only.replace(".eqx", "").split("__")
        if len(parts) == 6:
            file_model_type = parts[0]
            n_timesteps = int(parts[4].replace("ts", ""))
        elif len(parts) == 5:
            file_model_type = parts[0]
            n_timesteps = int(parts[3].replace("ts", ""))
        else:
            print(f"Unbekanntes Format: {name_only}")
            return

    # Build model template
    key = jrandom.PRNGKey(0)
    if file_model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif file_model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif file_model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"]
        )
    else:
        print(f"Unbekannter Modelltyp: {file_model_type}")
        return

    # Validate panel params early (fail fast)
    valid_panels = {"left", "right", "both"}
    if harmonic_panels not in valid_panels:
        raise ValueError(f"harmonic_panels must be one of {valid_panels}, got: {harmonic_panels}")
    if relaxation_panels not in valid_panels:
        raise ValueError(f"relaxation_panels must be one of {valid_panels}, got: {relaxation_panels}")

    # Validate legend params
    valid_legend_positions = {"auto", "below"}
    if legend_position not in valid_legend_positions:
        raise ValueError(f"legend_position must be one of {valid_legend_positions}, got: {legend_position}")

    # Generate test data
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps

    eps_h = sig_h = dts_h = None
    eps_r = sig_r = dts_r = None
    if show_harmonic:
        eps_h, sig_h, dts_h = _generate_test_data(n_ts_test, omegas, As, "harmonic", noise_std_rel)
    if show_relaxation:
        eps_r, sig_r, dts_r = _generate_test_data(n_ts_test, omegas, As, "relaxation", noise_std_rel)

    # Shared plot stuff
    n_lc = len(test_loadcases)
    cmap = plt.cm.tab10

    # Build default auto title components
    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    model_label = MODEL_LABELS.get(file_model_type, file_model_type.upper())

    def _make_legend_label(config, seed):
        if legend_label_fn is not None:
            return str(legend_label_fn(config, seed))
        if isinstance(legend_labels, dict) and config in legend_labels:
            return str(legend_labels[config])
        return f"{config} (s{seed})"

    def _make_title(test_kind: str):
        ctx = {
            "model_label": model_label,
            "test_kind": test_kind,
            "test_loadcases": test_loadcases,
            "tc_str": tc_str,
            "noise_std_rel": noise_std_rel,
            "noise_str": noise_str,
            "n_ts_test": n_ts_test,
            "ts_str": ts_str,
            "steps": steps,
            "model_type": file_model_type,
        }
        if title_fn is not None:
            return str(title_fn(ctx))
        if title is not None:
            return str(title)
        return f"{model_label} Best Seeds Comparison — {test_kind} — Test: {tc_str}{noise_str}{ts_str}"

    def _plot_one(test_kind: str, eps, sig_true, dts, panels: str):
        n_pts = len(eps[0])
        ns = np.linspace(0, 2 * np.pi, n_pts)

        show_left = panels in ("left", "both")
        show_right = panels in ("right", "both")
        ncols = (1 if (show_left ^ show_right) else 2)

        figsize = (6, 5) if ncols == 1 else (12, 5)
        fig, axs = plt.subplots(1, ncols, figsize=figsize)

        if ncols == 1:
            axs = [axs]

        ax_time = axs[0] if show_left else None
        ax_epssig = (axs[-1] if show_right else None)

        fig.suptitle(
            _make_title(test_kind),
            fontsize=title_fontsize,
            fontweight="bold" if title_bold else "normal",
        )

        # ---- Ground truth ----
        for i in range(n_lc):
            if gt_label is None:
                gt_lbl = f"True (A={As[i]},ω={omegas[i]})" if n_lc > 1 else ground_truth_label
            else:
                gt_lbl = gt_label

            if ax_time is not None:
                ax_time.plot(ns, sig_true[i], linestyle=":", color="black", linewidth=2,
                             label=gt_lbl if i == 0 else None)
            if ax_epssig is not None:
                ax_epssig.plot(eps[i], sig_true[i], linestyle=":", color="black", linewidth=2)

        # ---- Predictions ----
        for idx, (config, seed, filepath) in enumerate(model_files):
            model = storage.load_model(filepath, model_template)
            model = klax.finalize(model)
            sig_pred = jax.vmap(model)((eps, dts))
            c = cmap(idx % 10)

            for i in range(n_lc):
                label = _make_legend_label(config, seed) if i == 0 else None
                if ax_time is not None:
                    ax_time.plot(ns, sig_pred[i], color=c, alpha=0.8, label=label)
                if ax_epssig is not None:
                    ax_epssig.plot(eps[i], sig_pred[i], color=c, alpha=0.8)

        # ---- Labels / styling ----
        if ax_time is not None:
            ax_time.set_xlim([0, 2 * np.pi])
            ax_time.set_xlabel(axis_labels_time[0], fontsize=axis_label_fontsize)
            ax_time.set_ylabel(axis_labels_time[1], fontsize=axis_label_fontsize)

        if ax_epssig is not None:
            ax_epssig.set_xlabel(axis_labels_epssig[0], fontsize=axis_label_fontsize)
            ax_epssig.set_ylabel(axis_labels_epssig[1], fontsize=axis_label_fontsize)

        # ---- Legend handling (NEW) ----
        if show_legend:
            if legend_position == "auto":
                # Keep your old behavior: legend on first available axis
                legend_ax = ax_time if ax_time is not None else ax_epssig
                if legend_ax is not None:
                    legend_ax.legend(fontsize=legend_fontsize, loc="best")

                fig.tight_layout()
            else:
                # Place a shared legend below the axes (figure-level legend)
                # Collect handles/labels from whichever axis exists
                src_ax = ax_time if ax_time is not None else ax_epssig
                handles, labels = ([], [])
                if src_ax is not None:
                    handles, labels = src_ax.get_legend_handles_labels()

                # If there is nothing labeled, just layout normally
                if len(handles) == 0:
                    fig.tight_layout()
                else:
                    # Choose columns automatically if not provided
                    if legend_ncol is None:
                        legend_ncol_eff = min(len(labels), 4)  # sensible default
                    else:
                        legend_ncol_eff = int(legend_ncol)

                    fig.legend(
                        handles, labels,
                        loc="lower center",
                        bbox_to_anchor=(0.5, legend_below_y),
                        ncol=legend_ncol_eff,
                        fontsize=legend_fontsize,
                        frameon=False,
                    )
                    # Reserve bottom space for the legend, then tighten remaining
                    fig.subplots_adjust(bottom=legend_bottom_margin)
                    fig.tight_layout(rect=(0, legend_bottom_margin, 1, 1))
        else:
            # No legend at all
            fig.tight_layout()

        plt.show()

    # Render requested plots
    if show_harmonic:
        _plot_one("Harmonic", eps_h, sig_h, dts_h, panels=harmonic_panels)
    if show_relaxation:
        _plot_one("Relaxation", eps_r, sig_r, dts_r, panels=relaxation_panels)





def plot_heatmaps(configs=None, steps=250000, test_omegas=None, test_As=None,
                  test_type="harmonic", log=False, normalize=False, noise_std_rel=0.0,
                  search_dirs=None, model_type="gsm", n_test_timesteps=None):
    """Plot RMSE heatmaps for each config's best seed over a grid of (A, omega) test cases.

    Args:
        configs: List of config names or None for all in BEST_SEEDS_{GSM/RNN}
        steps: Training steps filter (default: 250000)
        test_omegas: List of omega values for the grid. Default: range(1,21)
        test_As: List of A values for the grid. Default: range(1,21)
        test_type: "harmonic" or "relaxation"
        log: If True, use logarithmic color scale
        normalize: If True, use NRMSE (RMSE / std(sigma_true)) instead of RMSE
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
        search_dirs: Optional list of directories to search
        model_type: "gsm", "simple_rnn", or "maxwell_nn" (selects best seeds + search dirs)

    Examples:
        plot_heatmaps()
        plot_heatmaps(["omega_1", "omega_4", "mixed_4"])
        plot_heatmaps(log=True)
        plot_heatmaps(normalize=True, log=True)
        plot_heatmaps(noise_std_rel=0.02)
        plot_heatmaps(test_omegas=range(1,11), test_As=range(1,11))
        plot_heatmaps(model_type="simple_rnn")
        plot_heatmaps(model_type="maxwell_nn")
        plot_heatmaps(n_test_timesteps=200)
    """
    from matplotlib.colors import LogNorm

    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_omegas is None:
        test_omegas = list(range(1, 21))
    if test_As is None:
        test_As = list(range(1, 21))

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe...")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Parse n_timesteps and file_model_type from first file
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        file_model_type = parts[0]
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        file_model_type = parts[0]
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if file_model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif file_model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif file_model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {file_model_type}")
        return

    n_om = len(test_omegas)
    n_A = len(test_As)

    # Use n_test_timesteps if provided, else n_timesteps from model file
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps

    # Train info for annotations
    _TRAIN_INFO = {
        "omega_1": "ω={1}",
        "omega_2": "ω={1,2}",
        "omega_3": "ω={1,2,3}",
        "omega_4": "ω={1,2,3,4}",
        "amp_2":   "A={1,2}",
        "amp_3":   "A={1,2,3}",
        "amp_4":   "A={1,2,3,4}",
        "mixed_4": "(ω,A)∈{1,4}²",
        "mixed_2": "(ω,A)∈{1,2}²",
    }

    # Descriptive titles for presentation (cartesian product notation)
    _PLOT_TITLES = {
        "omega_1": "(A,ω) ∈ {1} × {1}",
        "omega_2": "(A,ω) ∈ {1} × {1,2}",
        "omega_3": "(A,ω) ∈ {1} × {1,2,3}",
        "omega_4": "(A,ω) ∈ {1} × {1,2,3,4}",
        "amp_2":   "(A,ω) ∈ {1,2} × {1}",
        "amp_3":   "(A,ω) ∈ {1,2,3} × {1}",
        "amp_4":   "(A,ω) ∈ {1,2,3,4} × {1}",
        "mixed_4": "(A,ω) ∈ {1,4} × {1,4}",
        "mixed_2": "(A,ω) ∈ {1,2} × {1,2}",
        "sobolev_r1_corners":     "(A,ω) ∈ {1,4} × {1,4}",
        "sobolev_r2_omega_sweep": "(A,ω) ∈ {1} × {1,2,3,4}",
        "sobolev_r3_amp_sweep":   "(A,ω) ∈ {1,2,3,4} × {1}",
        "sobolev_r4_custom":      "(A,ω) ∈ {1,2,4,6} × {0.5,1,2,5}",
    }

    # Determine grid layout
    n_models = len(model_files)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols

    # Use GridSpec with extra column for colorbar
    from matplotlib.gridspec import GridSpec
    cell_size = 5
    fig = plt.figure(figsize=(cell_size * n_cols + 2, cell_size * n_rows + 1))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.05], wspace=0.3, hspace=0.35)
    metric_name = "NRMSE" if normalize else "RMSE"
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    model_label = MODEL_LABELS.get(file_model_type, file_model_type.upper())
    # suptitle removed for cleaner presentation slides

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_ax = fig.add_subplot(gs[:, -1])

    # First pass: compute all RMSE values
    print("Berechne RMSE-Werte...")
    rmse_per_model = []
    for config, seed, filepath in model_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)

        rmse_grid = np.zeros((n_A, n_om))
        for i, A in enumerate(test_As):
            for j, omega in enumerate(test_omegas):
                eps, sig, dts = _generate_test_data(
                    n_ts_test, [omega], [A], test_type, noise_std_rel)
                sig_pred = jax.vmap(model)((eps, dts))
                rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
                if normalize:
                    sig_std = float(np.std(np.array(sig)))
                    rmse = rmse / sig_std if sig_std > 1e-10 else rmse
                rmse_grid[i, j] = rmse

        rmse_per_model.append(rmse_grid)
        print(f"  {config} (seed {seed}): done")

    # Global color range
    all_vals = np.concatenate([r.ravel() for r in rmse_per_model])
    if log:
        log_floor = 1e-4
        norm = LogNorm(vmin=1e-4, vmax=4e-1)
    else:
        norm = None
        vmin = 0
        vmax = all_vals.max()

    # Second pass: plot
    im = None
    for idx, (config, seed, filepath) in enumerate(model_files):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        rmse_grid = rmse_per_model[idx]

        if log:
            plot_data = np.where(rmse_grid > 0, rmse_grid, log_floor)
            im = ax.imshow(plot_data, origin="lower", aspect="equal",
                           norm=norm, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])
        else:
            im = ax.imshow(rmse_grid, origin="lower", aspect="equal",
                           vmin=vmin, vmax=vmax, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])

        # Annotate cells with RMSE values (only if grid is small enough to read)
        if n_om <= 8 and n_A <= 8:
            thresh = norm(rmse_grid).data if log else rmse_grid / vmax if vmax > 0 else rmse_grid
            for i in range(n_A):
                for j in range(n_om):
                    val = rmse_grid[i, j]
                    t = thresh[i, j] if hasattr(thresh, '__getitem__') else 0.5
                    color = "white" if t > 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color)

        # Only show ticks at integer values (skip 0.5 steps)
        int_xticks = [i for i, v in enumerate(test_omegas) if v == int(v)]
        int_xlabels = [int(test_omegas[i]) for i in int_xticks]
        int_yticks = [i for i, v in enumerate(test_As) if v == int(v)]
        int_ylabels = [int(test_As[i]) for i in int_yticks]

        ax.set_xticks(int_xticks)
        ax.set_xticklabels(int_xlabels, fontsize=11)
        ax.set_yticks(int_yticks)
        ax.set_yticklabels(int_ylabels, fontsize=11)
        ax.set_xlabel("ω (test)", fontsize=13)
        ax.set_ylabel("A (test)", fontsize=13)

        plot_title = _PLOT_TITLES.get(config, config)
        ax.set_title(plot_title, fontsize=14, fontweight="bold")

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    # Colorbar in its own axis
    label = f"{metric_name} (log)" if log else metric_name
    cbar = fig.colorbar(im, cax=cbar_ax, label=label)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(label, fontsize=13)
    plt.show()


def plot_heatmaps_timestep_study(model_type, timestep_values=None, steps=100000,
                                  test_omegas=None, test_As=None,
                                  test_type="harmonic", log=True, normalize=True,
                                  noise_std_rel=0.0, search_dirs=None):
    """Plot RMSE heatmaps for a single model type across different timestep values.

    One heatmap per n_timesteps value, using best seed from BEST_SEEDS_TIMESTEP_STUDY.

    Args:
        model_type: "gsm", "maxwell_nn", or "simple_rnn"
        timestep_values: List of n_timesteps to show. Default: [50, 100, 200, 400]
        steps: Training steps filter (default: 100000)
        test_omegas: List of omega values for the grid
        test_As: List of A values for the grid
        test_type: "harmonic" or "relaxation"
        log: If True, use logarithmic color scale
        normalize: If True, use NRMSE
        noise_std_rel: Relative noise std on eps. Default: 0 (clean)
        search_dirs: Optional list of directories to search

    Examples:
        plot_heatmaps_timestep_study("gsm")
        plot_heatmaps_timestep_study("maxwell_nn", log=True)
        plot_heatmaps_timestep_study("simple_rnn", test_omegas=list(np.arange(0.5,6.5,0.5)))
    """
    from matplotlib.colors import LogNorm
    from matplotlib.gridspec import GridSpec

    if timestep_values is None:
        timestep_values = [50, 100, 200, 400]
    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]
    if test_omegas is None:
        test_omegas = list(range(1, 21))
    if test_As is None:
        test_As = list(range(1, 21))

    ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})
    if not ts_seeds:
        print(f"Keine best seeds für model_type='{model_type}' in BEST_SEEDS_TIMESTEP_STUDY")
        return

    # Collect model files
    model_files = []
    for n_ts in timestep_values:
        seed = ts_seeds.get(n_ts)
        if seed is None:
            print(f"Kein best seed für {model_type}, n_ts={n_ts}, überspringe...")
            continue
        pattern = f"{model_type}__mixed_4__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs, n_timesteps=n_ts)
        if f is not None:
            model_files.append((n_ts, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    n_om = len(test_omegas)
    n_A = len(test_As)

    # First pass: compute RMSE values
    print("Berechne RMSE-Werte...")
    rmse_per_model = []
    for n_ts, seed, filepath in model_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)

        rmse_grid = np.zeros((n_A, n_om))
        for i, A in enumerate(test_As):
            for j, omega in enumerate(test_omegas):
                eps, sig, dts = _generate_test_data(
                    n_ts, [omega], [A], test_type, noise_std_rel)
                sig_pred = jax.vmap(model)((eps, dts))
                rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
                if normalize:
                    sig_std = float(np.std(np.array(sig)))
                    rmse = rmse / sig_std if sig_std > 1e-10 else rmse
                rmse_grid[i, j] = rmse

        rmse_per_model.append(rmse_grid)
        print(f"  {model_type} n_ts={n_ts} (seed {seed}): done")

    # Global color range
    metric_name = "NRMSE" if normalize else "RMSE"
    all_vals = np.concatenate([r.ravel() for r in rmse_per_model])
    if log:
        log_floor = 1e-4
        norm = LogNorm(vmin=1e-4, vmax=1e0)
    else:
        norm = None
        vmin = 0
        vmax = all_vals.max()

    # Layout
    n_models = len(model_files)
    n_cols = min(n_models, 4)
    n_rows = (n_models + n_cols - 1) // n_cols

    cell_size = 5
    fig = plt.figure(figsize=(cell_size * n_cols + 2, cell_size * n_rows + 1))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.05], wspace=0.3, hspace=0.35)

    model_label = MODEL_LABELS.get(model_type, model_type.upper())
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    fig.suptitle(f"{model_label} — Timestep Study — {metric_name} ({test_type}){noise_str}",
                 fontsize=13, y=0.98)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    cbar_ax = fig.add_subplot(gs[:, -1])

    # Second pass: plot
    im = None
    for idx, (n_ts, seed, filepath) in enumerate(model_files):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]
        rmse_grid = rmse_per_model[idx]

        if log:
            plot_data = np.where(rmse_grid > 0, rmse_grid, log_floor)
            im = ax.imshow(plot_data, origin="lower", aspect="equal",
                           norm=norm, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])
        else:
            im = ax.imshow(rmse_grid, origin="lower", aspect="equal",
                           vmin=vmin, vmax=vmax, cmap="RdYlGn_r",
                           extent=[-0.5, n_om - 0.5, -0.5, n_A - 0.5])

        if n_om <= 8 and n_A <= 8:
            thresh = norm(rmse_grid).data if log else rmse_grid / vmax if vmax > 0 else rmse_grid
            for i in range(n_A):
                for j in range(n_om):
                    val = rmse_grid[i, j]
                    t = thresh[i, j] if hasattr(thresh, '__getitem__') else 0.5
                    color = "white" if t > 0.6 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color=color)

        # Only show ticks at integer values (skip 0.5 steps)
        int_xticks = [i for i, v in enumerate(test_omegas) if v == int(v)]
        int_xlabels = [int(test_omegas[i]) for i in int_xticks]
        int_yticks = [i for i, v in enumerate(test_As) if v == int(v)]
        int_ylabels = [int(test_As[i]) for i in int_yticks]

        ax.set_xticks(int_xticks)
        ax.set_xticklabels(int_xlabels, fontsize=11)
        ax.set_yticks(int_yticks)
        ax.set_yticklabels(int_ylabels, fontsize=11)
        ax.set_xlabel("ω (test)", fontsize=13)
        ax.set_ylabel("A (test)", fontsize=13)
        ax.set_title(f"n_timesteps={n_ts}\nseed {seed}", fontsize=9)

    # Hide unused axes
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis("off")

    label = f"{metric_name} (log)" if log else metric_name
    cbar = fig.colorbar(im, cax=cbar_ax, label=label)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(label, fontsize=13)
    plt.show()


# =============================================================================
# Timestep Study: Aggregated Metrics (Compute + Save + Plot)
# =============================================================================

def compute_timestep_study_metrics(
    timestep_values=None,
    model_types=None,
    steps=100000,
    search_dirs=None,
    save_path="artifacts/timestep_study/metrics.json",
):
    """Compute aggregated NRMSE metrics for the timestep study and save to JSON.

    For each (model_type, n_timesteps) combination, computes NRMSE over two regions:
    - Interpolation: (omega, A) in [1, 4] x [1, 4] with 0.5 steps, excluding 4 training corners
      -> 7x7 - 4 = 45 points
    - Extrapolation: (omega, A) in [0.5, 6] x [0.5, 6] with 0.5 steps, minus the interpolation region
      -> 12x12 - 7x7 = 95 points

    Metrics computed per region: mean, median, geometric_mean, min, max of per-point NRMSE values.

    Args:
        timestep_values: List of n_timesteps. Default: [50, 100, 200, 400]
        model_types: List of model types. Default: ["gsm", "maxwell_nn", "simple_rnn"]
        steps: Training steps filter. Default: 100000
        search_dirs: Directories to search. Default: ["artifacts/timestep_study"]
        save_path: Path for the output JSON file

    Returns:
        dict with all computed metrics (also saved to JSON)
    """
    from scipy.stats import gmean

    if timestep_values is None:
        timestep_values = [50, 100, 200, 400]
    if model_types is None:
        model_types = ["gsm", "maxwell_nn", "simple_rnn"]
    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]

    # Define grids
    interp_vals = [v / 2 for v in range(2, 9)]   # [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    extrap_vals = [v / 2 for v in range(1, 13)]   # [0.5, 1.0, ..., 5.5, 6.0]

    train_corners = {(1.0, 1.0), (1.0, 4.0), (4.0, 1.0), (4.0, 4.0)}
    interp_set = set((omega, A) for omega in interp_vals for A in interp_vals)

    interp_points = sorted(interp_set - train_corners)
    extrap_points = sorted(
        (omega, A) for omega in extrap_vals for A in extrap_vals
        if (omega, A) not in interp_set
    )

    print(f"Interpolation points: {len(interp_points)} (7x7 - 4 corners)")
    print(f"Extrapolation points: {len(extrap_points)} (12x12 - 7x7)")

    def _compute_nrmse_values(model, n_ts, points):
        """Compute per-point NRMSE for a list of (omega, A) pairs."""
        nrmse_vals = []
        for omega, A in points:
            eps, sig, dts = _generate_test_data(n_ts, [omega], [A], "harmonic", 0.0)
            sig_pred = jax.vmap(model)((eps, dts))
            rmse = float(np.sqrt(np.mean((np.array(sig_pred) - np.array(sig)) ** 2)))
            sig_std = float(np.std(np.array(sig)))
            nrmse = rmse / sig_std if sig_std > 1e-10 else rmse
            nrmse_vals.append(nrmse)
        return np.array(nrmse_vals)

    def _aggregate(vals):
        """Compute all aggregation metrics for an array of NRMSE values."""
        vals_pos = np.clip(vals, 1e-12, None)  # avoid log(0) for geometric mean
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "geometric_mean": float(gmean(vals_pos)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n_points": int(len(vals)),
        }

    # Build results
    results = {
        "description": "Aggregated NRMSE metrics for the timestep study",
        "normalization": "NRMSE = RMSE / std(sigma_true) per loadcase",
        "aggregation": "Metrics computed over per-point NRMSE values",
        "train_config": "mixed_4: (omega, A) in {1,4} x {1,4} (4 corners)",
        "interpolation_grid": {
            "omega_range": "[1.0, 4.0] step 0.5",
            "A_range": "[1.0, 4.0] step 0.5",
            "excluding": "4 training corners",
            "n_points": len(interp_points),
        },
        "extrapolation_grid": {
            "omega_range": "[0.5, 6.0] step 0.5",
            "A_range": "[0.5, 6.0] step 0.5",
            "excluding": "entire interpolation region [1,4]x[1,4]",
            "n_points": len(extrap_points),
        },
        "metrics": {},
    }

    for model_type in model_types:
        print(f"\n--- {model_type} ---")
        ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})

        # Build model template once
        key = jrandom.PRNGKey(0)
        if model_type == "gsm":
            model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
        elif model_type == "simple_rnn":
            model_template = tm.build(key=key)
        elif model_type == "maxwell_nn":
            model_template = tm.build_maxwell_nn(
                key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])

        results["metrics"][model_type] = {}

        for n_ts in timestep_values:
            seed = ts_seeds.get(n_ts)
            if seed is None:
                print(f"  n_ts={n_ts}: kein best seed, überspringe...")
                continue

            pattern = f"{model_type}__mixed_4__seed_{seed}"
            f = find_latest(pattern, steps=steps, search_dirs=search_dirs, n_timesteps=n_ts)
            if f is None:
                continue

            model = storage.load_model(f, model_template)
            model = klax.finalize(model)

            interp_nrmse = _compute_nrmse_values(model, n_ts, interp_points)
            extrap_nrmse = _compute_nrmse_values(model, n_ts, extrap_points)

            results["metrics"][model_type][str(n_ts)] = {
                "seed": seed,
                "interpolation": _aggregate(interp_nrmse),
                "extrapolation": _aggregate(extrap_nrmse),
            }
            print(f"  n_ts={n_ts} (seed {seed}): "
                  f"interp geomean={gmean(np.clip(interp_nrmse, 1e-12, None)):.4e}, "
                  f"extrap geomean={gmean(np.clip(extrap_nrmse, 1e-12, None)):.4e}")

    # Save to JSON
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f_out:
        json.dump(results, f_out, indent=2)
    print(f"\nMetrics saved to: {save_path}")

    return results


def plot_timestep_study_metrics(
    metrics=None,
    metrics_path="artifacts/timestep_study/metrics.json",
    metric="geometric_mean",
    style="scatter",
):
    """Plot aggregated NRMSE metrics vs n_timesteps for the timestep study.

    Args:
        metrics: Dict from compute_timestep_study_metrics, or None to load from file
        metrics_path: Path to metrics JSON (used if metrics is None)
        metric: Which aggregation to plot. One of:
                "geometric_mean", "mean", "median", "min", "max"
        style: "scatter" (markers only, no lines), "bar" (grouped bar chart)

    Examples:
        plot_timestep_study_metrics()                                    # scatter (default)
        plot_timestep_study_metrics(style="bar")                         # grouped bars
        plot_timestep_study_metrics(metric="median", style="scatter")
    """
    if metrics is None:
        with open(metrics_path, "r") as f_in:
            metrics = json.load(f_in)

    model_types = list(metrics["metrics"].keys())
    colors = {
        "gsm": MODEL_COLORS.get("gsm", "#1f77b4"),
        "maxwell_nn": MODEL_COLORS.get("maxwell_nn", "#ff7f0e"),
        "simple_rnn": MODEL_COLORS.get("simple_rnn", "#2ca02c"),
    }

    metric_title = metric.replace("_", " ").title()

    if style == "bar":
        _plot_ts_metrics_bar(metrics, model_types, colors, metric, metric_title)
    else:
        _plot_ts_metrics_scatter(metrics, model_types, colors, metric, metric_title)


def _plot_ts_metrics_scatter(metrics, model_types, colors, metric, metric_title):
    """Scatter plot: markers only, no connecting lines."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model_type in model_types:
        ts_data = metrics["metrics"][model_type]
        ts_values = sorted([int(k) for k in ts_data.keys()])

        interp_vals = [ts_data[str(ts)]["interpolation"][metric] for ts in ts_values]
        extrap_vals = [ts_data[str(ts)]["extrapolation"][metric] for ts in ts_values]

        label = MODEL_LABELS.get(model_type, model_type)
        color = colors.get(model_type, "gray")

        ax.scatter(ts_values, interp_vals, marker="o", s=80, color=color,
                   label=f"{label} (interpolation)", zorder=3)
        ax.scatter(ts_values, extrap_vals, marker="s", s=80, color=color,
                   facecolors="none", edgecolors=color, linewidths=2,
                   label=f"{label} (extrapolation)", zorder=3)

    ax.set_yscale("log")
    ax.set_xlabel("n_timesteps (training)")
    ax.set_ylabel(f"{metric_title} NRMSE")
    ax.set_title(f"Timestep Study — {metric_title} NRMSE\n"
                 f"Train: mixed_4 (corners), Interpolation vs Extrapolation")
    ax.set_xticks([50, 100, 200, 400])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_ts_metrics_bar(metrics, model_types, colors, metric, metric_title):
    """Grouped bar chart: interpolation + extrapolation side by side."""
    ts_data_first = metrics["metrics"][model_types[0]]
    ts_values = sorted([int(k) for k in ts_data_first.keys()])
    n_ts = len(ts_values)
    n_models = len(model_types)

    # Bar positions: group by n_timesteps, within each group one bar per (model, region)
    group_width = 0.8
    bar_width = group_width / (n_models * 2)
    x_base = np.arange(n_ts)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    for m_idx, model_type in enumerate(model_types):
        ts_data = metrics["metrics"][model_type]
        interp_vals = [ts_data[str(ts)]["interpolation"][metric] for ts in ts_values]
        extrap_vals = [ts_data[str(ts)]["extrapolation"][metric] for ts in ts_values]

        label = MODEL_LABELS.get(model_type, model_type)
        color = colors.get(model_type, "gray")

        offset_interp = (m_idx * 2) * bar_width - group_width / 2 + bar_width / 2
        offset_extrap = (m_idx * 2 + 1) * bar_width - group_width / 2 + bar_width / 2

        ax.bar(x_base + offset_interp, interp_vals, bar_width * 0.9,
               color=color, alpha=0.9, label=f"{label} (interp.)")
        ax.bar(x_base + offset_extrap, extrap_vals, bar_width * 0.9,
               color=color, alpha=0.45, label=f"{label} (extrap.)",
               edgecolor=color, linewidth=1.2)

    ax.set_yscale("log")
    ax.set_xlabel("n_timesteps (training)")
    ax.set_ylabel(f"{metric_title} NRMSE")
    ax.set_title(f"Timestep Study — {metric_title} NRMSE\n"
                 f"Train: mixed_4 (corners), Interpolation vs Extrapolation")
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(ts) for ts in ts_values])
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()


def plot_timestep_study_pointwise(
    test_points=None,
    timestep_values=None,
    model_types=None,
    steps=100000,
    search_dirs=None,
):
    """Plot NRMSE vs n_timesteps for specific (A, omega) test points.

    Instead of aggregating over many grid points, this shows the NRMSE for
    individual representative loadcases — e.g. one interpolation point and
    one extrapolation point.

    Each model is evaluated with the same n_timesteps it was trained on.

    Args:
        test_points: List of (A, omega) tuples to evaluate.
                     Default: [(1.5, 1.5), (6, 6)]
        timestep_values: List of n_timesteps. Default: [50, 100, 200, 400]
        model_types: List of model types. Default: ["gsm", "maxwell_nn", "simple_rnn"]
        steps: Training steps filter. Default: 100000
        search_dirs: Directories to search. Default: ["artifacts/timestep_study"]

    Examples:
        plot_timestep_study_pointwise()
        plot_timestep_study_pointwise(test_points=[(1.5, 1.5), (6, 6), (3, 3)])
        plot_timestep_study_pointwise(model_types=["gsm", "maxwell_nn"])
    """
    if test_points is None:
        test_points = [(1.5, 1.5), (6.0, 6.0)]
    if timestep_values is None:
        timestep_values = [50, 100, 200, 400]
    if model_types is None:
        model_types = ["gsm", "maxwell_nn", "simple_rnn"]
    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]

    n_points = len(test_points)
    fig, axes = plt.subplots(1, n_points, figsize=(4.5 * n_points, 4.5),
                             squeeze=False, sharey=True)
    axes = axes[0]

    # Track handles/labels for shared legend
    legend_handles = {}

    for pt_idx, (A, omega) in enumerate(test_points):
        ax = axes[pt_idx]

        for model_type in model_types:
            ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})
            if not ts_seeds:
                continue

            # Build model template
            key = jrandom.PRNGKey(0)
            if model_type == "gsm":
                model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
            elif model_type == "simple_rnn":
                model_template = tm.build(key=key)
            elif model_type == "maxwell_nn":
                model_template = tm.build_maxwell_nn(
                    key=key, E_infty=MATERIAL_PARAMS["E_infty"],
                    E_val=MATERIAL_PARAMS["E"])
            else:
                continue

            ts_vals_found = []
            nrmse_vals = []

            for n_ts in timestep_values:
                seed = ts_seeds.get(n_ts)
                if seed is None:
                    continue

                pattern = f"{model_type}__mixed_4__seed_{seed}"
                f = find_latest(pattern, steps=steps, search_dirs=search_dirs,
                                n_timesteps=n_ts)
                if f is None:
                    continue

                model = storage.load_model(f, model_template)
                model = klax.finalize(model)

                # Test data with same n_timesteps as training
                eps, sig, dts = _generate_test_data(
                    n_ts, [omega], [A], "harmonic", 0.0)
                sig_pred = jax.vmap(model)((eps, dts))

                rmse = float(np.sqrt(np.mean(
                    (np.array(sig_pred) - np.array(sig)) ** 2)))
                sig_std = float(np.std(np.array(sig)))
                nrmse = rmse / sig_std if sig_std > 1e-10 else rmse

                ts_vals_found.append(n_ts)
                nrmse_vals.append(nrmse)

            if ts_vals_found:
                label = MODEL_LABELS.get(model_type, model_type)
                color = MODEL_COLORS.get(model_type, "gray")
                ax.plot(ts_vals_found, nrmse_vals, color=color, alpha=0.3,
                        linewidth=1.5, zorder=2)
                sc = ax.scatter(ts_vals_found, nrmse_vals, marker="o", s=120,
                                color=color, zorder=3, edgecolors="white",
                                linewidths=0.8)
                if model_type not in legend_handles:
                    legend_handles[model_type] = (sc, label)

        ax.set_yscale("log")
        ax.set_xlabel("n_timesteps", fontsize=12)
        if pt_idx == 0:
            ax.set_ylabel("NRMSE", fontsize=12)
        ax.set_xticks(timestep_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.25, which="major")
        ax.grid(True, alpha=0.1, which="minor")

        # Subplot title: just the loadcase info
        in_interp = (1.0 <= A <= 4.0) and (1.0 <= omega <= 4.0)
        region = "Interpolation" if in_interp else "Extrapolation"
        ax.set_title(f"A={A}, ω={omega}  ({region})", fontsize=13, fontweight="bold")

    # Single shared legend below the figure
    handles = [legend_handles[mt][0] for mt in model_types if mt in legend_handles]
    labels = [legend_handles[mt][1] for mt in model_types if mt in legend_handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    plt.show()


def plot_training_loss(model_type=None, n_timesteps=None, seeds=None,
                       search_dirs=None, log=True):
    """Plot training loss curves from history JSON files.

    Flexible filtering: show all combos, or filter by model_type / n_timesteps / seeds.

    Args:
        model_type: Filter by model type (e.g. "gsm", "maxwell_nn", "simple_rnn"), or None for all
        n_timesteps: Filter by timestep value (e.g. 50, 100, 200, 400), or None for all
        seeds: List of seed indices to include, or None for all
        search_dirs: Directories to search. Default: ["artifacts/timestep_study"]
        log: If True, use log scale on y-axis. Default: True

    Examples:
        plot_training_loss()                                          # all
        plot_training_loss(model_type="gsm")                          # all GSM
        plot_training_loss(model_type="gsm", n_timesteps=200)         # GSM, 200 ts
        plot_training_loss(n_timesteps=400)                           # all models, 400 ts
        plot_training_loss(model_type="simple_rnn", seeds=[0, 1])     # RNN, seeds 0 & 1
    """
    from pathlib import Path

    if search_dirs is None:
        search_dirs = ["artifacts/timestep_study"]

    # Find all history JSONs
    all_jsons = []
    for d in search_dirs:
        p = Path(d)
        if p.exists():
            all_jsons.extend(p.glob("*_history.json"))

    if not all_jsons:
        print(f"Keine history JSONs gefunden in: {search_dirs}")
        return

    # Load and filter
    curves = []
    for jp in sorted(all_jsons):
        with open(jp, "r") as f_in:
            data = json.load(f_in)

        mt = data.get("model_type", "")
        nts = data.get("n_timesteps", 0)
        seed = data.get("seed", 0)
        losses = data.get("losses", [])

        if model_type is not None and mt != model_type:
            continue
        if n_timesteps is not None and nts != n_timesteps:
            continue
        if seeds is not None and seed not in seeds:
            continue
        if not losses:
            continue

        curves.append({
            "model_type": mt,
            "n_timesteps": nts,
            "seed": seed,
            "losses": losses,
            "final_loss": data.get("final_loss"),
        })

    if not curves:
        print("Keine passenden History-Dateien gefunden.")
        return

    print(f"Plotte {len(curves)} Trainingskurven")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Line styles per n_timesteps
    ts_linestyles = {50: "-", 100: "--", 200: "-.", 400: ":"}
    # Distinct seed colors (used when plotting a single model type)
    seed_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Detect if we're showing a single model type
    unique_models = set(c["model_type"] for c in curves)
    single_model = len(unique_models) == 1

    for c in curves:
        mt = c["model_type"]
        nts = c["n_timesteps"]
        seed = c["seed"]
        losses = c["losses"]

        if single_model:
            color = seed_colors[seed % len(seed_colors)]
        else:
            color = MODEL_COLORS.get(mt, "gray")
        ls = ts_linestyles.get(nts, "-")
        label = f"{MODEL_LABELS.get(mt, mt)} | ts={nts} | seed {seed}"

        # x-axis: steps (LOG_EVERY=500, so step_i = i * 500)
        steps_x = np.arange(1, len(losses) + 1) * 500
        ax.plot(steps_x, losses, color=color, linestyle=ls, alpha=0.85,
                linewidth=1.5, label=label)

    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Loss")

    # Build title from filters
    title_parts = ["Training Loss"]
    if model_type:
        title_parts.append(MODEL_LABELS.get(model_type, model_type))
    if n_timesteps:
        title_parts.append(f"ts={n_timesteps}")
    ax.set_title(" — ".join(title_parts))

    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_all_seeds(pattern, steps=None, seeds=None, test_loadcases=None, search_dirs=None,
                    noise_std_rel=0.0, n_timesteps=None):
    """Plot selected seeds for a config overlaid in one figure.

    Each seed gets a different color, ground truth is shown as black dashed line.
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # Collect filenames for requested seeds
    seed_files = []
    for seed in seeds:
        seed_pattern = f"{pattern}__seed_{seed}"
        f = find_latest(seed_pattern, steps=steps, search_dirs=search_dirs,
                        n_timesteps=n_timesteps)
        if f is not None:
            seed_files.append((seed, f))

    if not seed_files:
        print(f"Keine Modelle gefunden für Pattern '{pattern}'")
        return

    # Parse metadata from first file for title and model template
    name_only = str(seed_files[0][1]).split("/")[-1].split("\\")[-1]
    name_no_ext = name_only.replace(".eqx", "")
    parts = name_no_ext.split("__")

    if len(parts) == 6:
        model_type, experiment_name = parts[0], parts[1]
        train_steps = int(parts[3].replace("steps", ""))
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        model_type, experiment_name = parts[0], parts[1]
        train_steps = int(parts[2].replace("steps", ""))
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Dateinamen-Format: {name_only}")
        return

    # Build model template
    key = jrandom.PRNGKey(0)
    if model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    elif model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    # Generate test data
    eps_h, sig_h, dts_h = _generate_test_data(n_timesteps, omegas, As, "harmonic", noise_std_rel)
    eps_r, sig_r, dts_r = _generate_test_data(n_timesteps, omegas, As, "relaxation", noise_std_rel)

    n_pts = len(eps_h[0])
    ns = np.linspace(0, 2 * np.pi, n_pts)
    n_lc = len(test_loadcases)

    # Build title
    _TRAIN_INFO = {
        "omega_1": "Train: (A=1,ω=1)",
        "omega_2": "Train: (A=1,ω=1), (A=1,ω=2)",
        "omega_3": "Train: (A=1,ω=1..3)",
        "omega_4": "Train: (A=1,ω=1..4)",
        "amp_2":   "Train: (ω=1,A=1), (ω=1,A=2)",
        "amp_3":   "Train: (ω=1,A=1..3)",
        "amp_4":   "Train: (ω=1,A=1..4)",
        "mixed_4": "Train: (ω,A)∈{1,4}×{1,4}",
        "mixed_2": "Train: (ω,A)∈{1,2}×{1,2}",
    }
    train_info = _TRAIN_INFO.get(experiment_name, f"Train: {experiment_name}")
    noise_str = f" [noise={noise_std_rel:.0%}]" if noise_std_rel > 0 else ""
    base_title = f"{model_type.upper()} | {train_info} | {train_steps//1000}k steps | {len(seed_files)} seeds{noise_str}"

    # Seed colors (colormap)
    seed_cmap = plt.cm.tab10

    # --- Harmonic Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{base_title} — Harmonic Test", fontsize=11)

    for i in range(n_lc):
        axs[0].plot(ns, sig_h[i], linestyle=":", color="black", linewidth=1.5,
                    label=f"GT: ω={omegas[i]}, A={As[i]}" if i == 0 or n_lc > 1 else None)
        axs[1].plot(eps_h[i], sig_h[i], linestyle=":", color="black", linewidth=1.5)

    for seed, filepath in seed_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_h, dts_h))
        c = seed_cmap(seed)
        for i in range(n_lc):
            label = f"seed {seed}" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.7, label=label)
            axs[1].plot(eps_h[i], sig_pred[i], color=c, alpha=0.7)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()

    # --- Relaxation Plot ---
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{base_title} — Relaxation Test", fontsize=11)

    for i in range(n_lc):
        axs[0].plot(ns, sig_r[i], linestyle=":", color="black", linewidth=1.5,
                    label=f"GT: ω={omegas[i]}, A={As[i]}" if i == 0 or n_lc > 1 else None)
        axs[1].plot(eps_r[i], sig_r[i], linestyle=":", color="black", linewidth=1.5)

    for seed, filepath in seed_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)
        sig_pred = jax.vmap(model)((eps_r, dts_r))
        c = seed_cmap(seed)
        for i in range(n_lc):
            label = f"seed {seed}" if i == 0 else None
            axs[0].plot(ns, sig_pred[i], color=c, alpha=0.7, label=label)
            axs[1].plot(eps_r[i], sig_pred[i], color=c, alpha=0.7)

    axs[0].set_xlim([0, 2 * np.pi])
    axs[0].set_ylabel("stress $\\sigma$")
    axs[0].set_xlabel("time $t$")
    axs[0].legend(fontsize=8)
    axs[1].set_xlabel("strain $\\varepsilon$")
    axs[1].set_ylabel("stress $\\sigma$")
    fig.tight_layout()
    plt.show()


def plot_saved_model(filename: str, test_loadcases=None, noise_std_rel=0.0):
    """Load and plot predictions for a saved model file.

    Args:
        filename: Path to the .eqx model file (relative or absolute)
        test_loadcases: List of (A, omega) tuples. Default: [(1,1), (1,2), (1,3)]
        noise_std_rel: Relative noise std on eps (e.g. 0.02 = 2%). Default: 0 (clean)
    """
    # 1. Metadaten aus Dateinamen extrahieren
    #    Altes Format (5 Teile): {model}__{experiment}__{steps}steps__{n}ts__{timestamp}.eqx
    #    Neues Format (6 Teile): {model}__{experiment}__seed_{i}__{steps}steps__{n}ts__{timestamp}.eqx
    try:
        name_only = str(filename).split("/")[-1].split("\\")[-1]
        name_no_ext = name_only.replace(".eqx", "")
        parts = name_no_ext.split("__")

        if len(parts) == 5:
            model_type = parts[0]
            experiment_name = parts[1]
            seed_str = ""
            train_steps = int(parts[2].replace("steps", ""))
            n_timesteps = int(parts[3].replace("ts", ""))
        elif len(parts) == 6:
            model_type = parts[0]
            experiment_name = parts[1]
            seed_str = parts[2]  # e.g. "seed_0"
            train_steps = int(parts[3].replace("steps", ""))
            n_timesteps = int(parts[4].replace("ts", ""))
        else:
            raise ValueError(f"Unbekanntes Format ({len(parts)} Teile): {name_only}")

        # Lesbaren Titel bauen
        # Trainings-Konfigurationen aus experiment_name ableiten
        _TRAIN_INFO = {
            "omega_1": "Train: (A=1,ω=1)",
            "omega_2": "Train: (A=1,ω=1), (A=1,ω=2)",
            "omega_3": "Train: (A=1,ω=1..3)",
            "omega_4": "Train: (A=1,ω=1..4)",
            "amp_2":   "Train: (ω=1,A=1), (ω=1,A=2)",
            "amp_3":   "Train: (ω=1,A=1..3)",
            "amp_4":   "Train: (ω=1,A=1..4)",
            "mixed_4": "Train: (ω,A)∈{1,4}×{1,4}",
            "mixed_2": "Train: (ω,A)∈{1,2}×{1,2}",
        }
        train_info = _TRAIN_INFO.get(experiment_name, f"Train: {experiment_name}")
        steps_info = f"{train_steps//1000}k steps"
        seed_info = f", {seed_str}" if seed_str else ""
        model_title = f"{model_type.upper()} | {train_info} | {steps_info}{seed_info}"

        print(f"Lade Modell: {model_title}")
    except (ValueError, IndexError) as e:
        print(f"Konnte Metadaten nicht aus Dateinamen lesen: {e}")
        return
    
    # 2. Modell-Template erstellen (für Equinox Load)
    key = jrandom.PRNGKey(0)
    if model_type == "simple_rnn":
        model_template = tm.build(key=key)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, 
            E_infty=MATERIAL_PARAMS["E_infty"], 
            E_val=MATERIAL_PARAMS["E"]
        )
    elif model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0/MATERIAL_PARAMS["eta"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    # 3. Modell laden und finalisieren
    try:
        model = storage.load_model(filename, model_template)
        model = klax.finalize(model)
    except FileNotFoundError:
        print(f"Datei nicht gefunden: {filename}")
        return

    # 4. Testdaten generieren
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]
    
    noise_str = f" [noise={noise_std_rel:.0%}]" if noise_std_rel > 0 else ""

    # Harmonic Test
    print("Plotte Harmonic Test...")
    eps_h, sig_h, dts_h = _generate_test_data(n_timesteps, omegas, As, "harmonic", noise_std_rel)
    sig_pred_h = jax.vmap(model)((eps_h, dts_h))
    plot_model_pred(eps_h, sig_h, sig_pred_h, omegas, As,
                    title=f"{model_title} — Harmonic Test{noise_str}")

    # Relaxation Test
    print("Plotte Relaxation Test...")
    eps_r, sig_r, dts_r = _generate_test_data(n_timesteps, omegas, As, "relaxation", noise_std_rel)
    sig_pred_r = jax.vmap(model)((eps_r, dts_r))
    plot_model_pred(eps_r, sig_r, sig_pred_r, omegas, As,
                    title=f"{model_title} — Relaxation Test{noise_str}")

def plot_best_gamma(
    configs=None,
    steps=250000,
    test_loadcases=None,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=None,
):
    """
    Plot gamma(t) for the best GSM seed of each config vs analytical Maxwell gamma(t).

    Mirrors plot_best(...) but for internal variable gamma instead of stress sigma.
    Uses evaluation.simulate_model_batch to extract gamma trajectories.

    Args:
        configs: list like ["omega_4","amp_4","mixed_4"] (default: all BEST_SEEDS_GSM keys)
        steps: training steps filter, e.g. 250000
        test_loadcases: list of (A, omega) tuples, e.g. [(6,6)]
        search_dirs: optional override dirs; default GSM dirs
        noise_std_rel: relative noise on eps (0.0 for clean)
        n_test_timesteps: override number of timesteps in generated test trajectory
    """
    # --- choose best seeds + dirs (same logic as plot_best) ---
    best_seeds = BEST_SEEDS_GSM
    search_dirs = _get_search_dirs("gsm", search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    # --- find model files ---
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe...")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine GSM-Modelle gefunden.")
        return

    # --- determine timestep count from filename (same as plot_best) ---
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps

    # --- build templates for loading GSM + analytical Maxwell ---
    key = jrandom.PRNGKey(0)
    gsm_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    maxwell_model = tm.build_maxwell(
        E_infty=MATERIAL_PARAMS["E_infty"],
        E_val=MATERIAL_PARAMS["E"],
        eta=MATERIAL_PARAMS["eta"],
    )

    # --- generate test data (eps, dts) ---
    eps_h, sig_h, dts_h = _generate_test_data(
        n_ts_test, omegas, As, test_type="harmonic", noise_std_rel=noise_std_rel
    )

    # --- simulate Maxwell to get "true" gamma ---
    gamma_max, _ = ev.simulate_model_batch(maxwell_model, eps_h, dts_h)  # gamma: (N,T+1)

    # time axis (keep consistent with your other plots)
    T = eps_h.shape[1]
    t = np.linspace(0, 2 * np.pi, T + 1)

    # --- plot: one figure per loadcase (recommended) ---
    cmap = plt.cm.tab10
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""
    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])

    for lc_idx, (A, omega) in enumerate(test_loadcases):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.set_title(
            f"GSM γ(t) vs Maxwell γ(t) — Test {tc_str}{noise_str}{ts_str}\n"
            f"Showing loadcase A={A}, ω={omega}"
        )

        # ground truth gamma
        ax.plot(t, gamma_max[lc_idx], linestyle=":", color="black", linewidth=2, label="Maxwell γ (GT)")

        # each GSM config
        for idx, (config, seed, filepath) in enumerate(model_files):
            model = storage.load_model(filepath, gsm_template)
            model = klax.finalize(model)
            gamma_gsm, _ = ev.simulate_model_batch(model, eps_h, dts_h)
            c = cmap(idx % 10)
            ax.plot(t, gamma_gsm[lc_idx], color=c, alpha=0.85, linewidth=1.8, label=f"{config} (s{seed})")

        ax.set_xlabel("time t")
        ax.set_ylabel("internal variable γ")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        plt.show()

def plot_best_teacher_forced(
    configs=None,
    steps=250000,
    test_loadcases=None,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=None,
    plot_option_c: bool = True,
):

    """
    Option B diagnostic (teacher forcing) for GSM:

    Compare, for each best-seed GSM model:
      - closed-loop GSM sigma(t)
      - teacher-forced GSM sigma_TF(t) using gamma_true(t) from Maxwell
      - ground-truth Maxwell sigma(t) (from data generator)

    Works without retraining and without requiring a separate loader in user code.
    """

    # ----------- Select models (same scheme as plot_best) -----------
    model_type = "gsm"
    best_seeds = _get_best_seeds(model_type)
    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.0, 1.0)]

    As = [lc[0] for lc in test_loadcases]
    omegas = [lc[1] for lc in test_loadcases]

    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe.")
            continue
        pattern = f"{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # ----------- Parse n_timesteps from first file (same as plot_best) -----------
    name_only = str(model_files[0][2]).split("/")[-1].split("\\")[-1]
    parts = name_only.replace(".eqx", "").split("__")
    if len(parts) == 6:
        file_model_type = parts[0]
        n_timesteps = int(parts[4].replace("ts", ""))
    elif len(parts) == 5:
        file_model_type = parts[0]
        n_timesteps = int(parts[3].replace("ts", ""))
    else:
        print(f"Unbekanntes Format: {name_only}")
        return

    if file_model_type != "gsm":
        print(f"plot_best_teacher_forced ist nur für GSM gedacht, Datei-Typ war: {file_model_type}")
        return

    # ----------- Build templates (GSM + Maxwell) -----------
    key = jrandom.PRNGKey(0)
    gsm_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    maxwell_model = tm.build_maxwell(
        E_infty=MATERIAL_PARAMS["E_infty"],
        E_val=MATERIAL_PARAMS["E"],
        eta=MATERIAL_PARAMS["eta"],
    )

    # ----------- Generate test data (harmonic only for this diagnostic) -----------
    n_ts_test = n_test_timesteps if n_test_timesteps is not None else n_timesteps
    eps_h, sig_h, dts_h = _generate_test_data(n_ts_test, omegas, As, "harmonic", noise_std_rel)

    # time axis
    n_pts = len(eps_h[0])
    ts = np.linspace(0, 2 * np.pi, n_pts)

    # ----------- Get gamma_true(t) from Maxwell (evaluation only) -----------
    gamma_max, _ = ev.simulate_model_batch(maxwell_model, eps_h, dts_h)  # (N, T+1)

    # ----------- Teacher-forced sigma helper (no need to modify evaluation.py) -----------
    def teacher_forced_sigma(gsm_model, eps_batch, gamma_true_batch):
        """
        sigma_TF(t) = d/d_eps e_theta(eps(t), gamma_true(t))
        eps_batch: (N,T)
        gamma_true_batch: (N,T+1) or (N,T)
        returns: (N,T)
        """
        cell = gsm_model.cell
        if not hasattr(cell, "_energy"):
            raise ValueError("Given model does not look like GSM (missing cell._energy).")

        # align gamma length to T
        if gamma_true_batch.shape[1] == eps_batch.shape[1] + 1:
            gamma_T = gamma_true_batch[:, : eps_batch.shape[1]]
        else:
            gamma_T = gamma_true_batch

        de_deps = jax.grad(cell._energy, argnums=0)

        # vmap over time, then batch
        def one_case(eps_1, gam_1):
            return jax.vmap(de_deps)(eps_1, gam_1)

        sig_tf = jax.vmap(one_case)(jnp.asarray(eps_batch), jnp.asarray(gamma_T))
        return np.array(sig_tf)
    
    def teacher_forced_dsig_dgamma(gsm_model, eps_batch, gamma_forced_batch):
        """
        Compute dσ/dγ along a provided gamma trajectory:
            σ = ∂e/∂ε
            dσ/dγ = ∂/∂γ (∂e/∂ε) = ∂²e/(∂γ∂ε)

        eps_batch: (N,T)
        gamma_forced_batch: (N,T+1) or (N,T)
        returns: (N,T)
        """
        cell = gsm_model.cell
        if not hasattr(cell, "_energy"):
            raise ValueError("Given model does not look like GSM (missing cell._energy).")

        if gamma_forced_batch.shape[1] == eps_batch.shape[1] + 1:
            gamma_T = gamma_forced_batch[:, : eps_batch.shape[1]]
        else:
            gamma_T = gamma_forced_batch

        # σ(eps,gamma) = ∂e/∂eps
        de_deps = jax.grad(cell._energy, argnums=0)
        # dσ/dγ = ∂/∂γ (∂e/∂eps)
        dsig_dgamma = jax.grad(de_deps, argnums=1)

        def one_case(eps_1, gam_1):
            return jax.vmap(dsig_dgamma)(eps_1, gam_1)

        out = jax.vmap(one_case)(jnp.asarray(eps_batch), jnp.asarray(gamma_T))
        return np.array(out)


    # ----------- Plotting -----------
    n_lc = len(test_loadcases)
    cmap = plt.cm.tab10

    tc_str = ", ".join([f"(A={a},ω={w})" for a, w in test_loadcases])
    noise_str = f", noise={noise_std_rel:.0%}" if noise_std_rel > 0 else ""
    ts_str = f", test_ts={n_ts_test}" if n_test_timesteps is not None else ""

    E_true = MATERIAL_PARAMS["E"]
    dsig_dg_true = -float(E_true)


    # One figure per loadcase (keeps legend readable)
    for i_lc in range(n_lc):
        if plot_option_c:
            fig, axs = plt.subplots(2, 3, figsize=(16, 9))
            ax_t = axs[0, 0]
            ax_loop = axs[0, 1]
            ax_sens = axs[0, 2]

            # Bottom row (Option C panels)
            ax_sens_vs_gamma = axs[1, 0]
            ax_state_colored = axs[1, 1]
            ax_extra         = axs[1, 2]
        else:
            fig, axs = plt.subplots(1, 3, figsize=(16, 5))
            ax_t, ax_loop, ax_sens = axs
            # No bottom row in this case
            ax_sens_vs_gamma = None
            ax_state_colored = None
            ax_extra = None

        fig.suptitle(
            f"GSM Option B (Teacher forcing) — Harmonic — Test: {tc_str}{noise_str}{ts_str}\n"
            f"Showing loadcase A={As[i_lc]}, ω={omegas[i_lc]}",
            fontsize=11,
        )

        # Ground truth (from generator)
        ax_t.plot(ts, sig_h[i_lc], linestyle=":", color="black", linewidth=2, label="Ground Truth")
        ax_loop.plot(eps_h[i_lc], sig_h[i_lc], linestyle=":", color="black", linewidth=2)

        # Each best model: plot closed-loop (solid) + teacher-forced (dashed) in same color
        for idx, (config, seed, filepath) in enumerate(model_files):
            model = storage.load_model(filepath, gsm_template)
            model = klax.finalize(model)

            # closed-loop sigma
            sig_closed = np.array(jax.vmap(model)((eps_h, dts_h)))[i_lc]

            # teacher-forced sigma
            sig_tf = teacher_forced_sigma(model, eps_h, gamma_max)[i_lc]

            # gamma sensitivity of gsm energy
            dsig_dg_tf = teacher_forced_dsig_dgamma(model, eps_h, gamma_max)[i_lc]


            # sanity check
            gamma_gsm, _ = ev.simulate_model_batch(model, eps_h, dts_h)
            gmax = gamma_max[i_lc, :eps_h.shape[1]]
            ggsm = gamma_gsm[i_lc, :eps_h.shape[1]]
            print("max|γ_gsm-γ_max| =", np.max(np.abs(ggsm - gmax)))
            print("max|σ_closed-σ_TF| =", np.max(np.abs(sig_closed - sig_tf)))


            c = cmap(idx % 10)
            label_closed = f"{config} (s{seed}) closed" if i_lc == 0 else None
            label_tf = f"{config} (s{seed}) TF" if i_lc == 0 else None

            ax_t.plot(ts, sig_closed, color=c, alpha=0.85, linewidth=1.8, label=label_closed)
            ax_t.plot(ts, sig_tf, color="red", alpha=0.85, linewidth=1.8, linestyle="--", label=label_tf)

            ax_loop.plot(eps_h[i_lc], sig_closed, color=c, alpha=0.85, linewidth=1.8)
            ax_loop.plot(eps_h[i_lc], sig_tf, color=c, alpha=0.85, linewidth=1.8, linestyle="--")

            ax_sens.plot(ts, dsig_dg_tf, color=c, alpha=0.85, linewidth=1.4)

            if plot_option_c:
                # align teacher-forced gamma to length T
                gamma_tf_T = gamma_max[i_lc, :eps_h.shape[1]]
                eps_T = eps_h[i_lc]

                # (Row 2, Col 1): dσ/dγ vs γ, colored by ε
                sc1 = ax_sens_vs_gamma.scatter(
                    gamma_tf_T, dsig_dg_tf, c=eps_T, s=12, alpha=0.8
                )

                # (Row 2, Col 2): ε-γ state trajectory colored by dσ/dγ
                sc2 = ax_state_colored.scatter(
                    eps_T, gamma_tf_T, c=dsig_dg_tf, s=12, alpha=0.8
                )

                # also draw the path lightly to show loop ordering
                ax_state_colored.plot(eps_T, gamma_tf_T, linewidth=0.8, alpha=0.3)



        ax_t.set_xlim([0, 2 * np.pi])
        ax_t.set_xlabel("time $t$")
        ax_t.set_ylabel("stress $\\sigma$")
        ax_t.legend(fontsize=7, loc="best")

        ax_loop.set_xlabel("strain $\\varepsilon$")
        ax_loop.set_ylabel("stress $\\sigma$")

        ax_sens.set_xlabel("time $t$")
        ax_sens.set_ylabel(r"sensitivity $\partial\sigma/\partial\gamma$")
        ax_sens.set_title(r"$\partial\sigma/\partial\gamma$ (teacher-forced $\gamma_{max}$)")
        ax_sens.grid(alpha=0.25)
        # Existing (top row) sensitivity plot formatting
        ax_sens.axhline(dsig_dg_true, color="black", linestyle=":", linewidth=1.5, label="Maxwell: -E")
        ax_sens.legend(fontsize=7, loc="best")

        if plot_option_c:
            # Bottom-left: dσ/dγ vs γ
            ax_sens_vs_gamma.axhline(dsig_dg_true, color="black", linestyle=":", linewidth=1.5)
            ax_sens_vs_gamma.set_xlabel("internal variable γ (teacher-forced)")
            ax_sens_vs_gamma.set_ylabel(r"sensitivity $\partial\sigma/\partial\gamma$")
            ax_sens_vs_gamma.set_title(r"$\partial\sigma/\partial\gamma$ vs γ (colored by ε)")
            ax_sens_vs_gamma.grid(alpha=0.25)
            cbar1 = fig.colorbar(sc1, ax=ax_sens_vs_gamma)
            cbar1.set_label("strain ε")

            # Bottom-middle: ε–γ trajectory colored by sensitivity
            ax_state_colored.set_xlabel("strain ε")
            ax_state_colored.set_ylabel("internal variable γ (teacher-forced)")
            ax_state_colored.set_title("State trajectory (ε,γ) colored by ∂σ/∂γ")
            ax_state_colored.grid(alpha=0.25)
            cbar2 = fig.colorbar(sc2, ax=ax_state_colored)
            cbar2.set_label(r"$\partial\sigma/\partial\gamma$")

            # Bottom-right: leave empty or use for Δσ(t) later
            ax_extra.axis("off")




        fig.tight_layout()
        plt.show()

def plot_best_state_curvature(
    configs=None,
    steps=250000,
    search_dirs=None,
    noise_std_rel=0.0,
    n_test_timesteps=100,
    test_loadcases=None,
    eps_range=None,
    gamma_range=None,
    n_grid=120,
    show_training_states=True,
    show_test_states=True,
    model_type="gsm",
    n_timesteps=None,
    color_scale="linear",   # "linear" | "log" | "symlog"
    log_floor=1e-8,         # minimum magnitude for log scaling
    symlog_linthresh=1e-3
):
    """
    Plot energy curvature fields over state space (ε,γ) for best-seed models:
      - k_eps_eps(ε,γ) = ∂²e/∂ε²   (effective stiffness -> saturation if ~0)
      - k_eps_gam(ε,γ) = ∂²e/(∂γ∂ε) = ∂σ/∂γ  (coupling -> decoupling if ~0)

    Supports GSM models (learned energy) and Maxwell NN models (analytical energy).
    NOT supported for Simple RNN (no energy function).

    Overlays:
      - training trajectories (states visited by training loadcases for that config)
      - selected test trajectories (user-provided A,ω pairs)

    Parameters follow the same philosophy as plot_best(...):
      configs: list of config names (e.g. ["mixed_4"]) or None for all best seeds
      steps: train steps filter
      test_loadcases: list of (A, omega) tuples to overlay as OOD/ID examples
      eps_range/gamma_range: (min,max) for heatmap; if None, derived from overlay trajectories
      n_grid: resolution per axis
      model_type: "gsm" or "maxwell_nn" (selects best seeds + model template)
      n_timesteps: Optional filter for timestep study models (e.g. 50, 100, 200, 400)

    Examples:
      plot_best_state_curvature(["mixed_4"], model_type="gsm")
      plot_best_state_curvature(["mixed_4"], model_type="maxwell_nn", steps=100000)
      plot_best_state_curvature(["mixed_4"], model_type="gsm", n_timesteps=200,
                                steps=100000, search_dirs=["artifacts/timestep_study"])
    """

    import numpy as np
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    import matplotlib.pyplot as plt

    if model_type == "simple_rnn":
        print("Simple RNN hat keine Energiefunktion — Curvature-Plot nicht möglich.")
        return

    # Select best seeds: from timestep study dict or from main experiment dicts
    if n_timesteps is not None:
        ts_seeds = BEST_SEEDS_TIMESTEP_STUDY.get(model_type, {})
        seed_override = ts_seeds.get(n_timesteps)
        # Build a synthetic best_seeds dict with just one entry
        best_seeds = {}
        if seed_override is not None:
            for c in (configs or ["mixed_4"]):
                best_seeds[c] = seed_override
    else:
        best_seeds = _get_best_seeds(model_type)

    search_dirs = _get_search_dirs(model_type, search_dirs)

    if configs is None:
        configs = list(best_seeds.keys())
    if test_loadcases is None:
        test_loadcases = [(1.5, 1.5), (6.0, 6.0)]  # sensible default examples

    # Helper: training loadcases per experiment name (matches your plots.py TRAIN_INFO meaning)
    def _train_loadcases_for_config(name: str):
        # NOTE: This mapping matches how you've described/used omega_k, amp_k, mixed_k in your slides/plots.
        if name.startswith("omega_"):
            k = int(name.split("_")[1])
            return [(1.0, float(w)) for w in range(1, k + 1)]
        if name.startswith("amp_"):
            k = int(name.split("_")[1])
            return [(float(a), 1.0) for a in range(1, k + 1)]
        if name == "mixed_4":
            vals = [1.0, 2.0, 3.0, 4.0]
            return [(A, w) for A in vals for w in vals]
        if name == "mixed_2":
            vals = [1.0, 2.0]
            return [(A, w) for A in vals for w in vals]
        # fallback: assume trained on (1,1)
        return [(1.0, 1.0)]

    # Collect model files
    model_files = []
    for config in configs:
        seed = best_seeds.get(config)
        if seed is None:
            print(f"Kein best seed definiert für '{config}', überspringe.")
            continue
        pattern = f"{model_type}__{config}__seed_{seed}"
        f = find_latest(pattern, steps=steps, search_dirs=search_dirs,
                        n_timesteps=n_timesteps)
        if f is not None:
            model_files.append((config, seed, f))

    if not model_files:
        print("Keine Modelle gefunden.")
        return

    # Build model template for loading
    key = jrandom.PRNGKey(0)
    if model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(
            key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
    else:
        print(f"Unbekannter Modelltyp: {model_type}")
        return

    # Local helper to get (eps,gamma) trajectories from your analytical generator
    def _states_for_loadcases(loadcases):
        As = [lc[0] for lc in loadcases]
        omegas = [lc[1] for lc in loadcases]
        if noise_std_rel > 0:
            eps, gam, sig, dts = td.generate_data_harmonic_noisy_eps(
                MATERIAL_PARAMS["E_infty"], MATERIAL_PARAMS["E"], MATERIAL_PARAMS["eta"],
                n_test_timesteps, omegas, As,
                noise_std_rel=noise_std_rel, seed=0, recompute_eps_dot_from_noisy=False
            )
        else:
            eps, gam, sig, dts = td.generate_data_harmonic(
                MATERIAL_PARAMS["E_infty"], MATERIAL_PARAMS["E"], MATERIAL_PARAMS["eta"],
                n_test_timesteps, omegas, As
            )
        # eps,gam are (N,T)
        return np.asarray(eps), np.asarray(gam)

    # Compute default plotting ranges from overlays if not given
    def _infer_ranges(eps_train, gam_train, eps_test, gam_test):
        eps_all = []
        gam_all = []
        if eps_train is not None:
            eps_all.append(eps_train.reshape(-1))
            gam_all.append(gam_train.reshape(-1))
        if eps_test is not None:
            eps_all.append(eps_test.reshape(-1))
            gam_all.append(gam_test.reshape(-1))
        eps_all = np.concatenate(eps_all) if eps_all else np.array([-1, 1], dtype=float)
        gam_all = np.concatenate(gam_all) if gam_all else np.array([-1, 1], dtype=float)
        # small margins
        emn, emx = float(np.min(eps_all)), float(np.max(eps_all))
        gmn, gmx = float(np.min(gam_all)), float(np.max(gam_all))
        pad_e = 0.08 * (emx - emn + 1e-9)
        pad_g = 0.08 * (gmx - gmn + 1e-9)
        return (emn - pad_e, emx + pad_e), (gmn - pad_g, gmx + pad_g)

    # --- Main: one figure per config (2 heatmaps side-by-side) ---
    for config, seed, filepath in model_files:
        model = storage.load_model(filepath, model_template)
        model = klax.finalize(model)

        # training + test trajectories (state overlays)
        train_lcs = _train_loadcases_for_config(config)
        eps_tr, gam_tr = _states_for_loadcases(train_lcs) if show_training_states else (None, None)
        eps_te, gam_te = _states_for_loadcases(test_loadcases) if show_test_states else (None, None)

        # heatmap ranges
        if eps_range is None or gamma_range is None:
            (e0, e1), (g0, g1) = _infer_ranges(eps_tr, gam_tr, eps_te, gam_te)
        else:
            e0, e1 = eps_range
            g0, g1 = gamma_range

        # Build state grid
        eps_lin = jnp.linspace(e0, e1, n_grid)
        gam_lin = jnp.linspace(g0, g1, n_grid)
        EE, GG = jnp.meshgrid(eps_lin, gam_lin, indexing="xy")  # (n_grid,n_grid)

        # Access energy function e(ε,γ) depending on model type
        cell = model.cell
        if model_type == "gsm":
            if not hasattr(cell, "_energy"):
                raise ValueError("Model does not look like GSM (missing model.cell._energy).")
            def e_fun(eps_s, gam_s):
                return cell._energy(eps_s, gam_s)
        elif model_type == "maxwell_nn":
            E_inf = float(cell.E_infty)
            E_v = float(cell.E_val)
            def e_fun(eps_s, gam_s):
                return 0.5 * E_inf * eps_s**2 + 0.5 * E_v * (eps_s - gam_s)**2

        # Curvatures:
        # k_eps_eps = ∂²e/∂ε²
        # k_eps_gam = ∂²e/(∂γ∂ε) = ∂/∂γ(∂e/∂ε)
        de_deps = jax.grad(e_fun, argnums=0)
        k_eps_eps_fun = jax.grad(de_deps, argnums=0)
        k_eps_gam_fun = jax.grad(de_deps, argnums=1)

        # Vectorize over grid points
        pts = jnp.stack([EE.reshape(-1), GG.reshape(-1)], axis=1)  # (n_grid^2, 2)

        @jax.jit
        def eval_fields(pts_):
            eps_v = pts_[:, 0]
            gam_v = pts_[:, 1]
            k11 = jax.vmap(k_eps_eps_fun)(eps_v, gam_v)
            k12 = jax.vmap(k_eps_gam_fun)(eps_v, gam_v)
            return k11, k12

        k11_flat, k12_flat = eval_fields(pts)
        K11 = np.array(k11_flat).reshape(n_grid, n_grid)
        K12 = np.array(k12_flat).reshape(n_grid, n_grid)

        # --- NEW: norms for color scaling ---
        norm11 = None
        norm12 = None

        if color_scale == "log":
            # Only valid for strictly positive fields -> apply to K11, and to |K12| would lose sign (not recommended)
            k11_pos = np.abs(K11)
            vmin11 = max(log_floor, float(np.nanpercentile(k11_pos, 1)))
            vmax11 = float(np.nanpercentile(k11_pos, 99))
            norm11 = LogNorm(vmin=vmin11, vmax=max(vmax11, vmin11 * 10))

            # For K12: use symlog because it has negatives
            maxabs12 = float(np.nanpercentile(np.abs(K12), 99))
            norm12 = SymLogNorm(linthresh=symlog_linthresh, vmin=-maxabs12, vmax=maxabs12)

        elif color_scale == "symlog":
            # symlog for both (works even if positive-only)
            maxabs11 = float(np.nanpercentile(np.abs(K11), 99))
            maxabs12 = float(np.nanpercentile(np.abs(K12), 99))
            norm11 = SymLogNorm(linthresh=symlog_linthresh, vmin=-maxabs11, vmax=maxabs11)
            norm12 = SymLogNorm(linthresh=symlog_linthresh, vmin=-maxabs12, vmax=maxabs12)

        elif color_scale == "linear":
            norm11 = None
            norm12 = None
        else:
            raise ValueError("color_scale must be one of: 'linear', 'log', 'symlog'")


        # Plot
        model_label = MODEL_LABELS.get(model_type, model_type.upper())
        ts_str = f", ts={n_timesteps}" if n_timesteps is not None else ""
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{model_label} curvature fields — {config} (s{seed}) — {steps//1000}k steps{ts_str}\n"
            f"Heatmaps over state space (ε,γ), with trajectory overlays",
            fontsize=11
        )

        # Heatmap 1: stiffness in ε
        im0 = axs[0].imshow(
            K11,
            origin="lower",
            extent=[e0, e1, g0, g1],
            aspect="auto",
            norm=norm11,
        )

        im1 = axs[1].imshow(
            K12,
            origin="lower",
            extent=[e0, e1, g0, g1],
            aspect="auto",
            norm=norm12,
        )

        axs[1].set_title(r"$k_{\varepsilon\gamma}=\partial^2 e/(\partial\gamma\,\partial\varepsilon)=\partial\sigma/\partial\gamma$")
        axs[1].set_xlabel("strain ε")
        axs[1].set_ylabel("internal variable γ")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        # Overlay training trajectories (light gray)
        if show_training_states and eps_tr is not None:
            axs[0].scatter(eps_tr.reshape(-1), gam_tr.reshape(-1), s=3, alpha=0.12, color="white", edgecolors="none", label="train states")
            axs[1].scatter(eps_tr.reshape(-1), gam_tr.reshape(-1), s=3, alpha=0.12, color="white", edgecolors="none", label="train states")

        # Overlay selected test trajectories (colored)
        if show_test_states and eps_te is not None:
            cmap = plt.cm.tab10
            for i, (A, w) in enumerate(test_loadcases):
                c = cmap(i % 10)
                axs[0].plot(eps_te[i], gam_te[i], color=c, linewidth=1.2, alpha=0.9, label=f"test (A={A},ω={w})")
                axs[1].plot(eps_te[i], gam_te[i], color=c, linewidth=1.2, alpha=0.9, label=f"test (A={A},ω={w})")

        for ax in axs:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, loc="best")

        fig.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt

from .configs import MATERIAL_PARAMS
from . import data as td
from . import models as tm
from . import evaluation as ev


# -------------------------
# Coverage metrics
# -------------------------

def _convex_hull_monotone_chain(points: np.ndarray) -> np.ndarray:
    """Return convex hull vertices in CCW order using monotone chain. points: (N,2)."""
    pts = np.unique(points, axis=0)
    if pts.shape[0] <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]  # sort by x, then y

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace formula for polygon area. poly: (M,2) CCW."""
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def state_space_coverage_area(
    eps_batch: np.ndarray,
    gamma_batch: np.ndarray,
    method: str = "grid",
    bins: int = 200,
    eps_range=None,
    gamma_range=None,
) -> float:
    """
    Approximate covered 'area' in (ε,γ) state space for a set of trajectories.

    method:
      - "grid": occupancy-based union area estimate (recommended)
      - "hull": convex hull area (upper bound, can overestimate)

    eps_batch, gamma_batch: (N,T)
    """
    pts = np.stack([eps_batch.reshape(-1), gamma_batch.reshape(-1)], axis=1)

    if method == "hull":
        hull = _convex_hull_monotone_chain(pts)
        return _polygon_area(hull)

    if method != "grid":
        raise ValueError("method must be 'grid' or 'hull'")

    if eps_range is None:
        eps_min, eps_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    else:
        eps_min, eps_max = eps_range

    if gamma_range is None:
        g_min, g_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    else:
        g_min, g_max = gamma_range

    # avoid degenerate ranges
    eps_span = (eps_max - eps_min) if eps_max > eps_min else 1.0
    g_span = (g_max - g_min) if g_max > g_min else 1.0

    # bin indices
    ix = np.floor((pts[:, 0] - eps_min) / eps_span * bins).astype(int)
    iy = np.floor((pts[:, 1] - g_min) / g_span * bins).astype(int)
    ix = np.clip(ix, 0, bins - 1)
    iy = np.clip(iy, 0, bins - 1)

    occ = np.zeros((bins, bins), dtype=bool)
    occ[iy, ix] = True

    cell_area = (eps_span / bins) * (g_span / bins)
    return float(np.sum(occ) * cell_area)


# -------------------------
# Coverage visualization
# -------------------------

def plot_state_space_coverage_families(
    k: int = 6,
    n_timesteps: int = 400,
    families=("omega", "A", "both"),
    A0: float = 1.0,
    w0: float = 1.0,
    coverage_method: str = "grid",
    coverage_bins: int = 220,
):
    """
    Plot (ε,γ) state trajectories for 3 loadpath families and report coverage area.
    Uses analytical Maxwell model to compute γ from ε and dt.
    """

    def make_loadcases(fam: str):
        fam = fam.lower()
        if fam == "omega":
            return [(A0, float(w)) for w in range(1, k + 1)]
        if fam == "a":
            return [(float(A), w0) for A in range(1, k + 1)]
        if fam == "both":
            return [(float(i), float(i)) for i in range(1, k + 1)]
        raise ValueError("families must contain 'omega', 'A', or 'both'")

    E_inf = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    maxwell = tm.build_maxwell(E_infty=E_inf, E_val=E, eta=eta)

    fig, axs = plt.subplots(1, len(families), figsize=(5.6 * len(families), 4.8))
    if len(families) == 1:
        axs = [axs]

    for ax, fam in zip(axs, families):
        lcs = make_loadcases(fam)
        As = [lc[0] for lc in lcs]
        ws = [lc[1] for lc in lcs]

        eps, _, sig, dts = td.generate_data_harmonic(E_inf, E, eta, n_timesteps, ws, As)

        gamma_max, _ = ev.simulate_model_batch(maxwell, eps, dts)
        # simulate_model_batch often returns gamma with length T+1 (includes gamma0)
        if gamma_max.shape[1] == eps.shape[1] + 1:
            gamma_max = gamma_max[:, : eps.shape[1]]


        # plot all trajectories
        cmap = plt.cm.tab10
        for i, (A, w) in enumerate(lcs):
            c = cmap(i % 10)
            ax.plot(eps[i], gamma_max[i], color=c, alpha=0.85, linewidth=1.2, label=f"A={A},ω={w}")

        area_grid = state_space_coverage_area(eps, gamma_max, method=coverage_method, bins=coverage_bins)

        ax.set_title(f"Family: {fam}  |  coverage≈{area_grid:.2f} ({coverage_method})")
        ax.set_xlabel("strain ε")
        ax.set_ylabel("internal variable γ (Maxwell)")
        ax.grid(alpha=0.25)

        # legend can get huge; keep small or remove
        if len(lcs) <= 6:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"State-space coverage by training family (k={k}, T={n_timesteps})", fontsize=12)
    fig.tight_layout()
    plt.show()

def plot_state_space_coverage_custom(
    A_list,
    omega_list,
    n_timesteps: int = 400,
    combine: str = "product",  # "product" or "zip"
    coverage_method: str = "grid",
    coverage_bins: int = 220,
    title: str | None = None,
):
    """
    Plot (ε,γ) trajectories for a custom set of loadcases and report coverage.

    combine:
      - "product": use Cartesian product A×ω  (all pairs)
      - "zip": pair-wise (A_i, ω_i), requires same length
    """

    A_list = [float(a) for a in A_list]
    omega_list = [float(w) for w in omega_list]

    if combine not in ("product", "zip"):
        raise ValueError("combine must be 'product' or 'zip'")

    if combine == "zip":
        if len(A_list) != len(omega_list):
            raise ValueError("For combine='zip', A_list and omega_list must have same length.")
        loadcases = list(zip(A_list, omega_list))
    else:
        loadcases = [(A, w) for A in A_list for w in omega_list]

    E_inf = MATERIAL_PARAMS["E_infty"]
    E = MATERIAL_PARAMS["E"]
    eta = MATERIAL_PARAMS["eta"]

    # generate eps(t), dts(t) from harmonic generator
    As = [lc[0] for lc in loadcases]
    ws = [lc[1] for lc in loadcases]
    eps, _, _, dts = td.generate_data_harmonic(E_inf, E, eta, n_timesteps, ws, As)

    # simulate Maxwell to get gamma(t)
    maxwell = tm.build_maxwell(E_infty=E_inf, E_val=E, eta=eta)
    gamma_max, _ = ev.simulate_model_batch(maxwell, eps, dts)

    # align gamma length (T+1 -> T)
    if gamma_max.shape[1] == eps.shape[1] + 1:
        gamma_max = gamma_max[:, : eps.shape[1]]

    area = state_space_coverage_area(
        eps, gamma_max, method=coverage_method, bins=coverage_bins
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    cmap = plt.cm.tab20

    for i, (A, w) in enumerate(loadcases):
        c = cmap(i % 20)
        ax.plot(eps[i], gamma_max[i], color=c, alpha=0.85, linewidth=1.2, label=f"A={A},ω={w}")

    ax.set_xlabel("strain ε")
    ax.set_ylabel("internal variable γ (Maxwell)")
    ax.grid(alpha=0.25)

    if title is None:
        title = f"Custom coverage | N={len(loadcases)} | area≈{area:.2f} ({coverage_method}, bins={coverage_bins})"
    ax.set_title(title)

    # legend management
    if len(loadcases) <= 10:
        ax.legend(fontsize=8, loc="best")
    else:
        ax.legend([], [], frameon=False)

    fig.tight_layout()
    plt.show()

    return area


# =============================================================================
# Noise Robustness Comparison
# =============================================================================

def plot_noise_robustness(
    configs: list[str] | None = None,
    test_loadcase: tuple[float, float] = (6.0, 6.0),
    noise_std_rel: float = 0.05,
    model_types: list[str] | None = None,
    steps: int | dict[str, int] = None,
    search_dirs: list[str] | None = None,
    n_test_timesteps: int = 200,
    n_timesteps: int | None = None,
    test_type: str = "harmonic",
    figsize: tuple = (14, 5),
    fontsize_title: int = 18,
    fontsize_label: int = 16,
    fontsize_tick: int = 14,
    fontsize_legend: int = 14,
):
    """Compare noise robustness across model types.

    For each model: run prediction on clean and noisy eps, compute the difference
    delta_sig = sig_pred_clean - sig_pred_noisy, and report metrics on delta_sig.

    Args:
        configs: Training configs to use (default: ["mixed_4"]).
        test_loadcase: (A, omega) test case.
        noise_std_rel: Relative noise std on epsilon (e.g. 0.05 = 5%).
        model_types: List of model types (default: ["simple_rnn", "maxwell_nn", "gsm"]).
        steps: Training steps filter. Either a single int for all models,
               or a dict mapping model_type -> steps.
               Default: maxwell_nn=100000, others=250000.
        search_dirs: Override search directories (applied to all model types).
        n_test_timesteps: Number of test timesteps.
        n_timesteps: Optional filter for timestep study models.
        test_type: "harmonic" or "relaxation".

    Example:
        plot_noise_robustness(
            configs=["mixed_4"],
            test_loadcase=(6.0, 6.0),
            noise_std_rel=0.05,
            model_types=["simple_rnn", "maxwell_nn", "gsm"],
        )
        # Custom steps per model:
        plot_noise_robustness(steps={"maxwell_nn": 100000, "gsm": 250000, "simple_rnn": 250000})
    """
    if configs is None:
        configs = ["mixed_4"]
    if model_types is None:
        model_types = ["simple_rnn", "maxwell_nn", "gsm"]

    # Default steps: maxwell_nn trained with 100k, others with 250k
    _DEFAULT_STEPS = {"maxwell_nn": 100000}
    _DEFAULT_FALLBACK = 250000
    if steps is None:
        steps_dict = _DEFAULT_STEPS
    elif isinstance(steps, int):
        steps_dict = {mt: steps for mt in model_types}
    else:
        steps_dict = steps  # already a dict

    def _steps_for(mt):
        return steps_dict.get(mt, _DEFAULT_FALLBACK)

    A, omega = test_loadcase

    # --- Generate test data (clean and noisy) ---
    eps_clean, sig_true, dts = _generate_test_data(
        n_test_timesteps, [omega], [A], test_type, noise_std_rel=0.0)
    eps_noisy, _, _ = _generate_test_data(
        n_test_timesteps, [omega], [A], test_type, noise_std_rel=noise_std_rel)

    eps_clean = jnp.array(eps_clean)
    eps_noisy = jnp.array(eps_noisy)
    dts_jnp = jnp.array(dts)

    key = jrandom.PRNGKey(0)
    n_ts = len(eps_clean[0])
    ns = np.linspace(0, 2 * np.pi, n_ts)

    # --- Collect results per model ---
    results = {}  # model_label -> {delta_sig, metrics}

    for mt in model_types:
        best_seeds = _get_best_seeds(mt)
        sd = _get_search_dirs(mt, search_dirs)

        # Build model template
        if mt == "gsm" or mt == "gsm_sobolev":
            model_template = tm.build_gsm(key=key, g=1.0 / MATERIAL_PARAMS["eta"])
        elif mt == "simple_rnn":
            model_template = tm.build(key=key)
        elif mt == "maxwell_nn":
            model_template = tm.build_maxwell_nn(
                key=key, E_infty=MATERIAL_PARAMS["E_infty"], E_val=MATERIAL_PARAMS["E"])
        else:
            print(f"Unknown model_type: {mt}, skipping.")
            continue

        label = MODEL_LABELS.get(mt, mt.upper())
        color = MODEL_COLORS.get(mt, "#333333")

        for config in configs:
            seed = best_seeds.get(config)
            if seed is None:
                print(f"No best seed for {mt}/{config}, skipping.")
                continue

            pattern = f"{mt}__{config}__seed_{seed}" if mt != "gsm_sobolev" else f"gsm__{config}__seed_{seed}"
            filepath = find_latest(pattern, steps=_steps_for(mt), search_dirs=sd,
                                   n_timesteps=n_timesteps)
            if filepath is None:
                continue

            model = storage.load_model(filepath, model_template)
            model = klax.finalize(model)

            # Run clean and noisy predictions
            sig_pred_clean = np.array(jax.vmap(model)((eps_clean, dts_jnp)))
            sig_pred_noisy = np.array(jax.vmap(model)((eps_noisy, dts_jnp)))

            # Difference (squeeze from (1,T) to (T,))
            delta_sig = (sig_pred_clean - sig_pred_noisy).squeeze()
            sig_clean_sq = sig_pred_clean.squeeze()

            # Compute metrics
            rmse = float(np.sqrt(np.mean(delta_sig ** 2)))
            sig_std = float(np.std(sig_clean_sq))
            nrmse = rmse / sig_std if sig_std > 1e-10 else 0.0
            mean_abs = float(np.mean(np.abs(delta_sig)))
            median_abs = float(np.median(np.abs(delta_sig)))
            min_abs = float(np.min(np.abs(delta_sig)))
            max_abs = float(np.max(np.abs(delta_sig)))

            results[label] = {
                "delta_sig": delta_sig,
                "color": color,
                "metrics": {
                    "RMSE": rmse,
                    "NRMSE": nrmse,
                    "Mean": mean_abs,
                    "Median": median_abs,
                    "Min": min_abs,
                    "Max": max_abs,
                },
            }

            print(f"{label:15s}  RMSE={rmse:.6f}  NRMSE={nrmse:.6f}  "
                  f"Mean={mean_abs:.6f}  Max={max_abs:.6f}")

    if not results:
        print("No models found.")
        return

    # --- Plot ---
    fig, (ax_time, ax_bar) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: delta_sig(t) per model
    for label, res in results.items():
        ax_time.plot(ns, res["delta_sig"], color=res["color"], linewidth=2, label=label)
    ax_time.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_time.set_xlabel("time $t$", fontsize=fontsize_label)
    ax_time.set_ylabel(r"$\Delta\sigma = \sigma_{\mathrm{clean}} - \sigma_{\mathrm{noisy}}$",
                       fontsize=fontsize_label)
    ax_time.set_xlim([0, 2 * np.pi])
    ax_time.tick_params(labelsize=fontsize_tick)
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=fontsize_legend)

    # Right panel: grouped bar chart of metrics
    metric_names = ["RMSE", "NRMSE", "Mean", "Median", "Max"]
    model_labels = list(results.keys())
    n_metrics = len(metric_names)
    n_models = len(model_labels)
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    for i, label in enumerate(model_labels):
        vals = [results[label]["metrics"][m] for m in metric_names]
        ax_bar.bar(x + i * bar_width, vals, bar_width,
                   color=results[label]["color"], label=label, alpha=0.85)

    ax_bar.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax_bar.set_xticklabels(metric_names, fontsize=fontsize_tick)
    ax_bar.tick_params(axis="y", labelsize=fontsize_tick)
    ax_bar.set_ylabel("Value", fontsize=fontsize_label)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.legend(fontsize=fontsize_legend)

    fig.suptitle(
        f"Noise Robustness at $(A, \\omega) = ({A:.0f}, {omega:.0f})$, "
        f"noise$={noise_std_rel * 100:.0f}\\%$",
        fontsize=fontsize_title, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()

    return results


# =============================================================================
# Amplitude Ceiling Plot
# =============================================================================

def plot_amplitude_ceiling(
    filepath: str,
    omega: float = 6.0,
    A_values: np.ndarray | list | None = None,
    n_timesteps: int = 100,
    deriv_agg: str = "mean",
    mode: str = "line",
    model_type: str = "gsm",
    train_A_max: float | None = 4.0,
    figsize: tuple = (12, 5),
    fontsize_title: int = 18,
    fontsize_label: int = 16,
    fontsize_tick: int = 14,
    fontsize_legend: int = 14,
):
    """Plot sigma_max and energy derivatives vs test amplitude.

    Args:
        filepath: Path to a saved .eqx model file.
        omega: Fixed test frequency (default: 6.0).
        A_values: Test amplitudes. Default: np.arange(0.5, 15.5, 0.5).
        n_timesteps: Time discretization for each test (default: 100).
        deriv_agg: Aggregation for derivatives: "mean", "median", "max", "min".
                   Only used when mode="line".
        mode: "line" — aggregated line plot (one value per amplitude).
              "scatter" — scatter all trajectory points, x=amplitude, y=derivative value.
              "density" — KDE density plot.
        model_type: "gsm" or "maxwell_nn".
        train_A_max: Vertical line at max training amplitude (default: 4.0).
        figsize: Figure size.

    Examples:
        plot_amplitude_ceiling(f)
        plot_amplitude_ceiling(f, model_type="maxwell_nn")
        plot_amplitude_ceiling(f, mode="density")
    """
    if A_values is None:
        A_values = np.arange(0.5, 15.5, 0.5)

    mp = MATERIAL_PARAMS
    E_inf = mp["E_infty"]
    E_val = mp["E"]
    eta = mp["eta"]

    # --- Aggregation function ---
    _AGG_FNS = {
        "mean": jnp.mean, "median": jnp.median,
        "max": jnp.max, "min": jnp.min,
    }
    if deriv_agg not in _AGG_FNS:
        raise ValueError(f"deriv_agg must be one of {list(_AGG_FNS)}, got '{deriv_agg}'")
    agg_fn = _AGG_FNS[deriv_agg]

    # --- Load model ---
    key = jrandom.PRNGKey(0)
    if model_type == "gsm":
        model_template = tm.build_gsm(key=key, g=1.0 / eta)
    elif model_type == "maxwell_nn":
        model_template = tm.build_maxwell_nn(key=key, E_infty=E_inf, E_val=E_val)
    else:
        raise ValueError(f"model_type must be 'gsm' or 'maxwell_nn', got '{model_type}'")
    model = storage.load_model(filepath, model_template)
    model = klax.finalize(model)

    # --- Build energy function depending on model type ---
    cell_i = model.cell
    if model_type == "gsm":
        def e_fun(eps_s, gam_s):
            return cell_i._energy(eps_s, gam_s)
    else:  # maxwell_nn — analytical energy
        _E_inf = float(cell_i.E_infty)
        _E_v = float(cell_i.E_val)
        def e_fun(eps_s, gam_s):
            return 0.5 * _E_inf * eps_s**2 + 0.5 * _E_v * (eps_s - gam_s)**2

    de_deps = jax.grad(e_fun, argnums=0)
    k_ee_fun = jax.grad(de_deps, argnums=0)
    k_eg_fun = jax.grad(de_deps, argnums=1)
    de_dgamma_fun = jax.grad(e_fun, argnums=1)

    # --- Run model trajectory ---
    def _run_model_trajectory(model, eps_1d, dts_1d):
        cell_run = model.cell
        if model_type == "gsm":
            def step(gamma, x):
                eps_t, dt_t = x
                sig_t = de_deps(eps_t, gamma)
                d2e_ee_t = k_ee_fun(eps_t, gamma)
                d2e_eg_t = k_eg_fun(eps_t, gamma)
                de_dgamma = de_dgamma_fun(eps_t, gamma)
                gamma_new = gamma + dt_t * (-cell_run.g * de_dgamma)
                return gamma_new, (sig_t, d2e_ee_t, d2e_eg_t)
        else:  # maxwell_nn — learned evolution f(eps,gamma)*(eps-gamma)
            def step(gamma, x):
                eps_t, dt_t = x
                sig_t = de_deps(eps_t, gamma)
                d2e_ee_t = k_ee_fun(eps_t, gamma)
                d2e_eg_t = k_eg_fun(eps_t, gamma)
                f_val = cell_run.f_theta(eps_t, gamma)
                gamma_dot = f_val * (eps_t - gamma)
                gamma_new = gamma + dt_t * gamma_dot
                return gamma_new, (sig_t, d2e_ee_t, d2e_eg_t)
        init_gamma = jnp.array(0.0)
        _, (sig, d2e_ee, d2e_eg) = jax.lax.scan(step, init_gamma, (eps_1d, dts_1d))
        return sig, d2e_ee, d2e_eg

    @eqx.filter_jit
    def _run_jit(model, eps_1d, dts_1d):
        return _run_model_trajectory(model, eps_1d, dts_1d)

    # --- Collect data ---
    model_sig_max = []
    model_d2ee_agg = []
    model_d2eg_agg = []
    gt_sig_max = []

    # For scatter mode: raw trajectory points
    scatter_A_model = []
    scatter_d2ee_model = []
    scatter_d2eg_model = []

    # --- Detect if learned energy has γ with flipped sign ---
    # Only relevant for GSM (learned energy). Maxwell NN has analytical energy → no flip.
    _d2e_eg_check = float(k_eg_fun(jnp.array(0.5), jnp.array(0.0)))
    _gamma_sign_flip = -1.0 if _d2e_eg_check > 0 else 1.0

    _model_label = "GSM" if model_type == "gsm" else "Maxwell+NN"
    print(f"Computing amplitude ceiling ({_model_label}) for {len(A_values)} amplitudes at ω={omega}..."
          f" (γ sign flip: {_gamma_sign_flip == -1.0})")
    for A in A_values:
        eps, sig_true, dts = _generate_test_data(n_timesteps, [omega], [A], "harmonic")
        eps_1d = jnp.array(eps[0])
        dts_1d = jnp.array(dts[0])

        sig_m, d2e_ee_m, d2e_eg_m = _run_jit(model, eps_1d, dts_1d)
        d2e_eg_m = d2e_eg_m * _gamma_sign_flip  # correct for learned γ sign
        model_sig_max.append(float(jnp.max(sig_m)))
        gt_sig_max.append(float(np.max(sig_true[0])))

        if mode in ("scatter", "density"):
            n_pts = len(d2e_ee_m)
            scatter_A_model.extend([A] * n_pts)
            scatter_d2ee_model.extend(np.array(d2e_ee_m).tolist())
            scatter_d2eg_model.extend(np.array(d2e_eg_m).tolist())
        else:
            model_d2ee_agg.append(float(agg_fn(d2e_ee_m)))
            model_d2eg_agg.append(float(agg_fn(d2e_eg_m)))

    model_sig_max = np.array(model_sig_max)
    gt_sig_max = np.array(gt_sig_max)

    # --- Shared σ axis range ---
    all_sig = np.concatenate([model_sig_max, gt_sig_max])
    sig_ylim = (0, np.max(all_sig) * 1.08)

    # =====================================================================
    if mode in ("scatter", "density"):
        scatter_A_model = np.array(scatter_A_model)
        scatter_d2ee_model = np.array(scatter_d2ee_model)
        scatter_d2eg_model = np.array(scatter_d2eg_model)

        # Shared y-axis range for both derivative panels
        all_deriv_vals = np.concatenate([scatter_d2ee_model, scatter_d2eg_model,
                                         [E_inf + E_val, -E_val]])
        deriv_ylim = (min(float(np.min(all_deriv_vals)), -E_val) - 0.3,
                      max(float(np.max(all_deriv_vals)), E_inf + E_val) + 0.3)

        fig, axes = plt.subplots(1, 3, figsize=(figsize[0] * 1.3, figsize[1]))

        # Panel 1: σ_max (line, both model + GT)
        ax_sig = axes[0]
        h_gsm, = ax_sig.plot(A_values, model_sig_max, "-o", markersize=4, color="#f6a315")
        h_gt, = ax_sig.plot(A_values, gt_sig_max, "--", linewidth=2.5, color="#d32f2f", alpha=0.9)
        ax_sig.set_title(r"max $\sigma = \partial e / \partial\varepsilon$", fontsize=fontsize_title, fontweight="bold")
        ax_sig.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
        ax_sig.set_ylabel(r"max $\sigma$", fontsize=fontsize_label)
        ax_sig.tick_params(labelsize=fontsize_tick)
        ax_sig.set_ylim(sig_ylim)
        ax_sig.grid(True, alpha=0.3)

        # Vertical line: max training amplitude
        h_vline = None
        if train_A_max is not None:
            for ax in axes:
                ln = ax.axvline(train_A_max, color="#333333", linestyle=":", linewidth=2.5, alpha=0.85)
            h_vline = ln

        if mode == "density":
            from scipy.stats import gaussian_kde

            def _plot_kde(ax, x_data, y_data, gt_val, cmap, title, ylabel, ylim=None):
                """Helper: KDE-based smooth density plot. Falls back to scatter if data is degenerate."""
                x_min, x_max = x_data.min(), x_data.max()
                if ylim is not None:
                    y_min, y_max = ylim
                else:
                    y_min, y_max = y_data.min(), y_data.max()
                    y_pad = (y_max - y_min) * 0.1
                    y_min -= y_pad
                    y_max += y_pad

                try:
                    n_grid = 200
                    xi = np.linspace(x_min, x_max, n_grid)
                    yi = np.linspace(y_min, y_max, n_grid)
                    Xi, Yi = np.meshgrid(xi, yi)
                    positions = np.vstack([Xi.ravel(), Yi.ravel()])

                    kde = gaussian_kde(np.vstack([x_data, y_data]), bw_method=0.08)
                    Z = kde(positions).reshape(Xi.shape)

                    ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                              extent=[x_min, x_max, y_min, y_max],
                              interpolation="bilinear")
                except np.linalg.LinAlgError:
                    # Data is (near-)constant in y → KDE fails. Fall back to scatter.
                    color = {"Blues": "#436384", "Greens": "#16a48a"}.get(cmap, "#436384")
                    ax.scatter(x_data, y_data, s=2, alpha=0.3, color=color)
                    ax.grid(True, alpha=0.3)

                ax.axhline(gt_val, color="#d32f2f", linestyle="--", linewidth=2.5, alpha=0.9)
                ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
                ax.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
                ax.set_ylabel(ylabel, fontsize=fontsize_label)
                ax.tick_params(labelsize=fontsize_tick)
                if ylim is not None:
                    ax.set_ylim(ylim)

            _plot_kde(axes[1], scatter_A_model, scatter_d2ee_model,
                      E_inf + E_val, "Blues",
                      r"$\partial\sigma / \partial\varepsilon$",
                      r"$\partial\sigma / \partial\varepsilon$",
                      ylim=deriv_ylim)

            _plot_kde(axes[2], scatter_A_model, scatter_d2eg_model,
                      -E_val, "Greens",
                      r"$\partial\sigma / \partial\gamma$",
                      r"$\partial\sigma / \partial\gamma$",
                      ylim=deriv_ylim)

        else:  # scatter
            # Panel 2: scatter ∂σ/∂ε
            ax_ee = axes[1]
            ax_ee.scatter(scatter_A_model, scatter_d2ee_model, s=2, alpha=0.3, color="#436384")
            ax_ee.axhline(E_inf + E_val, color="#d32f2f", linestyle="--", linewidth=2.5, alpha=0.9)
            ax_ee.set_title(r"$\partial\sigma / \partial\varepsilon$", fontsize=fontsize_title, fontweight="bold")
            ax_ee.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
            ax_ee.set_ylabel(r"$\partial\sigma / \partial\varepsilon$", fontsize=fontsize_label)
            ax_ee.tick_params(labelsize=fontsize_tick)
            ax_ee.set_ylim(deriv_ylim)
            ax_ee.grid(True, alpha=0.3)

            # Panel 3: scatter ∂σ/∂γ
            ax_eg = axes[2]
            ax_eg.scatter(scatter_A_model, scatter_d2eg_model, s=2, alpha=0.3, color="#16a48a")
            ax_eg.axhline(-E_val, color="#d32f2f", linestyle="--", linewidth=2.5, alpha=0.9)
            ax_eg.set_title(r"$\partial\sigma / \partial\gamma$", fontsize=fontsize_title, fontweight="bold")
            ax_eg.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
            ax_eg.set_ylabel(r"$\partial\sigma / \partial\gamma$", fontsize=fontsize_label)
            ax_eg.tick_params(labelsize=fontsize_tick)
            ax_eg.set_ylim(deriv_ylim)
            ax_eg.grid(True, alpha=0.3)

        # Shared legend below all panels
        legend_handles = [h_gsm]
        legend_labels = [_model_label]
        if h_vline is not None:
            legend_handles.append(h_vline)
            legend_labels.append(f"Train $A_{{\\mathrm{{max}}}}$")
        legend_handles.append(h_gt)
        legend_labels.append("Ground Truth")
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=len(legend_handles), fontsize=fontsize_legend + 1,
                   frameon=False, bbox_to_anchor=(0.5, -0.06))

        fig.suptitle(f"Amplitude Response at $\\omega = {omega}$",
                     fontsize=fontsize_title + 1, fontweight="bold", y=1.02)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        plt.show()

    # =====================================================================
    else:  # mode == "line"
        model_d2ee_agg = np.array(model_d2ee_agg)
        model_d2eg_agg = np.array(model_d2eg_agg)
        gt_d2ee = np.full_like(A_values, E_inf + E_val, dtype=float)
        gt_d2eg = np.full_like(A_values, -E_val, dtype=float)

        all_deriv = np.concatenate([model_d2ee_agg, model_d2eg_agg, gt_d2ee, gt_d2eg])
        deriv_min = np.min(all_deriv)
        deriv_max = np.max(all_deriv)
        deriv_pad = (deriv_max - deriv_min) * 0.15
        deriv_ylim = (deriv_min - deriv_pad, deriv_max + deriv_pad)

        agg_label = deriv_agg

        fig, (ax_model, ax_gt) = plt.subplots(1, 2, figsize=figsize)

        # Vertical line: max training amplitude
        if train_A_max is not None:
            for ax in (ax_model, ax_gt):
                ax.axvline(train_A_max, color="#333333", linestyle=":", linewidth=2.5,
                           alpha=0.85, label=f"train $A_{{max}}={train_A_max:.0f}$")

        # === Left: Model ===
        ax_model.plot(A_values, model_sig_max, "-o", markersize=4, label=r"GSM max $\sigma$", color="#f6a315")
        ax_model.set_title("Trained GSM Model", fontsize=fontsize_title, fontweight="bold")
        ax_model.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
        ax_model.set_ylabel(r"max $\sigma$", fontsize=fontsize_label, color="#f6a315")
        ax_model.tick_params(axis="y", labelsize=fontsize_tick, colors="#f6a315")
        ax_model.tick_params(axis="x", labelsize=fontsize_tick)
        ax_model.set_ylim(sig_ylim)
        ax_model.grid(True, alpha=0.3)

        ax_model_r = ax_model.twinx()
        ax_model_r.plot(A_values, model_d2ee_agg, "-s", markersize=4,
                        label=rf"{agg_label} $\partial^2 e / \partial\varepsilon^2$", color="#436384")
        ax_model_r.plot(A_values, model_d2eg_agg, "-^", markersize=4,
                        label=rf"{agg_label} $\partial^2 e / \partial\varepsilon\,\partial\gamma$", color="#16a48a")
        ax_model_r.set_ylabel("Derivatives", fontsize=fontsize_label, color="#436384")
        ax_model_r.tick_params(axis="y", labelsize=fontsize_tick, colors="#436384")
        ax_model_r.set_ylim(deriv_ylim)

        # Legend outside below
        lines_l, labels_l = ax_model.get_legend_handles_labels()
        lines_r, labels_r = ax_model_r.get_legend_handles_labels()
        ax_model.legend(lines_l + lines_r, labels_l + labels_r,
                        fontsize=fontsize_legend, loc="lower center",
                        bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

        # === Right: Ground Truth ===
        ax_gt.plot(A_values, gt_sig_max, "-o", markersize=4, label=r"max $\sigma$", color="#c24c4c")
        ax_gt.set_title("Ground Truth (Maxwell)", fontsize=fontsize_title, fontweight="bold")
        ax_gt.set_xlabel("Test Amplitude $A$", fontsize=fontsize_label)
        ax_gt.set_ylabel(r"max $\sigma$", fontsize=fontsize_label, color="#c24c4c")
        ax_gt.tick_params(axis="y", labelsize=fontsize_tick, colors="#c24c4c")
        ax_gt.tick_params(axis="x", labelsize=fontsize_tick)
        ax_gt.set_ylim(sig_ylim)
        ax_gt.grid(True, alpha=0.3)

        ax_gt_r = ax_gt.twinx()
        ax_gt_r.plot(A_values, gt_d2ee, "-s", markersize=4,
                     label=rf"$\partial^2 e / \partial\varepsilon^2 = E_\infty + E$", color="#436384")
        ax_gt_r.plot(A_values, gt_d2eg, "-^", markersize=4,
                     label=rf"$\partial^2 e / \partial\varepsilon\,\partial\gamma = -E$", color="#16a48a")
        ax_gt_r.set_ylabel("Derivatives", fontsize=fontsize_label, color="#436384")
        ax_gt_r.tick_params(axis="y", labelsize=fontsize_tick, colors="#436384")
        ax_gt_r.set_ylim(deriv_ylim)

        # Legend outside below
        lines_l, labels_l = ax_gt.get_legend_handles_labels()
        lines_r, labels_r = ax_gt_r.get_legend_handles_labels()
        ax_gt.legend(lines_l + lines_r, labels_l + labels_r,
                     fontsize=fontsize_legend, loc="lower center",
                     bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

        fig.suptitle(f"Amplitude Response at $\\omega = {omega}$",
                     fontsize=fontsize_title + 1, fontweight="bold", y=1.02)
        fig.subplots_adjust(bottom=0.22)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        plt.show()
