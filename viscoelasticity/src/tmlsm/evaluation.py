"""
evaluation.py

Modular evaluation utilities for viscoelastic models in this repository.

Design principles:
- Each function does ONE job.
- No hidden global state.
- Works with models defined in models.py (Simple RNN, Maxwell, Maxwell+NN, GSM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
from . import metrics as tm_metrics


# ============================================================================
# Data containers
# ============================================================================

@dataclass(frozen=True)
class Trajectory:
    """Container for a single loadcase trajectory."""
    eps: np.ndarray          # (T,)
    dts: np.ndarray          # (T,)
    sig: np.ndarray          # (T,)
    eps_dot: Optional[np.ndarray] = None  # (T,) if available


@dataclass(frozen=True)
class SimulationResult:
    """Model simulation output for one loadcase."""
    gamma: np.ndarray   # (T+1,) including initial gamma_0
    sig: np.ndarray     # (T,)


# ============================================================================
# Core simulation (crucial building block)
# ============================================================================

def simulate_model(model, eps, x2):
    eps_j = jnp.asarray(eps)
    x2_j  = jnp.asarray(x2)
    xs = jnp.stack([eps_j, x2_j], axis=1)  # (T,2)

    def scan_fn(state, x):
        return model.cell(state, x)

    init_state = jnp.array(0.0)
    _, ys = jax.lax.scan(scan_fn, init_state, xs)

    # For your models, ys is sigma history (T,) because scan returns (_, y)
    # But in our previous version we unpacked (gamma_new, sig). Let's do explicit:
    def scan_fn2(gamma, x):
        gamma_new, sig = model.cell(gamma, x)
        return gamma_new, (gamma_new, sig)

    _, (gamma_hist, sig_hist) = jax.lax.scan(scan_fn2, init_state, xs)
    gamma_full = jnp.concatenate([init_state[None], gamma_hist], axis=0)

    return gamma_full, sig_hist


def simulate_model_batch(model, eps_batch, x2_batch):
    def _one(eps, x2):
        return simulate_model(model, eps, x2)

    gamma_b, sig_b = jax.vmap(_one)(eps_batch, x2_batch)
    return np.array(gamma_b), np.array(sig_b)




# ============================================================================
# Error arrays + scalar metrics
# ============================================================================

def stress_error(sig_true: np.ndarray, sig_pred: np.ndarray) -> np.ndarray:
    """Pointwise error e(t) = sig_pred - sig_true."""
    return np.asarray(sig_pred) - np.asarray(sig_true)


def abs_stress_error(sig_true: np.ndarray, sig_pred: np.ndarray) -> np.ndarray:
    """Pointwise absolute error |e(t)|."""
    return np.abs(stress_error(sig_true, sig_pred))


def compute_metrics_per_case(sig_true_batch: np.ndarray, sig_pred_batch: np.ndarray) -> Dict[int, Dict[str, float]]:
    """
    Compute standard metrics per loadcase (index -> metrics dict).

    Shapes:
        sig_true_batch: (N, T)
        sig_pred_batch: (N, T)
    """
    out: Dict[int, Dict[str, float]] = {}
    for i in range(sig_true_batch.shape[0]):
        out[i] = tm_metrics.compute_all_metrics(sig_true_batch[i], sig_pred_batch[i])
    return out


def summarize_metrics(metrics_per_case: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Average metrics over loadcases."""
    keys = list(next(iter(metrics_per_case.values())).keys())
    return {k: float(np.mean([m[k] for m in metrics_per_case.values()])) for k in keys}


# ============================================================================
# Hysteresis / dissipation proxies from sigma-epsilon
# ============================================================================

def hysteresis_area(eps: np.ndarray, sig: np.ndarray) -> float:
    """
    Approximate loop area ∮ sigma d eps.

    For a full closed cycle (harmonic steady state), this is dissipated energy per cycle.
    For non-closed paths, it is net work over the path.
    """
    eps = np.asarray(eps)
    sig = np.asarray(sig)
    # numerical line integral: ∫ sigma d eps
    return float(np.trapz(sig, eps))


def hysteresis_area_batch(eps_batch: np.ndarray, sig_batch: np.ndarray) -> np.ndarray:
    """Compute hysteresis area per loadcase."""
    return np.array([hysteresis_area(eps_batch[i], sig_batch[i]) for i in range(eps_batch.shape[0])])


# ============================================================================
# Energy & dissipation for specific model families
# ============================================================================

def maxwell_energy(eps: np.ndarray, gamma: np.ndarray, E_infty: float, E: float) -> np.ndarray:
    """
    e(eps,gamma) = 0.5 E_infty eps^2 + 0.5 E (eps - gamma)^2
    gamma should be aligned with eps. If eps has length T, use gamma[0:T].
    """
    eps = np.asarray(eps)
    gamma = np.asarray(gamma)
    g = gamma[: len(eps)]
    return 0.5 * E_infty * eps**2 + 0.5 * E * (eps - g) ** 2


def maxwell_gamma_dot(eps: np.ndarray, gamma: np.ndarray, E: float, eta: float) -> np.ndarray:
    """
    gamma_dot = (E/eta) * (eps - gamma)
    (continuous form; discrete evaluation at time steps)
    """
    eps = np.asarray(eps)
    g = np.asarray(gamma)[: len(eps)]
    return (E / eta) * (eps - g)


def maxwell_dissipation_density(eps: np.ndarray, gamma: np.ndarray, E: float, eta: float) -> np.ndarray:
    """
    D = (E^2/eta) * (eps - gamma)^2  >= 0
    """
    eps = np.asarray(eps)
    g = np.asarray(gamma)[: len(eps)]
    return (E**2 / eta) * (eps - g) ** 2


# ---------------- Maxwell + NN: extract f_theta ----------------------------

def maxwell_nn_f(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Extract f_theta(eps,gamma) from MaxwellNNCell by re-running its MLP.
    This relies on the structure in your models.py: model.cell.layers + activations.

    Returns f evaluated at each time step n, aligned with eps_n and gamma_n.
    """
    cell = model.cell
    if not hasattr(cell, "layers") or not hasattr(cell, "activations"):
        raise ValueError("Model does not look like MaxwellNNModel (missing layers/activations).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    def f_one(e, g):
        x = jnp.array([e, g])
        for layer, act in zip(cell.layers, cell.activations):
            x = act(layer(x))
        return x[0]

    f_vals = jax.vmap(f_one)(eps_j, gamma_j)
    return np.array(f_vals)


def maxwell_nn_gamma_dot(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    gamma_dot = f_theta(eps,gamma) * (eps - gamma)
    """
    f = maxwell_nn_f(model, eps, gamma)
    g = np.asarray(gamma)[: len(eps)]
    return f * (np.asarray(eps) - g)


def maxwell_nn_dissipation_density(model: Any, eps: np.ndarray, gamma: np.ndarray, E: float) -> np.ndarray:
    """
    D = E * (eps - gamma) * gamma_dot = E * f * (eps - gamma)^2
    Requires E (non-equilibrium spring stiffness).
    """
    f = maxwell_nn_f(model, eps, gamma)
    g = np.asarray(gamma)[: len(eps)]
    return E * f * (np.asarray(eps) - g) ** 2


# ---------------- GSM: access learned energy + gradients --------------------

def gsm_energy(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Evaluate learned energy e_theta(eps,gamma) using model.cell._energy.
    """
    cell = model.cell
    if not hasattr(cell, "_energy"):
        raise ValueError("Model does not look like GSMModel (missing _energy).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    e_vals = jax.vmap(cell._energy)(eps_j, gamma_j)
    return np.array(e_vals)


def gsm_de_dgamma(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Compute de/dgamma along a trajectory."""
    cell = model.cell
    if not hasattr(cell, "_energy"):
        raise ValueError("Model does not look like GSMModel (missing _energy).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    de_dg_fn = jax.grad(cell._energy, argnums=1)
    vals = jax.vmap(de_dg_fn)(eps_j, gamma_j)
    return np.array(vals)


def gsm_dissipation_density(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    D = g * (de/dgamma)^2  >= 0
    """
    cell = model.cell
    if not hasattr(cell, "g"):
        raise ValueError("Model does not look like GSMModel (missing g).")

    de_dg = gsm_de_dgamma(model, eps, gamma)
    return float(cell.g) * de_dg**2


# ============================================================================
# Plotting helpers (modular)
# ============================================================================

from typing import Dict, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt


def plot_multi_model_predictions(
    eps_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_by_model: Dict[str, np.ndarray],
    omegas: Sequence[float],
    As: Sequence[float],
    title: str = "Model comparison",
    cases: Optional[Union[int, Sequence[int]]] = None,
    t: Optional[np.ndarray] = None,
) -> None:
    """
    Plot GT + multiple model predictions, but *one figure per loadcase*.

    Parameters
    ----------
    eps_batch : (N,T)
    sig_true_batch : (N,T)
    sig_pred_by_model : dict[name -> (N,T)]
        Each entry must have same shape as sig_true_batch.
    omegas, As : length N
        Metadata used in titles.
    title : str
        Base title prefix.
    cases : None | int | list[int]
        Which loadcases to plot.
        - None: plot all loadcases (one figure per case)
        - int: plot that case
        - list/tuple: plot those cases
    t : optional np.ndarray (T,)
        If None, uses linspace(0, 2π, T) (harmonic assumption).
        If you want true time, pass cumulative sum of dt externally.
    """
    eps_batch = np.asarray(eps_batch)
    sig_true_batch = np.asarray(sig_true_batch)

    n_cases, T = eps_batch.shape

    # Validate shapes
    if sig_true_batch.shape != (n_cases, T):
        raise ValueError(f"sig_true_batch must have shape {(n_cases, T)}, got {sig_true_batch.shape}")

    for name, sig_pred in sig_pred_by_model.items():
        sig_pred = np.asarray(sig_pred)
        if sig_pred.shape != (n_cases, T):
            raise ValueError(f"sig_pred_by_model['{name}'] must have shape {(n_cases, T)}, got {sig_pred.shape}")

    # Normalize cases argument
    if cases is None:
        case_list = list(range(n_cases))
    elif isinstance(cases, int):
        case_list = [cases]
    else:
        case_list = list(cases)

    # bounds check
    for c in case_list:
        if c < 0 or c >= n_cases:
            raise IndexError(f"case index {c} out of bounds for N={n_cases}")

    # Time axis
    if t is None:
        ts = np.linspace(0, 2 * np.pi, T)
    else:
        ts = np.asarray(t)
        if ts.shape != (T,):
            raise ValueError(f"t must have shape {(T,)}, got {ts.shape}")

    # Plot one figure per case
    for i in case_list:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{title} — case {i} (ω={omegas[i]}, A={As[i]})")

        # sigma vs time
        ax = axs[0]
        ax.plot(ts, sig_true_batch[i], "--", alpha=0.8, linewidth=2.0, label="GT")
        for name, sig_pred in sig_pred_by_model.items():
            ax.plot(ts, sig_pred[i], "-", linewidth=1.8, label=name)

        ax.set_xlabel("time t")
        ax.set_ylabel("stress σ")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        # sigma vs eps (hysteresis)
        ax = axs[1]
        ax.plot(eps_batch[i], sig_true_batch[i], "--", alpha=0.8, linewidth=2.0, label="GT")
        for name, sig_pred in sig_pred_by_model.items():
            ax.plot(eps_batch[i], sig_pred[i], "-", linewidth=1.8, label=name)

        ax.set_xlabel("strain ε")
        ax.set_ylabel("stress σ")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

        fig.tight_layout()
        plt.show()



def plot_error_vs_time(
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Stress error vs time",
) -> None:
    n_cases, T = sig_true_batch.shape
    ts = np.linspace(0, 2 * np.pi, T)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    for i in range(n_cases):
        err = sig_pred_batch[i] - sig_true_batch[i]
        ax.plot(ts, err, label=f"case {i}")

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("time t")
    ax.set_ylabel("σ_pred - σ_true")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_abs_error_vs_strain(
    eps_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Absolute stress error vs strain",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    n_cases = eps_batch.shape[0]
    for i in range(n_cases):
        err = np.abs(sig_pred_batch[i] - sig_true_batch[i])
        ax.scatter(eps_batch[i], err, s=8, alpha=0.6, label=f"case {i}")

    ax.set_xlabel("strain ε")
    ax.set_ylabel("|σ_pred - σ_true|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_abs_error_vs_strain_rate(
    eps_dot_batch: np.ndarray,
    sig_true_batch: np.ndarray,
    sig_pred_batch: np.ndarray,
    title: str = "Absolute stress error vs strain rate",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title)

    n_cases = eps_dot_batch.shape[0]
    for i in range(n_cases):
        err = np.abs(sig_pred_batch[i] - sig_true_batch[i])
        ax.scatter(eps_dot_batch[i], err, s=8, alpha=0.6, label=f"case {i}")

    ax.set_xlabel("strain rate ε̇")
    ax.set_ylabel("|σ_pred - σ_true|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_energy_and_dissipation(
    ts: np.ndarray,
    energy_by_model: Dict[str, np.ndarray],
    diss_by_model: Dict[str, np.ndarray],
    title: str = "Energy and dissipation",
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    ax = axs[0]
    for name, e in energy_by_model.items():
        ax.plot(ts, e, label=name)
    ax.set_xlabel("time t")
    ax.set_ylabel("energy e")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axs[1]
    for name, d in diss_by_model.items():
        ax.plot(ts, d, label=name)
    ax.set_xlabel("time t")
    ax.set_ylabel("dissipation density D")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()

def maxwell_nn_coeff_series(model_mnn, eps: np.ndarray, dts: np.ndarray):
    """
    Compute f_theta(t) along a trajectory by scanning the internal state gamma.

    Parameters
    ----------
    model_mnn : MaxwellNNModel (or any object with .cell.f_theta and .cell(...) )
    eps : (T,)
    dts : (T,)

    Returns
    -------
    gamma_full : (T+1,)  includes gamma_0 at index 0
    sig_hist   : (T,)
    f_hist     : (T,)    f_theta evaluated at (eps_n, gamma_n)
    """
    eps_j = jnp.asarray(eps)
    dts_j = jnp.asarray(dts)
    xs = jnp.stack([eps_j, dts_j], axis=1)  # (T,2)

    def scan_fn(gamma, x):
        eps_n = x[0]
        dt_n = x[1]

        f = model_mnn.cell.f_theta(eps_n, gamma)      # f at (eps_n, gamma_n)
        gamma_new, sig = model_mnn.cell(gamma, x)     # uses same update
        return gamma_new, (gamma_new, sig, f)

    gamma0 = jnp.array(0.0)
    _, (gamma_hist, sig_hist, f_hist) = jax.lax.scan(scan_fn, gamma0, xs)

    gamma_full = jnp.concatenate([gamma0[None], gamma_hist], axis=0)

    return np.array(gamma_full), np.array(sig_hist), np.array(f_hist)



################################### NEW Section #################################################

# ============================================================================
# Energy plots vs state (epsilon / gamma) + time reconstruction
# ============================================================================

from typing import Literal, Sequence, Optional


def time_from_dts(dts: np.ndarray, t0: float = 0.0) -> np.ndarray:
    """
    Reconstruct time axis from per-step dt.

    Parameters
    ----------
    dts : (T,)
        Time step sizes for each increment.
    t0 : float
        Initial time value.

    Returns
    -------
    ts : (T,)
        Time values aligned with eps[t], sigma[t], energy[t] (i.e., per step).
        We return ts as the "left" time at each step: ts[0]=t0, ts[k]=t0+sum_{i<k} dts[i].
    """
    dts = np.asarray(dts).reshape(-1)
    ts = np.empty_like(dts, dtype=float)
    t = float(t0)
    for k in range(len(dts)):
        ts[k] = t
        t += float(dts[k])
    return ts


def _prep_energy_state_xy(
    eps: np.ndarray,
    gamma: np.ndarray,
    e: np.ndarray,
    x_axis: Literal["eps", "gamma"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal: align shapes and pick x-axis.
    eps: (T,)
    gamma: (T+1,) or (T,)  -> we use gamma[:T]
    e: (T,)
    """
    eps = np.asarray(eps).reshape(-1)
    e = np.asarray(e).reshape(-1)
    if e.shape[0] != eps.shape[0]:
        raise ValueError(f"Energy length {e.shape[0]} must match eps length {eps.shape[0]}.")

    gamma = np.asarray(gamma).reshape(-1)
    if gamma.shape[0] == eps.shape[0] + 1:
        gT = gamma[: eps.shape[0]]
    elif gamma.shape[0] == eps.shape[0]:
        gT = gamma
    else:
        raise ValueError(
            f"gamma must have length T or T+1 (got {gamma.shape[0]} vs T={eps.shape[0]})."
        )

    x = eps if x_axis == "eps" else gT
    return x, e


def plot_energy_vs_state(
    eps: np.ndarray,
    gamma: np.ndarray,
    energy_by_model: Dict[str, np.ndarray],
    *,
    x_axis: Literal["eps", "gamma"] = "eps",
    mode: Literal["line_time_order", "scatter_time_colored"] = "line_time_order",
    dts: Optional[np.ndarray] = None,
    title: str = "Energy vs state",
    yscale: Optional[Literal["linear", "log"]] = "log",
    center_energy: bool = False,
) -> None:
    """
    Plot energy vs epsilon or vs gamma.

    Notes
    -----
    - Harmonic loading makes e(eps) multi-valued (loading/unloading branches).
      Use `mode="line_time_order"` (default) to preserve the trajectory loop,
      or `mode="scatter_time_colored"` to show the multivalued structure.
    - `center_energy=True` subtracts the mean energy of each curve (useful if
      the learned energy has an arbitrary additive offset).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.set_title(title)

    # time array only needed for scatter coloring
    ts = None
    if mode == "scatter_time_colored":
        if dts is not None:
            ts = time_from_dts(dts)
        else:
            # fall back: index-based time
            # (still fine for coloring; preserves ordering)
            ts = np.arange(len(eps), dtype=float)

    # plot each model curve
    for name, e in energy_by_model.items():
        x, y = _prep_energy_state_xy(eps, gamma, e, x_axis=x_axis)
        if center_energy:
            y = y - np.mean(y)

        if mode == "line_time_order":
            ax.plot(x, y, label=name)
        elif mode == "scatter_time_colored":
            sc = ax.scatter(x, y, c=ts, s=10, alpha=0.7, label=name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    ax.set_xlabel("strain ε" if x_axis == "eps" else "internal variable γ")
    ax.set_ylabel("energy e" + (" (centered)" if center_energy else ""))

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.legend(fontsize=8)

    if mode == "scatter_time_colored":
        # colorbar from the last scatter; OK for quick diagnostics
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("time / step")

    fig.tight_layout()
    plt.show()


def plot_energy_vs_eps_and_gamma(
    eps: np.ndarray,
    gamma: np.ndarray,
    energy_by_model: Dict[str, np.ndarray],
    *,
    dts: Optional[np.ndarray] = None,
    mode: Literal["line_time_order", "scatter_time_colored"] = "line_time_order",
    title_prefix: str = "Energy",
    yscale: Optional[Literal["linear", "log"]] = "log",
    center_energy: bool = False,
) -> None:
    """
    Convenience wrapper: makes 2 plots
      1) e vs eps
      2) e vs gamma
    """
    plot_energy_vs_state(
        eps,
        gamma,
        energy_by_model,
        x_axis="eps",
        mode=mode,
        dts=dts,
        title=f"{title_prefix} vs ε",
        yscale=yscale,
        center_energy=center_energy,
    )

    plot_energy_vs_state(
        eps,
        gamma,
        energy_by_model,
        x_axis="gamma",
        mode=mode,
        dts=dts,
        title=f"{title_prefix} vs γ",
        yscale=yscale,
        center_energy=center_energy,
    )


def plot_energy_vs_state_loadcases(
    eps_batch: np.ndarray,
    dts_batch: Optional[np.ndarray],
    gamma_batch: np.ndarray,
    energy_true_batch: np.ndarray,
    energy_pred_batch: np.ndarray,
    *,
    loadcase_labels: Optional[Sequence[str]] = None,
    which: Sequence[int] = (0,),
    mode: Literal["line_time_order", "scatter_time_colored"] = "line_time_order",
    yscale: Optional[Literal["linear", "log"]] = "log",
    center_energy: bool = False,
) -> None:
    """
    Plot energy vs eps/gamma for multiple loadcases.

    Parameters
    ----------
    eps_batch : (N,T)
    dts_batch : (N,T) or None
    gamma_batch : (N,T+1) or (N,T)
    energy_true_batch : (N,T)
    energy_pred_batch : (N,T)
    loadcase_labels : list of strings (len N) like "A=..., ω=..."
    which : subset of loadcase indices to plot
    """
    eps_batch = np.asarray(eps_batch)
    gamma_batch = np.asarray(gamma_batch)
    energy_true_batch = np.asarray(energy_true_batch)
    energy_pred_batch = np.asarray(energy_pred_batch)

    for i in which:
        label = f"case {i}"
        if loadcase_labels is not None and i < len(loadcase_labels):
            label = loadcase_labels[i]

        dts_i = None
        if dts_batch is not None:
            dts_i = np.asarray(dts_batch)[i]

        eps_i = eps_batch[i]
        gamma_i = gamma_batch[i]

        e_true_i = energy_true_batch[i]
        e_pred_i = energy_pred_batch[i]

        plot_energy_vs_eps_and_gamma(
            eps_i,
            gamma_i,
            {"True": e_true_i, "gsm": e_pred_i},
            dts=dts_i,
            mode=mode,
            title_prefix=f"Energy ({label})",
            yscale=yscale,
            center_energy=center_energy,
        )

# ---------------- GSM: teacher-forced stress (sigma) ------------------------

def gsm_stress(model: Any, eps: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Compute GSM stress sigma along a provided (eps(t), gamma(t)) trajectory:

        sigma(t) = d e_theta(eps(t), gamma(t)) / d eps

    This does NOT evolve gamma. It is the key building block for teacher forcing.

    Parameters
    ----------
    model : GSMModel (trained)
    eps : (T,)
    gamma : (T,) or (T+1,)
        If (T+1,), gamma[0:T] is used to align with eps[t].

    Returns
    -------
    sig : (T,)
    """
    cell = model.cell
    if not hasattr(cell, "_energy"):
        raise ValueError("Model does not look like GSMModel (missing _energy).")

    eps_j = jnp.asarray(eps)
    gamma_j = jnp.asarray(gamma[: len(eps)])

    de_deps_fn = jax.grad(cell._energy, argnums=0)
    sig = jax.vmap(de_deps_fn)(eps_j, gamma_j)
    return np.array(sig)


def gsm_stress_batch(model: Any, eps_batch: np.ndarray, gamma_batch: np.ndarray) -> np.ndarray:
    """
    Batch version of gsm_stress.

    Parameters
    ----------
    eps_batch : (N,T)
    gamma_batch : (N,T) or (N,T+1)

    Returns
    -------
    sig_batch : (N,T)
    """
    def _one(eps, gamma):
        return gsm_stress(model, eps, gamma)

    sig_b = jax.vmap(_one)(jnp.asarray(eps_batch), jnp.asarray(gamma_batch))
    return np.array(sig_b)


def gsm_teacher_forced_sigma(
    gsm_model: Any,
    eps_batch: np.ndarray,
    gamma_true_batch: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper: teacher-forced GSM sigma using gamma_true(t).

    This is exactly Option B:
        sigma_TF(t) = d e_theta(eps(t), gamma_true(t)) / d eps

    Returns (N,T).
    """
    return gsm_stress_batch(gsm_model, eps_batch, gamma_true_batch)
