"""Data generation for task 3."""

import numpy as np


def harmonic_data(E_infty, E, eta, n, omega, A):
    """
    Solution of the generalized Maxwell model, using the explicit Euler scheme.

    n: total amount of time steps
    periods: number of periods
    amplitude: amplitude of the oscillation


    E_infty: stiffness of equilibrium spring
    E: stiffness of non-equilibrium spring
    eta: viscosity of damper

    """

    t = np.linspace(0, 2 * np.pi, n)

    eps = A * np.sin(omega * t)
    eps_dot = A * omega * np.cos(omega * t)
    sig = np.zeros_like(eps)
    gamma = np.zeros_like(eps)

    dt = 2 * np.pi / (n - 1)
    dts = np.ones_like(eps) * dt

    for i in range(len(t) - 1):
        gamma[i + 1] = gamma[i] + dt * E / eta * (eps[i] - gamma[i])
        sig[i + 1] = E_infty * eps[i + 1] + E * (eps[i + 1] - gamma[i + 1])

    return eps, eps_dot, sig, dts


def relaxation_data(E_infty, E, eta, n, omega, A):
    t = np.linspace(0, 2 * np.pi, n)

    n1 = int(np.round(n / 4.0 / omega))

    eps = A * np.sin(omega * t[0:n1])
    eps_dot = A * omega * np.cos(omega * t[0:n1])

    eps = np.concatenate([eps, A * np.sin(omega * t[n1]) * np.ones(n - n1)])
    eps_dot = np.concatenate([eps_dot, np.cos(omega * t[n1]) * np.zeros(n - n1)])

    sig = np.zeros_like(eps)
    gamma = np.zeros_like(eps)

    dt = 2 * np.pi / (n - 1)
    dts = np.ones_like(eps) * dt

    for i in range(len(t) - 1):
        gamma[i + 1] = gamma[i] + dt * E / eta * (eps[i] - gamma[i])
        sig[i + 1] = E_infty * eps[i + 1] + E * (eps[i + 1] - gamma[i + 1])

    return eps, eps_dot, sig, dts


def generate_data_harmonic(E_infty, E, eta, n, omegas, As):
    eps = []
    eps_dot = []
    sig = []
    dts = []

    for i in range(len(omegas)):
        eps2, eps_dot2, sig2, dts2 = harmonic_data(E_infty, E, eta, n, omegas[i], As[i])

        eps.append(eps2)
        eps_dot.append(eps_dot2)
        sig.append(sig2)
        dts.append(dts2)

    eps = np.vstack(eps)
    eps_dot = np.vstack(eps_dot)
    sig = np.vstack(sig)
    dts = np.vstack(dts)

    return eps, eps_dot, sig, dts


def generate_data_relaxation(E_infty, E, eta, n, omegas, As):
    eps = []
    eps_dot = []
    sig = []
    dts = []

    for i in range(len(omegas)):
        eps2, eps_dot2, sig2, dts2 = relaxation_data(
            E_infty, E, eta, n, omegas[i], As[i]
        )

        eps.append(eps2)
        eps_dot.append(eps_dot2)
        sig.append(sig2)
        dts.append(dts2)

    eps = np.vstack(eps)
    eps_dot = np.vstack(eps_dot)
    sig = np.vstack(sig)
    dts = np.vstack(dts)

    return eps, eps_dot, sig, dts

def add_noise_eps(
    eps: np.ndarray,
    noise_std: float = 0.0,
    noise_std_rel: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
    clip: float | None = None,
) -> np.ndarray:
    """
    Add noise to epsilon.

    Parameters
    ----------
    eps : array (N,T) or (T,)
        Clean strain.
    noise_std : float
        Absolute Gaussian std-dev (same units as eps).
    noise_std_rel : float
        Relative std-dev as fraction of per-case amplitude:
        std_i = noise_std_rel * (max(eps_i) - min(eps_i))/2
    noise_type : {"gaussian"}
        Currently gaussian only (easy to extend).
    seed : int or None
        RNG seed.
    clip : float or None
        If set, clip eps_noisy to [-clip, clip].

    Returns
    -------
    eps_noisy : array same shape as eps
    """
    eps = np.asarray(eps)
    rng = np.random.default_rng(seed)

    # compute per-case scale if eps is batched
    if eps.ndim == 2:
        amp = 0.5 * (eps.max(axis=1) - eps.min(axis=1))  # (N,)
        std_rel = noise_std_rel * amp                     # (N,)
        std_rel = std_rel[:, None]                        # (N,1)
    else:
        amp = 0.5 * (eps.max() - eps.min())
        std_rel = noise_std_rel * amp

    std = noise_std + std_rel

    if noise_type.lower() == "gaussian":
        noise = rng.normal(loc=0.0, scale=1.0, size=eps.shape) * std
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    eps_noisy = eps + noise

    if clip is not None:
        eps_noisy = np.clip(eps_noisy, -clip, clip)

    return eps_noisy

def eps_dot_from_eps(eps: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """
    Compute eps_dot from eps using finite differences.

    Uses a simple first-order scheme:
        eps_dot[n] = (eps[n+1]-eps[n]) / dt[n]
    and repeats the last value for length consistency.

    Shapes:
        eps:  (N,T) or (T,)
        dts:  (N,T) or (T,)
    """
    eps = np.asarray(eps)
    dts = np.asarray(dts)

    if eps.shape != dts.shape:
        raise ValueError(f"eps and dts must have same shape, got {eps.shape} vs {dts.shape}")

    if eps.ndim == 1:
        out = np.zeros_like(eps)
        out[:-1] = (eps[1:] - eps[:-1]) / dts[:-1]
        out[-1] = out[-2]
        return out

    # batched
    out = np.zeros_like(eps)
    out[:, :-1] = (eps[:, 1:] - eps[:, :-1]) / dts[:, :-1]
    out[:, -1] = out[:, -2]
    return out

def generate_data_harmonic_noisy_eps(
    E_infty, E, eta, n, omegas, As,
    *,
    noise_std: float = 0.0,
    noise_std_rel: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
    recompute_eps_dot_from_noisy: bool = False,
    clip: float | None = None,
    return_clean_eps: bool = False,
):
    """
    Generate harmonic data (clean physics), then corrupt eps for model input.

    Returns same tuple structure as generate_data_harmonic:
        eps_used, eps_dot_used, sig_true, dts

    Important:
    - sig_true is always generated from the clean model (ground truth).
    - eps_used is eps_noisy if noise_std/std_rel > 0, otherwise equals eps_clean.
    - eps_dot_used:
        - if recompute_eps_dot_from_noisy=True: derived from eps_used and dts
        - else: uses analytical eps_dot from the clean signal (as in your current generator)

    If return_clean_eps=True, returns:
        eps_used, eps_dot_used, sig_true, dts, eps_clean
    """
    # 1) Generate clean data
    eps_clean, eps_dot_clean, sig_true, dts = generate_data_harmonic(
        E_infty, E, eta, n, omegas, As
    )

    # 2) Add noise to eps (for inputs)
    eps_used = add_noise_eps(
        eps_clean,
        noise_std=noise_std,
        noise_std_rel=noise_std_rel,
        noise_type=noise_type,
        seed=seed,
        clip=clip,
    )

    # 3) Decide eps_dot used
    if recompute_eps_dot_from_noisy:
        eps_dot_used = eps_dot_from_eps(eps_used, dts)
    else:
        eps_dot_used = eps_dot_clean

    if return_clean_eps:
        return eps_used, eps_dot_used, sig_true, dts, eps_clean

    return eps_used, eps_dot_used, sig_true, dts

def generate_data_relaxation_noisy_eps(
    E_infty, E, eta, n, omegas, As,
    *,
    noise_std: float = 0.0,
    noise_std_rel: float = 0.0,
    noise_type: str = "gaussian",
    seed: int | None = None,
    recompute_eps_dot_from_noisy: bool = False,
    clip: float | None = None,
    return_clean_eps: bool = False,
):
    """
    Generate relaxation data (clean physics), then corrupt eps for model input.

    Returns same tuple structure as generate_data_relaxation:
        eps_used, eps_dot_used, sig_true, dts

    Notes
    -----
    - sig_true is always generated from the clean model (ground truth).
    - eps_used is eps_noisy if noise_std/std_rel > 0, otherwise equals eps_clean.
    - eps_dot_used:
        - if recompute_eps_dot_from_noisy=True: derived from eps_used and dts
        - else: uses eps_dot from the clean signal (as in generate_data_relaxation)

    If return_clean_eps=True, returns:
        eps_used, eps_dot_used, sig_true, dts, eps_clean
    """
    # 1) Generate clean relaxation data (ground truth)
    eps_clean, eps_dot_clean, sig_true, dts = generate_data_relaxation(
        E_infty, E, eta, n, omegas, As
    )

    # 2) Add noise to eps (for model input)
    eps_used = add_noise_eps(
        eps_clean,
        noise_std=noise_std,
        noise_std_rel=noise_std_rel,
        noise_type=noise_type,
        seed=seed,
        clip=clip,
    )

    # 3) Decide eps_dot used
    if recompute_eps_dot_from_noisy:
        eps_dot_used = eps_dot_from_eps(eps_used, dts)
    else:
        eps_dot_used = eps_dot_clean

    if return_clean_eps:
        return eps_used, eps_dot_used, sig_true, dts, eps_clean

    return eps_used, eps_dot_used, sig_true, dts
