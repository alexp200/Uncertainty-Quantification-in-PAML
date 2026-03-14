"""Model implementations."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import he_normal
from jaxtyping import PRNGKeyArray
import klax


class Cell(eqx.Module):
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]

    def __init__(self, *, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = (
            klax.nn.Linear(3, 16, weight_init=he_normal(), key=k1),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=k2),
            klax.nn.Linear(16, 2, weight_init=he_normal(), key=k3),
        )

        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            lambda x: x,
        )

    def __call__(self, gamma, x):
        eps = x[0]
        h = x[1]

        x = jnp.array([gamma, eps, h])

        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))

        gamma = x[0]
        sig = x[1]

        return gamma, sig


class Model(eqx.Module):
    cell: Callable

    def __init__(self, *, key: PRNGKeyArray):
        self.cell = Cell(key=key)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build(*, key: PRNGKeyArray):
    """Make and return a Simple RNN model instance."""
    return Model(key=key)


# ===========================================================================
# Model 2a: Analytical Maxwell Model (not trainable)
# ===========================================================================

class MaxwellCell(eqx.Module):
    """Analytical Maxwell cell with fixed parameters.

    Uses explicit Euler to evolve the internal variable gamma.
    Not trainable - all parameters are fixed constants.
    """
    E_infty: float
    E_val: float
    eta: float

    def __init__(self, E_infty: float = 0.5, E_val: float = 2.0, eta: float = 1.0):
        self.E_infty = E_infty
        self.E_val = E_val
        self.eta = eta

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        # Evolution equation (explicit Euler)
        gamma_new = gamma + dt * (self.E_val / self.eta) * (eps - gamma)
        # Stress
        sig = self.E_infty * eps + self.E_val * (eps - gamma)

        return gamma_new, sig


class MaxwellModel(eqx.Module):
    """Analytical Maxwell model over a time series."""
    cell: MaxwellCell

    def __init__(self, E_infty: float = 0.5, E_val: float = 2.0, eta: float = 1.0):
        self.cell = MaxwellCell(E_infty=E_infty, E_val=E_val, eta=eta)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build_maxwell(E_infty: float = 0.5, E_val: float = 2.0, eta: float = 1.0):
    """Make and return an analytical Maxwell model instance."""
    return MaxwellModel(E_infty=E_infty, E_val=E_val, eta=eta)


# ===========================================================================
# Model 2b: Maxwell with trainable evolution equation
# ===========================================================================

class MaxwellNNCell(eqx.Module):
    """Maxwell cell where the evolution equation is learned by a FFNN.

    Energy and stress remain analytical:
        e(eps, gamma) = 0.5 * E_infty * eps^2 + 0.5 * E * (eps - gamma)^2
        sigma = E_infty * eps + E * (eps - gamma)

    Evolution equation is generalized:
        gamma_dot = f(eps, gamma) * (eps - gamma),  f > 0
    where f is represented by a FFNN with softplus output to ensure positivity.
    """
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]
    E_infty: float
    E_val: float

    def __init__(self, *, key: PRNGKeyArray, E_infty: float = 0.5, E_val: float = 2.0):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = (
            klax.nn.Linear(2, 16, weight_init=he_normal(), key=k1),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=k2),
            klax.nn.Linear(16, 1, weight_init=he_normal(), key=k3),
        )
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,  # Ensures f > 0
        )
        self.E_infty = E_infty
        self.E_val = E_val

    # Helper to return FFNN outputs for evaluation
    def f_theta(self, eps, gamma):
        """Compute f(eps, gamma) > 0 (FFNN output)."""
        x = jnp.array([eps, gamma])
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x[0]

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        f = self.f_theta(eps, gamma)

        gamma_dot = f * (eps - gamma)
        gamma_new = gamma + dt * gamma_dot

        sig = self.E_infty * eps + self.E_val * (eps - gamma)
        return gamma_new, sig



class MaxwellNNModel(eqx.Module):
    """Maxwell model with trainable evolution equation over a time series."""
    cell: MaxwellNNCell

    def __init__(self, *, key: PRNGKeyArray, E_infty: float = 0.5, E_val: float = 2.0):
        self.cell = MaxwellNNCell(key=key, E_infty=E_infty, E_val=E_val)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build_maxwell_nn(*, key: PRNGKeyArray, E_infty: float = 0.5, E_val: float = 2.0):
    """Make and return a Maxwell model with trainable evolution equation."""
    return MaxwellNNModel(key=key, E_infty=E_infty, E_val=E_val)


# ===========================================================================
# Model 3: GSM Model (Generalized Standard Materials)
# ===========================================================================

class GSMCell(eqx.Module):
    """GSM cell with learned energy function.

    Energy e(eps, gamma) is represented by a FFNN.
    Stress: sigma = de/d_eps (via autodiff)
    Evolution: gamma_dot = -g * de/d_gamma (via autodiff)
    with g = eta^{-1} = const > 0.

    Thermodynamically consistent by construction:
        D = g * (de/d_gamma)^2 >= 0
    """
    layers: tuple[Callable, ...]
    activations: tuple[Callable, ...]
    g: float  # g = 1/eta

    def __init__(self, *, key: PRNGKeyArray, g: float = 1.0):
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = (
            klax.nn.Linear(2, 16, weight_init=he_normal(), key=k1),
            klax.nn.Linear(16, 16, weight_init=he_normal(), key=k2),
            klax.nn.Linear(16, 1, weight_init=he_normal(), key=k3),
        )
        self.activations = (
            jax.nn.softplus,
            jax.nn.softplus,
            jax.nn.softplus,
        )
        self.g = g

    def _energy(self, eps, gamma):
        """Compute internal energy e(eps, gamma)."""
        x = jnp.array([eps, gamma])
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        return x[0]

    def __call__(self, gamma, x):
        eps = x[0]
        dt = x[1]

        # Stress: sigma = de/d_eps
        de_deps = jax.grad(self._energy, argnums=0)(eps, gamma)
        sig = de_deps

        # Evolution equation: gamma_dot = -g * de/d_gamma
        de_dgamma = jax.grad(self._energy, argnums=1)(eps, gamma)
        gamma_dot = -self.g * de_dgamma
        gamma_new = gamma + dt * gamma_dot

        return gamma_new, sig


class GSMModel(eqx.Module):
    """GSM model over a time series."""
    cell: GSMCell

    def __init__(self, *, key: PRNGKeyArray, g: float = 1.0):
        self.cell = GSMCell(key=key, g=g)

    def __call__(self, xs):
        def scan_fn(state, x):
            return self.cell(state, x)

        init_state = jnp.array(0.0)
        _, ys = jax.lax.scan(scan_fn, init_state, xs)
        return ys


def build_gsm(*, key: PRNGKeyArray, g: float = 1.0):
    """Make and return a GSM model instance."""
    return GSMModel(key=key, g=g)
