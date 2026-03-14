"""Experiment configurations for viscoelasticity models."""

from dataclasses import dataclass, field
from typing import Any, Literal
import numpy as np


# =============================================================================
# Material Parameters (fixed for all experiments)
# =============================================================================
MATERIAL_PARAMS = {
    "E_infty": 0.5,
    "E": 2.0,
    "eta": 1.0,
}


# =============================================================================
# Model Types
# =============================================================================
ModelType = Literal["simple_rnn", "maxwell", "maxwell_nn", "gsm"]

ALL_MODELS: list[ModelType] = ["simple_rnn", "maxwell", "maxwell_nn", "gsm"]
TRAINABLE_MODELS: list[ModelType] = ["simple_rnn", "maxwell_nn", "gsm"]


# =============================================================================
# Experiment Configuration
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str

    # Load cases: list of (A, omega) tuples
    train_loadcases: list[tuple[float, float]]
    test_loadcases: list[tuple[float, float]]

    # Time discretization
    n_timesteps: int = 100  # points per period
    n_periods: int = 1  # number of periods for training

    # Training parameters
    train_steps: int = 20_000
    log_every: int = 100

    # Models to run (default: all trainable)
    models: list[ModelType] = field(default_factory=lambda: TRAINABLE_MODELS.copy())

    # Include relaxation test
    test_relaxation: bool = True


# =============================================================================
# Predefined Experiments
# =============================================================================

# Baseline: Train on (1,1), test on all
BASELINE = ExperimentConfig(
    name="baseline",
    description="Train on (A=1, omega=1), test on all load cases",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Multi-loadcase training
MULTI_LOADCASE = ExperimentConfig(
    name="multi_loadcase",
    description="Train on multiple load cases for better generalization",
    train_loadcases=[(1.0, 1.0), (1.0, 2.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# All loadcases for training
ALL_LOADCASES = ExperimentConfig(
    name="all_loadcases",
    description="Train on all load cases",
    train_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Fine timesteps (higher resolution)
FINE_TIMESTEPS = ExperimentConfig(
    name="fine_timesteps",
    description="Higher time resolution (n=200)",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_timesteps=200,
)

# Coarse timesteps (lower resolution)
COARSE_TIMESTEPS = ExperimentConfig(
    name="coarse_timesteps",
    description="Lower time resolution (n=50)",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_timesteps=50,
)

# Multiple periods for training
MULTI_PERIOD = ExperimentConfig(
    name="multi_period",
    description="Train on 3 periods instead of 1",
    train_loadcases=[(1.0, 1.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
    n_periods=3,
)

# Different training loadcase
TRAIN_ON_OMEGA2 = ExperimentConfig(
    name="train_omega2",
    description="Train on (A=1, omega=2) instead of (1,1)",
    train_loadcases=[(1.0, 2.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)

# Higher amplitude training
TRAIN_HIGH_AMPLITUDE = ExperimentConfig(
    name="train_high_amplitude",
    description="Train on (A=2, omega=3)",
    train_loadcases=[(2.0, 3.0)],
    test_loadcases=[(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)],
)


# =============================================================================
# Experiment Collections
# =============================================================================

# Core experiments (answer the main questions from the task)
CORE_EXPERIMENTS = [
    BASELINE,
    MULTI_LOADCASE,
    ALL_LOADCASES,
]

# Timestep experiments
TIMESTEP_EXPERIMENTS = [
    COARSE_TIMESTEPS,
    BASELINE,  # n=100 as reference
    FINE_TIMESTEPS,
]

# Training loadcase experiments
LOADCASE_EXPERIMENTS = [
    BASELINE,
    TRAIN_ON_OMEGA2,
    TRAIN_HIGH_AMPLITUDE,
]

# All experiments
ALL_EXPERIMENTS = [
    BASELINE,
    MULTI_LOADCASE,
    ALL_LOADCASES,
    FINE_TIMESTEPS,
    COARSE_TIMESTEPS,
    MULTI_PERIOD,
    TRAIN_ON_OMEGA2,
    TRAIN_HIGH_AMPLITUDE,
]
# =============================================================================
# Sweep Generation
# =============================================================================
def generate_sweep_configs(
    param_name: Literal["A", "omega", "n_timesteps", "train_steps", "train_loadcases"],
    min_val: float | None = None,
    max_val: float | None = None,
    n_steps: int | None = None,
    fixed_val: float = 1.0,
    fixed_loadcase: list[tuple[float, float]] = [(1.0, 1.0)],
    base_name: str = "sweep",
    train_steps: int = 100_000,
    n_timesteps: int = 100,
    sweep_values: list[Any] | None = None,
) -> list[ExperimentConfig]:
    """Generate a list of experiment configurations for a parameter sweep.

    Args:
        param_name: Parameter to sweep
        min_val: Minimum value (for numeric sweep)
        max_val: Maximum value (for numeric sweep)
        n_steps: Number of steps (for numeric sweep)
        fixed_val: Value for the non-swept numeric parameter (A/omega)
        fixed_loadcase: Base loadcase list [[(A, w), ...]] used when sweeping other params
        base_name: Base name for experiments
        train_steps: Default training steps
        n_timesteps: Default time discretization
        sweep_values: Explicit list of values to sweep over (overrides min/max/n_steps)

    Returns:
        List of ExperimentConfig objects
    """
    if sweep_values is not None:
        values = sweep_values
    else:
        if min_val is None or max_val is None or n_steps is None:
             raise ValueError("Must provide either sweep_values OR (min_val, max_val, n_steps)")
        values = np.linspace(min_val, max_val, n_steps)

    configs = []

    for i, val in enumerate(values):
        # Defaults
        current_train_steps = train_steps
        current_n_timesteps = n_timesteps
        current_train_loadcases = fixed_loadcase

        if param_name == "A":
            # For A sweep, we assume fixed_loadcase structure but replace A?
            # Or simpler: single loadcase (val, fixed_val)
            # To support multiple loadcases + sweeping A, it gets complex.
            # Fallback to simple behavior for backward compatibility:
            current_train_loadcases = [(float(val), fixed_val)]
            name = f"{base_name}_A_{val:.2f}"
            description = f"Sweep A={val:.2f}, omega={fixed_val}"
        
        elif param_name == "omega":
            current_train_loadcases = [(fixed_val, float(val))]
            name = f"{base_name}_w_{val:.2f}"
            description = f"Sweep A={fixed_val}, omega={val:.2f}"
        
        elif param_name == "n_timesteps":
            current_n_timesteps = int(val)
            name = f"{base_name}_ts_{current_n_timesteps}"
            description = f"Sweep n_timesteps={current_n_timesteps}"
        
        elif param_name == "train_steps":
            current_train_steps = int(val)
            name = f"{base_name}_steps_{current_train_steps}"
            description = f"Sweep train_steps={current_train_steps}"
            
        elif param_name == "train_loadcases":
            # Val is a list of tuples [(A, w), ...]
            current_train_loadcases = val
            name = f"{base_name}_lc_{i}"
            description = f"Sweep loadcases (config {i})"
        
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

        # Standard test set from BASELINE
        test_loadcases = [(1.0, 1.0), (1.0, 2.0), (2.0, 3.0)]

        config = ExperimentConfig(
            name=name,
            description=description,
            train_loadcases=current_train_loadcases,
            test_loadcases=test_loadcases,
            train_steps=current_train_steps,
            n_timesteps=current_n_timesteps,
        )
        configs.append(config)

    return configs
