"""Explicit optimization options without study configuration."""

import math
from dataclasses import dataclass, field
from numbers import Integral, Real


def validate_patch_batch_size(value: int | None) -> None:
    """Validate the optional encoder chunk size."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("patch_batch_size must be a positive integer or None")


@dataclass(frozen=True)
class SolverConfig:
    """Explicit float64 resistance-solver settings.

    Ordinary CG requires no AMG dependencies. ``use_amg=True`` requires the
    toolkit's optional ``amg`` extra and constructs its hierarchy on the host.
    Exhausting ``max_steps`` is an error; no fallback changes the calculation.
    """

    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 1000
    use_amg: bool = False

    def __post_init__(self):
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"solver {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.rtol == 0 and self.atol == 0:
            raise ValueError("At least one solver tolerance must be positive")
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, Integral)
            or self.max_steps <= 0
        ):
            raise ValueError("solver max_steps must be a positive integer")
        if not isinstance(self.use_amg, bool):
            raise ValueError("solver use_amg must be a boolean")
        object.__setattr__(self, "max_steps", int(self.max_steps))


@dataclass(frozen=True)
class FitConfig:
    """Fixed training budget; validation only changes predictor selection.

    A fixed Adam learning rate is used. Epoch zero is included in the reported
    history and is eligible for validation selection. No test data is accepted.
    MLPE uses separate regional variance parameters with the same Adam learning
    rate, an explicit floor/jitter and optional initial (population, residual)
    variances. Otherwise initialization uses training-target sample variance.
    """

    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 0
    jit: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)
    objective: str = "direct_log1p"
    mlpe_variance_floor: float = 1e-10
    mlpe_jitter: float = 0.0
    mlpe_initial_variances: tuple[float, float] | None = None

    def __post_init__(self):
        if self.objective not in {"direct_log1p", "mlpe"}:
            raise ValueError("objective must be direct_log1p or mlpe")
        if (
            isinstance(self.mlpe_variance_floor, bool)
            or not isinstance(self.mlpe_variance_floor, Real)
            or not math.isfinite(self.mlpe_variance_floor)
            or self.mlpe_variance_floor <= 0
        ):
            raise ValueError("mlpe_variance_floor must be finite and positive")
        if (
            isinstance(self.mlpe_jitter, bool)
            or not isinstance(self.mlpe_jitter, Real)
            or not math.isfinite(self.mlpe_jitter)
            or not 0 <= self.mlpe_jitter <= 1e-6
        ):
            raise ValueError("mlpe_jitter must be finite and in [0, 1e-6]")
        if self.mlpe_initial_variances is not None:
            values = tuple(self.mlpe_initial_variances)
            if len(values) != 2 or any(
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= self.mlpe_variance_floor
                for value in values
            ):
                raise ValueError(
                    "mlpe_initial_variances must be two finite variances above the floor"
                )
            object.__setattr__(self, "mlpe_initial_variances", tuple(float(v) for v in values))
        if not isinstance(self.jit, bool):
            raise ValueError("jit must be a boolean")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        for name in ("learning_rate", "mlpe_variance_floor", "mlpe_jitter"):
            object.__setattr__(self, name, float(getattr(self, name)))
