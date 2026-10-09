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


@dataclass(frozen=True)
class FitConfig:
    """Fixed training budget; validation only changes predictor selection.

    A fixed Adam learning rate is used. Epoch zero is included in the reported
    history and is eligible for validation selection. No test data is accepted.
    """

    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 0
    jit: bool = True
    solver: SolverConfig = field(default_factory=SolverConfig)

    def __post_init__(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
