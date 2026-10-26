"""Training and resistance calculation settings."""

import math
from dataclasses import dataclass
from numbers import Integral, Real

from .losses import AbstractLoss


def validate_patch_microbatch_size(value: int | None) -> None:
    """Validation of the optional number of raster patches processed together."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("patch_microbatch_size must be a positive integer or None")


@dataclass(frozen=True)
class ResistanceSolverConfig:
    """Float64 settings for resistance calculation.

    Conjugate gradient is the default solver; ``use_amg=True`` enables algebraic
    multigrid preconditioning through the optional ``amg`` extra.
    The preconditioner is prepared before resistance calculation.
    Exhausting ``max_steps`` raises an error.
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
class TrainingConfig:
    """Optimization budget and numerical settings for a landscape genetics model.

    Each Adam update samples up to ``region_batch_size`` regions and then
    ``species_batch_size`` species within each region; ``None`` uses all available.
    ``species_microbatch_size`` controls how many species are processed together
    before their gradients are combined into one update.
    Evaluation at step zero, every ``eval_every`` updates, and the final step
    determines model selection.
    """

    loss: AbstractLoss
    steps: int = 100
    eval_every: int = 1
    learning_rate: float = 0.001
    region_batch_size: int | None = None
    species_batch_size: int | None = None
    species_microbatch_size: int | None = None
    seed: int = 0
    jit: bool = True

    def __post_init__(self):
        if not isinstance(self.loss, AbstractLoss):
            raise ValueError("loss must be an AbstractLoss")
        if not isinstance(self.jit, bool):
            raise ValueError("jit must be a boolean")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if (
            isinstance(self.eval_every, bool)
            or not isinstance(self.eval_every, int)
            or self.eval_every < 1
        ):
            raise ValueError("eval_every must be a positive integer")
        for name in ("region_batch_size", "species_batch_size", "species_microbatch_size"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
