"""Training and effective-resistance layer settings."""

import math
from dataclasses import dataclass
from numbers import Real

from ._checks import _positive_integer
from .losses import AbstractLoss


@dataclass(frozen=True)
class ResistanceSolverConfig:
    """Float64 solver settings for the effective-resistance layer.

    Conjugate gradient is the default solver; ``use_amg=True`` enables algebraic
    multigrid (AMG) preconditioning through the optional ``amg`` extra.
    The preconditioner is prepared before solving.
    Exhausting ``max_steps`` raises an error. ``rtol`` and ``atol`` default to
    1e-6 and ``max_steps`` to 1000; at least one tolerance must be positive.
    Tighter tolerances increase solve work and can expose conditioning limits.
    Settings belong to the conductance model, independently of training budgets.

    Attributes:
        rtol: Relative residual tolerance for the resistance solve.
        atol: Absolute residual tolerance for the resistance solve.
        max_steps: Maximum conjugate-gradient iterations.
        use_amg: Enable the optional algebraic multigrid preconditioner.

    Notes:
        AMG preparation is specific to graph shape and must be rebuilt for a new
        region shape; it does not fix poorly chosen ecological boundaries.
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
        object.__setattr__(self, "max_steps", _positive_integer("solver max_steps", self.max_steps))
        if not isinstance(self.use_amg, bool):
            raise ValueError("solver use_amg must be a boolean")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization budget and scientific settings for training a landscape genetics model.

    Each Adam update samples up to ``region_batch_size`` regions and then
    ``species_batch_size`` species within each region; ``None`` uses all available.
    ``species_microbatch_size`` controls how many species are processed together
    before their gradients are combined into one update.
    Evaluation at step zero, every ``eval_every`` updates, and the final step
    determines model selection.

    Attributes:
        loss: Statistical criterion and its numerical settings.
        steps: Total update budget, including completed updates when resuming.
        eval_every: Number of updates between evaluations.
        learning_rate: Adam learning rate.
        region_batch_size: Maximum regions sampled per update; ``None`` uses all.
        species_batch_size: Maximum species sampled per region; ``None`` uses all.
        species_microbatch_size: Maximum species processed together before their
            gradients are combined; ``None`` processes all sampled species together.
        seed: Seed for training's random sampling.

    Notes:
        Training always compiles its numerical calculations. Debug them eagerly
        with ``jax.disable_jit()``.
    """

    loss: AbstractLoss
    steps: int = 100
    eval_every: int = 1
    learning_rate: float = 0.001
    region_batch_size: int | None = None
    species_batch_size: int | None = None
    species_microbatch_size: int | None = None
    seed: int = 0

    def __post_init__(self):
        if not isinstance(self.loss, AbstractLoss):
            raise ValueError("loss must be an AbstractLoss")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        object.__setattr__(self, "eval_every", _positive_integer("eval_every", self.eval_every))
        for name in ("region_batch_size", "species_batch_size", "species_microbatch_size"):
            object.__setattr__(
                self, name, _positive_integer(name, getattr(self, name), optional=True)
            )
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, Real)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
