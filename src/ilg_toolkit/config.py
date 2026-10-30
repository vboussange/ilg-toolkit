"""Training configuration and effective-resistance solver settings."""

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import optax

from ._checks import _is_finite_real, _positive_integer
from .losses import AbstractLoss


@dataclass(frozen=True)
class ResistanceSolverConfig:
    """Settings of the float64 solver in the effective-resistance layer.

    The solver runs conjugate gradient, optionally preconditioned with algebraic
    multigrid (AMG) through the ``amg`` extra, and raises an error when it reaches
    ``max_steps`` before either tolerance. A conductance model carries its solver
    settings.

    Args:
        rtol: Relative residual tolerance, finite and nonnegative.
        atol: Absolute residual tolerance, finite and nonnegative; at least one of
            ``rtol`` and ``atol`` is positive.
        max_steps: Maximum number of conjugate-gradient iterations.
        use_amg: Whether to precondition with AMG.

    Raises:
        ValueError: If a setting is invalid.
    """

    rtol: float = 1e-6
    atol: float = 1e-6
    max_steps: int = 1000
    use_amg: bool = False

    def __post_init__(self):
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if not _is_finite_real(value) or value < 0:
                raise ValueError(f"solver {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.rtol == 0 and self.atol == 0:
            raise ValueError("At least one solver tolerance must be positive")
        object.__setattr__(self, "max_steps", _positive_integer("solver max_steps", self.max_steps))
        if not isinstance(self.use_amg, bool):
            raise ValueError("solver use_amg must be a boolean")


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration: loss, optimizer, update budget, batching settings and penalty.

    Each update samples up to ``region_batch_size`` regions, then up to
    ``species_batch_size`` species within each sampled region, and processes
    ``species_microbatch_size`` species at a time before combining their gradients.
    Training evaluates at step zero, every ``eval_every`` updates and at the final
    step, and selects the model among these evaluations.

    The optimizer receives, with each update, the training objective's value, its
    gradient and a function evaluating the same training objective (same regions,
    species and random key), so line-search optimizers such as ``optax.lbfgs()``
    work through [`train`][ilg_toolkit.train]. Configuration equality ignores the
    optimizer. Training compiles its update for each optimizer object, so reusing
    one optimizer object across folds, runs and checkpoint loads reuses the compiled
    update.

    The training objective is the aggregate loss plus ``penalty`` divided by the
    number of training pairs across all training region–species groups. The penalty
    enters once per update, whichever regions and species the update samples, and a
    prior's negative log density keeps its meaning at any pair count. Configuration
    equality covers the penalty: two separately written functions compare unequal,
    and ``dataclasses.replace(config, steps=...)`` keeps the same object. Training
    compiles its update for each penalty object, so reusing one penalty object across
    folds, runs and checkpoint loads reuses the compiled update. Changing a penalty
    object's attributes after its first use does not change the compiled update.

    Args:
        loss: Loss of every region–species group.
        steps: Total number of updates, including completed updates when continuing;
            with ``gradient_tolerance``, the maximum number of updates.
        eval_every: Number of updates between evaluations.
        optimizer: Optax optimizer applying the updates; Adam with learning rate
            1e-3 by default.
        gradient_tolerance: Gradient-norm threshold ending training: training stops
            after the first update whose gradient norm (global L2 norm over the
            trainable parameters for that update's training objective) is at or below
            it; ``None`` runs all ``steps``.
        region_batch_size: Maximum number of regions sampled per update; ``None``
            samples every region.
        species_batch_size: Maximum number of species sampled per region; ``None``
            samples every species.
        species_microbatch_size: Maximum number of species processed together;
            ``None`` processes all sampled species of a region together.
        seed: Seed of the random sampling and random layers, in ``[0, 2**32)``.
        penalty: Callable of the distance model returning a scalar, such as
            [`RidgePenalty`][ilg_toolkit.RidgePenalty]; ``None`` trains on the
            aggregate loss alone.

    Raises:
        ValueError: If a setting is invalid.
    """

    loss: AbstractLoss
    steps: int = 100
    eval_every: int = 1
    optimizer: optax.GradientTransformation = field(default=optax.adam(1e-3), compare=False)
    gradient_tolerance: float | None = None
    region_batch_size: int | None = None
    species_batch_size: int | None = None
    species_microbatch_size: int | None = None
    seed: int = 0
    # Excluded from the hash: a penalty module holding arrays is unhashable.
    penalty: Callable[[Any], Any] | None = field(default=None, hash=False)

    def __post_init__(self):
        if not isinstance(self.loss, AbstractLoss):
            raise ValueError("loss must be an AbstractLoss")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, Integral)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be an integer in [0, 2**32)")
        object.__setattr__(self, "seed", int(self.seed))
        if isinstance(self.steps, bool) or not isinstance(self.steps, Integral) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "eval_every", _positive_integer("eval_every", self.eval_every))
        for name in ("region_batch_size", "species_batch_size", "species_microbatch_size"):
            object.__setattr__(
                self, name, _positive_integer(name, getattr(self, name), optional=True)
            )
        if not isinstance(self.optimizer, optax.GradientTransformation):
            raise ValueError("optimizer must be an optax.GradientTransformation")
        if self.gradient_tolerance is not None:
            if not _is_finite_real(self.gradient_tolerance) or self.gradient_tolerance < 0:
                raise ValueError("gradient_tolerance must be finite and nonnegative, or None")
            object.__setattr__(self, "gradient_tolerance", float(self.gradient_tolerance))
        if self.penalty is not None and not callable(self.penalty):
            raise ValueError("penalty must be callable or None")
