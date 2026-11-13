"""Training configuration and resistance solver settings."""

from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Literal

import optax

from ._checks import _is_finite_real, _positive_integer
from .losses import AbstractLoss


@dataclass(frozen=True)
class ResistanceSolverConfig:
    """Settings of the float64 resistance solver of the effective-resistance layer.

    ``solver`` selects one of four resistance solvers:

    - ``"multigrid"`` (default) runs conjugate gradient (CG) preconditioned with
      geometric multigrid on a CPU or a GPU, with no optional dependency. Each solve
      builds its hierarchy from the conductances it solves. Choose it on a CPU, and on
      a GPU without the ``cudss`` extra or when the cuDSS factorization does not fit
      in GPU memory. Its iterations grow on conductance surfaces that vary by orders
      of magnitude from one vertex to the next.
    - ``"cudss"`` runs the NVIDIA CUDA Direct Sparse Solver (cuDSS) on an NVIDIA GPU,
      through the ``cudss`` extra. Choose it on an NVIDIA GPU when the factorization
      fits in GPU memory, in particular to train a deep patch encoder, whose
      conductance surface can reach its floor. It has no tolerances or iteration
      limit, and its cost does not depend on conductance contrast.
    - ``"cg"`` runs CG with no preconditioner on a CPU or a GPU, with no optional
      dependency. Its iterations grow with both grid size and conductance contrast.
    - ``"amg"`` runs CG preconditioned with algebraic multigrid (AMG) on a CPU or a
      GPU, through the ``amg`` extra. Its hierarchy is prepared once on a uniform
      conductance grid graph, and its iterations grow with conductance contrast.

    ``"cg"``, ``"amg"`` and ``"multigrid"`` raise an error when they reach
    ``max_steps`` before their tolerance.

    A conductance model carries its solver settings. Constructing, saving and
    loading them work on any backend. Preparing ``"cudss"`` raises an ``ImportError``
    or a ``RuntimeError`` naming the ``cudss`` extra without spineax or a CUDA-enabled
    GPU backend. No other resistance solver replaces it.

    Args:
        solver: The resistance solver: ``"multigrid"`` (default), ``"cudss"``,
            ``"cg"`` or ``"amg"``.
        rtol: Relative residual tolerance of ``"cg"``, ``"amg"`` and ``"multigrid"``,
            finite and nonnegative, and positive for ``"multigrid"``; ``None``
            (default) selects 1e-6.
        atol: Absolute residual tolerance of ``"cg"`` and ``"amg"``, finite and
            nonnegative; ``None`` (default) selects 1e-6. At least one of ``rtol``
            and ``atol`` is positive.
        max_steps: Maximum number of CG iterations of ``"cg"``, ``"amg"`` and
            ``"multigrid"``; ``None`` (default) selects 1000.

    Raises:
        ValueError: If a setting is invalid, ``solver`` is unknown, ``"multigrid"``
            receives ``atol``, or ``"cudss"`` receives ``rtol``, ``atol`` or
            ``max_steps``.
    """

    solver: Literal["cg", "amg", "multigrid", "cudss"] = "multigrid"
    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None

    def __post_init__(self):
        if self.solver not in ("cg", "amg", "multigrid", "cudss"):
            raise ValueError(
                f"solver must be one of 'cg', 'amg', 'multigrid' or 'cudss', got {self.solver!r}"
            )
        unused = {"cudss": ("rtol", "atol", "max_steps"), "multigrid": ("atol",)}
        for name in unused.get(self.solver, ()):
            if getattr(self, name) is not None:
                raise ValueError(
                    f"{name} does not apply to solver {self.solver!r}; leave it as None"
                )
        if self.solver == "cudss":
            return
        tolerances = ("rtol",) if self.solver == "multigrid" else ("rtol", "atol")
        for name in tolerances:
            value = getattr(self, name)
            value = 1e-6 if value is None else value
            if not _is_finite_real(value) or value < 0:
                raise ValueError(f"solver {name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if not any(getattr(self, name) for name in tolerances):
            raise ValueError("At least one solver tolerance must be positive")
        max_steps = 1000 if self.max_steps is None else self.max_steps
        object.__setattr__(self, "max_steps", _positive_integer("solver max_steps", max_steps))


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
    one optimizer object across folds and runs reuses the compiled update.

    The training objective is the aggregate loss plus ``penalty`` divided by the
    number of training pairs across all training region–species groups. The penalty
    enters once per update, whichever regions and species the update samples, and a
    prior's negative log density keeps its meaning at any pair count. Configuration
    equality covers the penalty: two separately written functions compare unequal,
    and ``dataclasses.replace(config, steps=...)`` keeps the same object. Training
    compiles its update for each penalty object, so reusing one penalty object across
    folds and runs reuses the compiled update.

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
        penalty: Function or Equinox module of the distance model returning a scalar,
            such as [`RidgePenalty`][ilg_toolkit.RidgePenalty]; ``None`` trains on the
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
    penalty: Callable[[Any], Any] | None = None

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
