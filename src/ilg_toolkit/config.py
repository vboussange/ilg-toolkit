"""Training configuration and resistance solver settings."""

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Literal

import optax

from ._checks import _positive_integer
from .losses import AbstractLoss


@dataclass(frozen=True)
class ResistanceSolverConfig:
    """Settings of the float64 resistance solver of the effective-resistance layer.

    ``solver`` selects one of three resistance solvers:

    - ``"cg"`` runs conjugate gradient (CG) on a CPU or a GPU, with no optional
      dependency. Choose it for small conductance grid graphs with low conductance
      contrast. Its iterations grow with both grid size and conductance contrast.
    - ``"amg"`` runs CG preconditioned with algebraic multigrid (AMG) on a CPU or a
      GPU, through the ``amg`` extra. Its hierarchy is prepared once on a uniform
      conductance grid graph. Choose it for larger conductance grid graphs with
      conductance contrast below 4 orders of magnitude. Its iterations grow with
      conductance contrast and not with grid size.
    - ``"cudss"`` runs the NVIDIA CUDA Direct Sparse Solver (cuDSS) on an NVIDIA GPU,
      through the ``cudss`` extra. Choose it on an NVIDIA GPU when the factorization
      fits in GPU memory, in particular to train a deep patch encoder, whose
      conductance surface can reach its floor. It has no tolerances or iteration
      limit, and its cost does not depend on conductance contrast.

    ``"cg"`` and ``"amg"`` raise an error when they reach ``max_steps`` before either
    tolerance.

    A conductance model carries its solver settings. Constructing, saving and
    loading them work on any backend. Preparing ``"cudss"`` raises an ``ImportError``
    or a ``RuntimeError`` naming the ``cudss`` extra without spineax or a CUDA-enabled
    GPU backend. No other resistance solver replaces it.

    Args:
        solver: The resistance solver: ``"cg"`` (default), ``"amg"`` or ``"cudss"``.
        rtol: Relative residual tolerance of ``"cg"`` and ``"amg"``, finite and
            nonnegative; ``None`` (default) selects 1e-6.
        atol: Absolute residual tolerance of ``"cg"`` and ``"amg"``, finite and
            nonnegative; ``None`` (default) selects 1e-6. At least one of ``rtol``
            and ``atol`` is positive.
        max_steps: Maximum number of CG iterations of ``"cg"`` and ``"amg"``;
            ``None`` (default) selects 1000.

    Raises:
        ValueError: If a setting is invalid, ``solver`` is unknown, or ``"cudss"``
            receives ``rtol``, ``atol`` or ``max_steps``.
    """

    solver: Literal["cg", "amg", "cudss"] = "cg"
    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None

    def __post_init__(self):
        if self.solver not in ("cg", "amg", "cudss"):
            raise ValueError(f"solver must be one of 'cg', 'amg' or 'cudss', got {self.solver!r}")
        if self.solver == "cudss":
            for name in ("rtol", "atol", "max_steps"):
                if getattr(self, name) is not None:
                    raise ValueError(f"{name} does not apply to solver 'cudss'; leave it as None")
            return
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            value = 1e-6 if value is None else value
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
        max_steps = 1000 if self.max_steps is None else self.max_steps
        object.__setattr__(self, "max_steps", _positive_integer("solver max_steps", max_steps))


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration: loss, optimizer, update budget and batching settings.

    Each update samples up to ``region_batch_size`` regions, then up to
    ``species_batch_size`` species within each sampled region, and processes
    ``species_microbatch_size`` species at a time before combining their gradients.
    Training evaluates at step zero, every ``eval_every`` updates and at the final
    step, and selects the model among these evaluations.

    The optimizer receives, with each update, the objective value, its gradient and
    a function evaluating the same objective (same regions, species and random key),
    so line-search optimizers such as ``optax.lbfgs()`` work through
    [`train`][ilg_toolkit.train]. Configuration equality ignores the optimizer.
    Training compiles its update for each optimizer object, so reusing one
    optimizer object across folds, runs and checkpoint loads reuses the compiled
    update.

    Args:
        loss: Loss of every region–species group.
        steps: Total number of updates, including completed updates when continuing;
            with ``gradient_tolerance``, the maximum number of updates.
        eval_every: Number of updates between evaluations.
        optimizer: Optax optimizer applying the updates; Adam with learning rate
            1e-3 by default.
        gradient_tolerance: Gradient-norm threshold ending training: training stops
            after the first update whose gradient norm (global L2 norm over the
            trainable parameters for that update's objective) is at or below it;
            ``None`` runs all ``steps``.
        region_batch_size: Maximum number of regions sampled per update; ``None``
            samples every region.
        species_batch_size: Maximum number of species sampled per region; ``None``
            samples every species.
        species_microbatch_size: Maximum number of species processed together;
            ``None`` processes all sampled species of a region together.
        seed: Seed of the random sampling and random layers, in ``[0, 2**32)``.

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
            if (
                isinstance(self.gradient_tolerance, bool)
                or not isinstance(self.gradient_tolerance, Real)
                or not math.isfinite(self.gradient_tolerance)
                or self.gradient_tolerance < 0
            ):
                raise ValueError("gradient_tolerance must be finite and nonnegative, or None")
            object.__setattr__(self, "gradient_tolerance", float(self.gradient_tolerance))
