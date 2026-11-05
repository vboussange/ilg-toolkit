"""The Laplace approximation and the Newton step at a distance model trained with a penalty."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._checks import _is_finite_real, _require_float64
from ._training_utils import (
    _group_name,
    _group_names,
    _naming_failures,
    _prepare_training_inputs,
    _RegionSpeciesKey,
)
from .config import TrainingConfig
from .data import PairwiseObservations, RegionBatch
from .models import AbstractDistanceModel
from .training import TrainingState


@final
class LaplaceApproximation(eqx.Module):
    """A Laplace approximation: the Gaussian posterior approximation at a trained distance model.

    The posterior is that of training with a penalty on one region–species group. The
    parameters are the distance-model leaves a selector selects, followed by the group's
    statistical parameters. The Hessian is that of the summed training objective, on
    whose scale its inverse is the posterior covariance.

    Attributes:
        where: Selector of the distance-model leaves, as given to
            [`laplace_approximation`][ilg_toolkit.laplace_approximation].
        parameter_names: Name of each parameter, in order: the path of a selected leaf in
            the distance model with the entry's index, such as ``coefficients[0, 1]``,
            then the path of a statistical parameter in the region–species group's
            statistical parameter state, such as ``raw_variances[0]`` of
            [`MLPELoss`][ilg_toolkit.MLPELoss].
        hessian: Hessian of the summed training objective, shape ``(n, n)`` for ``n``
            parameters, symmetrized.
        covariance: Inverse of the Hessian, shape ``(n, n)``.
        standard_deviations: Square roots of the covariance's diagonal, shape ``(n,)``;
            not-a-number when the Hessian is not positive definite.
        positive_definite: Whether the Hessian is finite and positive definite.
    """

    where: Callable[[Any], Any] = eqx.field(static=True)
    parameter_names: tuple[str, ...]
    hessian: jax.Array
    covariance: jax.Array
    standard_deviations: jax.Array
    positive_definite: bool


def summed_training_objective(
    region: RegionBatch,
    observations: PairwiseObservations,
    distance_model: AbstractDistanceModel,
    loss_state: Any,
    *,
    config: TrainingConfig,
) -> jax.Array:
    """Returns the summed loss of one region–species group plus the penalty.

    For one region–species group, the summed training objective is the number of
    pairs times the training objective. On this scale, the inverse of its Hessian at a
    distance model trained with a penalty is the covariance of the Laplace approximation.

    Args:
        region: Region batch holding the observed sampling units.
        observations: Pairwise genetic observations of the region–species group.
        distance_model: Distance model, evaluated in inference mode.
        loss_state: Statistical parameter state, as in ``TrainingState.loss_state``.
        config: Training configuration holding the loss and the penalty.

    Returns:
        The scalar summed training objective, not-a-number when the maximum-likelihood
        population effects (MLPE) loss rejects the distances.

    Raises:
        ValueError: If the inputs do not form one region–species group of the distance
            model, the statistical parameter state does not hold that group, or the loss
            rejects the observations.
        RuntimeError: If float64 is disabled.
    """
    _region_species_group(region, observations, distance_model, "The summed training objective")
    summed = config.loss.sum_over_pairs(region, observations, distance_model, loss_state)
    if config.penalty is None:
        return summed
    return summed + jnp.asarray(config.penalty(distance_model), jnp.float64)


def laplace_approximation(
    region: RegionBatch,
    observations: PairwiseObservations,
    state: TrainingState,
    *,
    where: Callable[[Any], Any],
    step: float = 1e-4,
) -> LaplaceApproximation:
    """Computes the Laplace approximation at the latest model of a training state.

    The parameters are the distance-model leaves the selector selects, followed by the
    statistical parameters of the training state's region–species group. Leaves the
    selector does not select stay fixed at their values.

    The Hessian comes from central differences of the exact gradient of the summed
    training objective, and is symmetrized. The MLPE loss profiles its generalized least
    squares coefficients, so its gradient is exact but ``jax.hessian`` holds those
    coefficients fixed. One compiled gradient serves the ``2n`` evaluations for ``n``
    parameters, which run one after another.

    Args:
        region: Region batch of the training state's region–species group.
        observations: Pairwise genetic observations the training state was trained on.
        state: Training state on one region–species group, such as one trained with a
            penalty and ``optax.lbfgs()`` to a gradient tolerance.
        where: Selector: a function returning one leaf of a distance model, an array, or
            a tuple of leaves, as in ``eqx.tree_at``.
        step: Step of the central differences, in units of the parameters, finite and
            positive. A resistance solver's error reaches the Hessian divided by the step,
            so conjugate gradient needs tolerances of 1e-10 or tighter for the default.

    Returns:
        The Laplace approximation.

    Raises:
        ValueError: If ``step`` is not finite and positive, or the inputs do not form the
            one region–species group the training state was trained on.
        RuntimeError: If float64 is disabled, or a solve fails at a perturbed point, with
            the error ``train`` raises for a failed solve.
    """
    _require_float64()
    if not _is_finite_real(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    group = _trained_group(region, observations, state, "The Laplace approximation")
    gradient, parameters = _summed_gradient(region, observations, state, where, group)
    compiled = eqx.filter_jit(gradient)
    distance_model = state.latest_model.distance_model
    with _naming_failures(distance_model.species, (group,), "Laplace approximation"):
        # A loop, not `vmap`: a sparse direct solver factorizes one surface at a time.
        columns = [
            (compiled(parameters + step * unit) - compiled(parameters - step * unit)) / (2 * step)
            for unit in jnp.eye(parameters.size)
        ]
        hessian = jax.block_until_ready(jnp.stack(columns, axis=1))
    hessian = (hessian + hessian.T) / 2
    covariance = jnp.linalg.inv(hessian)
    positive_definite = bool(
        jnp.all(jnp.isfinite(hessian)) and jnp.all(jnp.linalg.eigvalsh(hessian) > 0)
    )
    standard_deviations = (
        jnp.sqrt(jnp.diag(covariance)) if positive_definite else jnp.full(parameters.size, jnp.nan)
    )
    return LaplaceApproximation(
        where,
        _parameter_names(
            distance_model, where, state.config.loss._group_state(state.loss_state, group)
        ),
        hessian,
        covariance,
        standard_deviations,
        positive_definite,
    )


def newton_step(
    region: RegionBatch,
    observations: PairwiseObservations,
    state: TrainingState,
    approximation: LaplaceApproximation,
) -> jax.Array:
    """Returns the Newton step at the latest model of a training state.

    The Newton step is ``-H⁻¹g``: the update of Newton's method, with ``g`` the gradient
    of the summed training objective over the parameters of the Laplace approximation,
    at the training state's latest model and statistical parameter state, and ``H`` the
    approximation's Hessian. In standard deviations of the Laplace approximation, the
    Newton step that remains shows whether training reached an optimum. The Laplace
    approximation may come from training on other pairs, as a proxy: a training state of
    a population-holdout fold can use the approximation from all pairs.

    The gradient is not compiled as a whole, and its compiled parts take the pairs as
    arguments, so a later call on as many pairs of the same region compiles nothing.

    Args:
        region: Region batch of the training state's region–species group.
        observations: Pairwise genetic observations of the region–species group, such as
            those the training state was trained on.
        state: Training state on one region–species group.
        approximation: Laplace approximation whose selector and Hessian the Newton step
            uses.

    Returns:
        The Newton step, shape ``(n,)``, in the order of the approximation's
        ``parameter_names``.

    Raises:
        ValueError: If the inputs do not form the one region–species group the training
            state was trained on, or the training state has another number of parameters
            than the approximation.
        RuntimeError: If float64 is disabled, or a solve fails, with the error ``train``
            raises for a failed solve.
    """
    _require_float64()
    group = _trained_group(region, observations, state, "The Newton step")
    gradient, parameters = _summed_gradient(region, observations, state, approximation.where, group)
    if approximation.hessian.shape != (parameters.size, parameters.size):
        raise ValueError(
            f"The Laplace approximation has {len(approximation.parameter_names)} parameters, "
            f"and the training state has {parameters.size}"
        )
    with _naming_failures(state.latest_model.distance_model.species, (group,), "Newton step"):
        value = jax.block_until_ready(gradient(parameters))
    return -jnp.linalg.solve(approximation.hessian, value)


def _region_species_group(
    region: RegionBatch,
    observations: PairwiseObservations,
    distance_model: AbstractDistanceModel,
    computation: str,
) -> _RegionSpeciesKey:
    """Returns the one region–species group of the inputs of a computation."""
    groups = _prepare_training_inputs(region, observations, distance_model.species).training
    if len(groups) != 1:
        names = _group_names(distance_model.species, [group.key for group in groups])
        raise ValueError(
            f"{computation} covers one region–species group, and the inputs form "
            f"{len(groups)}: {names}"
        )
    return groups[0].key


def _trained_group(
    region: RegionBatch,
    observations: PairwiseObservations,
    state: TrainingState,
    computation: str,
) -> _RegionSpeciesKey:
    """Returns the one region–species group of the inputs, the training state's group."""
    distance_model = state.latest_model.distance_model
    group = _region_species_group(region, observations, distance_model, computation)
    if state.training_groups != (group,):
        trained = _group_names(distance_model.species, state.training_groups)
        raise ValueError(
            f"{computation} covers the one region–species group of its training state: "
            f"the inputs form {_group_name(distance_model.species, group)}, and the "
            f"training state was trained on {trained}"
        )
    return group


def _summed_gradient(
    region: RegionBatch,
    observations: PairwiseObservations,
    state: TrainingState,
    where: Callable[[Any], Any],
    group: _RegionSpeciesKey,
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """Returns the gradient of the summed training objective over the flat parameters.

    The parameters are the selected leaves of the training state's latest model, then
    the group's statistical parameters.
    """
    distance_model = state.latest_model.distance_model
    loss = state.config.loss
    group_state = loss._group_state(state.loss_state, group)
    parameters, unravel = ravel_pytree((where(distance_model), group_state))

    def objective(flat: jax.Array) -> jax.Array:
        leaves, perturbed_state = unravel(flat)
        model = eqx.tree_at(where, distance_model, leaves)
        loss_state = (
            state.loss_state
            if group_state is None
            else eqx.tree_at(
                lambda current: loss._group_state(current, group),
                state.loss_state,
                perturbed_state,
            )
        )
        return summed_training_objective(
            region, observations, model, loss_state, config=state.config
        )

    return jax.grad(objective), parameters


def _parameter_names(
    distance_model: AbstractDistanceModel, where: Callable[[Any], Any], group_state: Any
) -> tuple[str, ...]:
    """Returns the name of each entry of the selected leaves and of the group's state."""
    paths = jax.tree_util.tree_map_with_path(
        lambda path, _: jax.tree_util.keystr(path), distance_model
    )
    leaves = zip(jax.tree.leaves(where(paths)), jax.tree.leaves(where(distance_model)), strict=True)
    statistical = (
        (jax.tree_util.keystr(path), leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(group_state)
    )
    return tuple(
        path.removeprefix(".") + (f"[{', '.join(map(str, index))}]" if index else "")
        for path, leaf in (*leaves, *statistical)
        for index in np.ndindex(np.shape(leaf))
    )
