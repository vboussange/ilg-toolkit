"""The Laplace approximation and the Newton step at a penalized fit of one region–species group."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ._checks import _is_finite_real, _require_float64
from ._training_utils import _group_name, _prepare_training_inputs, _RegionSpeciesKey
from .config import TrainingConfig
from .data import PairwiseObservations, RegionBatch
from .models import AbstractDistanceModel
from .training import _naming_failures, TrainingState


@final
class LaplaceApproximation(eqx.Module):
    """A Laplace approximation: the Gaussian approximation of the posterior at a penalized fit.

    The parameters are the distance-model leaves a selector picks, followed by the
    statistical parameters of one region–species group. The Hessian is that of the
    summed training objective, on whose scale its inverse is the posterior covariance.

    Attributes:
        parameter_names: Name of each parameter, in order: the path of a selected leaf in
            the distance model with the element's index, such as ``coefficients[0, 1]``,
            then the path of a statistical parameter in the region–species group's state.
            The maximum-likelihood population effects (MLPE) loss has ``raw_variances[0]``
            and ``raw_variances[1]``: the population and residual variances are each the
            loss's ``variance_floor`` plus the softplus of their parameter.
        hessian: Hessian of the summed training objective, shape ``(n, n)`` for ``n``
            parameters, symmetrized.
        covariance: Inverse of the Hessian, shape ``(n, n)``.
        standard_deviations: Square roots of the covariance's diagonal, shape ``(n,)``;
            not-a-number when the Hessian is not positive definite.
        positive_definite: Whether the Hessian is finite and positive definite.
    """

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
    penalized fit is the covariance of the Laplace approximation.

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

    The parameters are the distance-model leaves ``where`` selects, followed by the
    statistical parameters of the training state's region–species group. Leaves that
    ``where`` does not select stay fixed at their values.

    The Hessian comes from central differences of the exact gradient of the summed
    training objective, and is symmetrized. The MLPE loss profiles its generalized least
    squares coefficients, so its gradient is exact but ``jax.hessian`` holds those
    coefficients fixed. One compiled gradient serves the ``2n`` evaluations for ``n``
    parameters, which run one after another.

    Args:
        region: Region batch of the training state's region–species group.
        observations: Pairwise genetic observations the training state was trained on.
        state: Training state on one region–species group, such as a penalized fit
            with ``optax.lbfgs()`` to a gradient tolerance.
        where: Function returning the selected leaves of a distance model, as in
            ``eqx.tree_at``.
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

    Examples:
        The Laplace approximation over the coefficients of a log-linear patch encoder,
        with its intercept fixed, and the Newton step that remains at the fit, in
        standard deviations:

        >>> where = lambda model: model.coefficients
        >>> approximation = laplace_approximation(region, observations, state, where=where)
        >>> step = newton_step(region, observations, state.latest_model.distance_model,
        ...                    state.loss_state, config=state.config, where=where,
        ...                    hessian=approximation.hessian)
        >>> float(jnp.max(jnp.abs(step) / approximation.standard_deviations)) < 0.01
        True
    """
    _require_float64()
    if not _is_finite_real(step) or step <= 0:
        raise ValueError("step must be finite and positive")
    distance_model = state.latest_model.distance_model
    group = _region_species_group(region, observations, distance_model, "The Laplace approximation")
    if state.training_groups != (group,):
        trained = "; ".join(
            _group_name(distance_model.species, trained) for trained in state.training_groups
        )
        raise ValueError(
            "The Laplace approximation covers the one region–species group of its training "
            f"state: the inputs form {_group_name(distance_model.species, group)}, and the "
            f"training state was trained on {trained}"
        )
    gradient, parameters = _summed_gradient(
        region, observations, distance_model, state.loss_state, state.config, where, group
    )
    compiled = eqx.filter_jit(gradient)
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
    distance_model: AbstractDistanceModel,
    loss_state: Any,
    *,
    config: TrainingConfig,
    where: Callable[[Any], Any],
    hessian: jax.Array,
) -> jax.Array:
    """Returns the Newton step at a distance model and statistical parameter state.

    The Newton step is ``-H⁻¹g``: the update of Newton's method, with ``g`` the gradient
    of the summed training objective over the parameters of the Laplace approximation
    and ``H`` a Hessian over the same parameters. At the latest model of a training
    state, the Newton step that remains, in standard deviations of the Laplace
    approximation, shows whether training reached an optimum. The Hessian may come from
    a fit on other pairs, as a proxy, such as the Laplace approximation from all pairs at
    the fit of a population-holdout fold.

    The gradient is not compiled as a whole, and its compiled parts take the pairs as
    arguments, so a later call on as many pairs of the same region compiles nothing.

    Args:
        region: Region batch holding the observed sampling units.
        observations: Pairwise genetic observations of one region–species group.
        distance_model: Distance model, evaluated in inference mode.
        loss_state: Statistical parameter state holding the region–species group, as in
            ``TrainingState.loss_state``.
        config: Training configuration holding the loss and the penalty.
        where: Function returning the selected leaves of a distance model, as in
            ``eqx.tree_at``; the same as the Laplace approximation's.
        hessian: Hessian over the parameters, shape ``(n, n)``, such as
            ``LaplaceApproximation.hessian``.

    Returns:
        The Newton step, shape ``(n,)``, in the order of the Laplace approximation's
        ``parameter_names``.

    Raises:
        ValueError: If the inputs form other than one region–species group, or the
            Hessian does not have one row and one column per parameter.
        RuntimeError: If float64 is disabled, or a solve fails, with the error ``train``
            raises for a failed solve.
    """
    _require_float64()
    group = _region_species_group(region, observations, distance_model, "The Newton step")
    gradient, parameters = _summed_gradient(
        region, observations, distance_model, loss_state, config, where, group
    )
    hessian = jnp.asarray(hessian)
    if hessian.shape != (parameters.size, parameters.size):
        raise ValueError(
            f"hessian must have shape {(parameters.size, parameters.size)}, one row and one "
            f"column per parameter, got {hessian.shape}"
        )
    with _naming_failures(distance_model.species, (group,), "Newton step"):
        value = jax.block_until_ready(gradient(parameters))
    return -jnp.linalg.solve(hessian, value)


def _region_species_group(
    region: RegionBatch,
    observations: PairwiseObservations,
    distance_model: AbstractDistanceModel,
    computation: str,
) -> _RegionSpeciesKey:
    """Returns the one region–species group of the inputs of a computation."""
    groups = _prepare_training_inputs(region, observations, distance_model.species).training
    if len(groups) != 1:
        names = "; ".join(_group_name(distance_model.species, group.key) for group in groups)
        raise ValueError(
            f"{computation} covers one region–species group, and the inputs form "
            f"{len(groups)}: {names}"
        )
    return groups[0].key


def _summed_gradient(
    region: RegionBatch,
    observations: PairwiseObservations,
    distance_model: AbstractDistanceModel,
    loss_state: Any,
    config: TrainingConfig,
    where: Callable[[Any], Any],
    group: _RegionSpeciesKey,
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """Returns the gradient of the summed training objective over the flat parameters.

    The parameters are the selected leaves, then the group's statistical parameters,
    at the distance model and statistical parameter state.
    """
    loss = config.loss
    group_state = loss._group_state(loss_state, group)
    parameters, unravel = ravel_pytree((where(distance_model), group_state))

    def objective(flat: jax.Array) -> jax.Array:
        leaves, perturbed_state = unravel(flat)
        model = eqx.tree_at(where, distance_model, leaves)
        perturbed = (
            loss_state
            if group_state is None
            else eqx.tree_at(
                lambda state: loss._group_state(state, group), loss_state, perturbed_state
            )
        )
        return summed_training_objective(region, observations, model, perturbed, config=config)

    return jax.grad(objective), parameters


def _parameter_names(
    distance_model: AbstractDistanceModel, where: Callable[[Any], Any], group_state: Any
) -> tuple[str, ...]:
    """Returns the name of each element of the selected leaves and of the group's state."""
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
