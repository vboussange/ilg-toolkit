"""Penalties: functions of a distance model added to the training objective."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp

from ._checks import _is_finite_real
from .models import AbstractDistanceModel


@final
class RidgePenalty(eqx.Module):
    r"""A ridge penalty: the negative log density of a zero-mean Gaussian on selected leaves.

    For selected leaves $\theta$ and prior standard deviation $\sigma$, the penalty
    is $\lVert\theta\rVert^2/(2\sigma^2)$, without the Gaussian's additive constant.
    Training divides the penalty by the number of training pairs (see
    [`TrainingConfig`][ilg_toolkit.TrainingConfig]), so ``prior_sd`` means the same
    at any pair count.

    Args:
        prior_sd: Standard deviation of the Gaussian, finite and positive.
        where: Function returning the selected leaves of a distance model, as in
            ``eqx.tree_at``; it reaches nested fields and slices.

    Raises:
        ValueError: If ``prior_sd`` is not finite and positive, or ``where`` is not
            callable.

    Examples:
        A ridge penalty on the coefficients of a log-linear patch encoder:

        >>> penalty = RidgePenalty(prior_sd=1.5, where=lambda model: model.coefficients)
        >>> config = TrainingConfig(loss=MLPELoss(), optimizer=optax.lbfgs(), penalty=penalty)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
    """

    prior_sd: float
    where: Callable[[Any], Any] = eqx.field(static=True)

    def __init__(self, prior_sd: float, where: Callable[[Any], Any]):
        if not _is_finite_real(prior_sd) or prior_sd <= 0:
            raise ValueError("prior_sd must be finite and positive")
        if not callable(where):
            raise ValueError("where must be callable")
        self.prior_sd = float(prior_sd)
        self.where = where

    def __call__(self, distance_model: AbstractDistanceModel) -> jax.Array:
        """Returns the negative log density of the selected leaves, without the constant.

        Args:
            distance_model: Distance model holding the selected leaves.

        Returns:
            The scalar sum of the squared selected leaves divided by ``2 * prior_sd**2``.
        """
        leaves = jax.tree.leaves(self.where(distance_model))
        return jnp.asarray(sum(jnp.sum(leaf**2) for leaf in leaves)) / (2 * self.prior_sd**2)
