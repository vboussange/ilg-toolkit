"""Penalties: functions of a distance model added to the training objective."""

import math
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import AbstractDistanceModel


def _floats(numbers: float | tuple[float, ...]) -> tuple[float, ...]:
    """Returns a tuple of numbers as a tuple of floats, and one number as a tuple of one."""
    return tuple(
        float(number) for number in (numbers if isinstance(numbers, tuple) else (numbers,))
    )


@final
class RidgePenalty(eqx.Module):
    r"""A ridge penalty: the negative log density of a Gaussian on the leaves a selector selects.

    For selected leaves $\theta_k$ with prior standard deviations $\sigma_k$ and prior
    means $\mu_k$, the penalty is $\sum_k \lVert\theta_k - \mu_k\rVert^2/(2\sigma_k^2)$,
    without the Gaussians' additive constants; each prior mean is broadcast over its
    leaf's entries. Training divides the penalty by the number of training pairs (see
    [`TrainingConfig`][ilg_toolkit.TrainingConfig]), so ``prior_sd`` means the same at
    any pair count. Ridge penalties with equal numbers and the same selector share the
    compiled update of [`train`][ilg_toolkit.train].

    Args:
        prior_sd: Standard deviation of the Gaussian, finite and positive: one number,
            alone or in a tuple of one, for every selected leaf, or a tuple with one
            number per leaf.
        where: Selector: a function returning one leaf of a distance model, an array, or
            a tuple of leaves, as in ``eqx.tree_at``; it reaches nested fields and slices.
        prior_mean: Mean of the Gaussian, finite: one number, alone or in a tuple of one,
            for every selected leaf, or a tuple with one number per leaf; 0 by default.

    Raises:
        ValueError: If a prior standard deviation is not finite and positive, or a prior
            mean is not finite.

    Examples:
        A ridge penalty on the coefficients of a log-linear patch encoder, with prior
        standard deviation 1.5, and its intercept pinned at 1 with prior standard
        deviation 0.01:

        >>> penalty = RidgePenalty(
        ...     prior_sd=(1.5, 0.01),
        ...     where=lambda model: (model.coefficients, model.intercepts),
        ...     prior_mean=(0.0, 1.0),
        ... )
        >>> config = TrainingConfig(loss=MLPELoss(), optimizer=optax.lbfgs(), penalty=penalty)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
    """

    prior_sd: tuple[float, ...] = eqx.field(converter=_floats)
    where: Callable[[Any], Any] = eqx.field(static=True)
    prior_mean: tuple[float, ...] = eqx.field(converter=_floats, default=0.0)

    def __check_init__(self):
        if not all(math.isfinite(sd) and sd > 0 for sd in self.prior_sd):
            raise ValueError("prior_sd must be finite and positive")
        if not all(math.isfinite(mean) for mean in self.prior_mean):
            raise ValueError("prior_mean must be finite")

    def __call__(self, distance_model: AbstractDistanceModel) -> jax.Array:
        """Returns the negative log density of the selected leaves, without the constants.

        Args:
            distance_model: Distance model holding the selected leaves.

        Returns:
            The scalar sum, over the selected leaves, of the squared differences between
            a leaf's entries and its prior mean, divided by ``2 * prior_sd**2`` for that
            leaf.

        Raises:
            ValueError: If ``prior_sd`` or ``prior_mean`` has several numbers, and not one
                per leaf the selector selects.
        """
        selection = self.where(distance_model)
        leaves = selection if isinstance(selection, tuple) else (selection,)
        for name, numbers in (("prior_sd", self.prior_sd), ("prior_mean", self.prior_mean)):
            if len(numbers) not in (1, len(leaves)):
                raise ValueError(
                    f"{name} has {len(numbers)} numbers for the {len(leaves)} leaves the "
                    "selector selects; give one number, or one per leaf"
                )

        def per_leaf(numbers: tuple[float, ...]) -> tuple[float, ...]:
            return numbers * len(leaves) if len(numbers) == 1 else numbers

        terms = (
            optax.tree.norm(jax.tree.map(lambda entries: entries - prior_mean, leaf), squared=True)
            / (2 * prior_sd**2)
            for leaf, prior_sd, prior_mean in zip(
                leaves, per_leaf(self.prior_sd), per_leaf(self.prior_mean), strict=True
            )
        )
        return jnp.asarray(sum(terms, 0.0))
