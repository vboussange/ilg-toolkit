"""Penalties: functions of a distance model added to the training objective."""

import math
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .models import AbstractDistanceModel


def _floats(numbers: float | tuple[float, ...]) -> float | tuple[float, ...]:
    """Returns one number as a float, and a tuple of numbers as a tuple of floats."""
    if isinstance(numbers, tuple):
        return tuple(float(number) for number in numbers)
    return float(numbers)


def _as_tuple(numbers: float | tuple[float, ...], count: int = 1) -> tuple[float, ...]:
    """Returns a tuple of numbers unchanged, and one number repeated ``count`` times."""
    return numbers if isinstance(numbers, tuple) else (numbers,) * count


@final
class RidgePenalty(eqx.Module):
    r"""A ridge penalty: the negative log density of a Gaussian on selected leaves.

    For selected leaves $\theta_k$ with prior standard deviations $\sigma_k$ and means
    $\mu_k$, the penalty is $\sum_k \lVert\theta_k - \mu_k\rVert^2/(2\sigma_k^2)$, without
    the Gaussians' additive constants; each mean is broadcast over its leaf's entries.
    Training divides the penalty by the number of training pairs (see
    [`TrainingConfig`][ilg_toolkit.TrainingConfig]), so ``prior_sd`` means the same at
    any pair count. Ridge penalties with equal numbers and the same ``where`` function
    share the compiled update of [`train`][ilg_toolkit.train].

    Args:
        prior_sd: Standard deviation of the Gaussian, finite and positive: one number
            for every selected leaf, or a tuple with one number per leaf.
        where: Function returning the selected leaves of a distance model, one leaf or a
            tuple of leaves, as in ``eqx.tree_at``; it reaches nested fields and slices.
        mean: Mean of the Gaussian, finite: one number for every selected leaf, or a
            tuple with one number per leaf; 0 by default.

    Raises:
        ValueError: If a standard deviation is not finite and positive, a mean is not
            finite, or ``prior_sd`` and ``mean`` are tuples of different lengths.

    Examples:
        A ridge penalty on the coefficients of a log-linear patch encoder, with prior
        standard deviation 1.5, and its intercept pinned at 1 with standard deviation
        0.01:

        >>> penalty = RidgePenalty(
        ...     prior_sd=(1.5, 0.01),
        ...     where=lambda model: (model.coefficients, model.intercepts),
        ...     mean=(0.0, 1.0),
        ... )
        >>> config = TrainingConfig(loss=MLPELoss(), optimizer=optax.lbfgs(), penalty=penalty)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
    """

    prior_sd: float | tuple[float, ...] = eqx.field(converter=_floats)
    where: Callable[[Any], Any] = eqx.field(static=True)
    mean: float | tuple[float, ...] = eqx.field(converter=_floats, default=0.0)

    def __check_init__(self):
        if not all(math.isfinite(sd) and sd > 0 for sd in _as_tuple(self.prior_sd)):
            raise ValueError("prior_sd must be finite and positive")
        if not all(math.isfinite(mean) for mean in _as_tuple(self.mean)):
            raise ValueError("mean must be finite")
        if (
            isinstance(self.prior_sd, tuple)
            and isinstance(self.mean, tuple)
            and len(self.prior_sd) != len(self.mean)
        ):
            raise ValueError(
                f"prior_sd has {len(self.prior_sd)} numbers and mean has {len(self.mean)}; "
                "tuples of both must have the same length"
            )

    def __call__(self, distance_model: AbstractDistanceModel) -> jax.Array:
        """Returns the negative log density of the selected leaves, without the constants.

        Args:
            distance_model: Distance model holding the selected leaves.

        Returns:
            The scalar sum, over the selected leaves, of the squared differences between
            a leaf's entries and its mean, divided by ``2 * prior_sd**2`` for that leaf.

        Raises:
            ValueError: If ``prior_sd`` or ``mean`` is a tuple whose length differs from
                the number of leaves ``where`` selects.
        """
        selection = self.where(distance_model)
        leaves = selection if isinstance(selection, tuple) else (selection,)
        lengths = {
            name: len(numbers)
            for name, numbers in (("prior_sd", self.prior_sd), ("mean", self.mean))
            if isinstance(numbers, tuple)
        }
        if any(length != len(leaves) for length in lengths.values()):
            raise ValueError(
                f"The number of leaves where selects, {len(leaves)}, differs from the length "
                f"of {' and '.join(lengths)}, {next(iter(lengths.values()))}"
            )
        terms = (
            optax.tree.norm(jax.tree.map(lambda entries: entries - mean, leaf), squared=True)
            / (2 * prior_sd**2)
            for leaf, prior_sd, mean in zip(
                leaves,
                _as_tuple(self.prior_sd, len(leaves)),
                _as_tuple(self.mean, len(leaves)),
                strict=True,
            )
        )
        return jnp.asarray(sum(terms, 0.0))
