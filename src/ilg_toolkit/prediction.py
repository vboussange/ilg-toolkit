"""Pair-oriented genetic point predictions and explicit matrix conversion."""

from collections.abc import Sequence
from typing import cast, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import ObservationPartition, TargetSpec


@final
class Prediction(eqx.Module):
    """Original-scale point values aligned with ordered labelled pairs.

    Attributes:
        values: Floating JAX vector of shape ``(m,)``. Uncovered evaluation
            queries can be NaN; signed inverse-transformed MLPE points are valid.
        pairs: Distinct unordered identities, retaining caller order/orientation.
        region_name: Region qualifying every pair identity.
        target: Genetic meaning, units and fitted transform of the points.
        sampling_unit_ids: Default matrix order: full query-region order for
            model results, first-appearance query endpoints for standalone MLPE.
        scale: Always ``"original"``; model-scale diagnostics live separately.
    """

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    region_name: str
    target: TargetSpec
    sampling_unit_ids: tuple[str, ...]
    scale: str = "original"

    def __post_init__(self):
        self.pairs = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in self.pairs))
        self.sampling_unit_ids = tuple(self.sampling_unit_ids)
        ObservationPartition(self.region_name, self.pairs, role="query")
        if (
            not isinstance(self.target, TargetSpec)
            or self.scale != "original"
            or self.values.shape != (len(self.pairs),)
            or not jnp.issubdtype(self.values.dtype, jnp.floating)
            or bool(jnp.isinf(self.values).any())
        ):
            raise ValueError("Prediction requires aligned original-scale floating point values")
        self._validate_order(self.sampling_unit_ids)

    def _validate_order(self, order):
        if (
            not order
            or any(not isinstance(label, str) or not label for label in order)
            or len(set(order)) != len(order)
            or not {label for pair in self.pairs for label in pair}.issubset(order)
        ):
            raise ValueError("Matrix order requires unique labels and every predicted endpoint")

    def as_matrix(self, order: Sequence[str] | None = None) -> jax.Array:
        """Represent stored points as a symmetric dissimilarity matrix.

        Args:
            order: Unique sampling-unit labels containing every predicted
                endpoint; defaults to ``sampling_unit_ids``. Extra units are
                allowed and have missing off-diagonal entries.

        Returns:
            A JAX array of shape ``(n, n)`` with the stored floating dtype.
            Unrequested off-diagonal entries are NaN. The zero diagonal is a
            structural dissimilarity convention, not fitted self-predictions.
            Conversion evaluates no additional pairs and implies no covariance.

        Raises:
            ValueError: Labels are invalid or the target has no supported
                dissimilarity diagonal convention.
        """
        if self.target.kind != "dissimilarity":
            raise ValueError("Matrix conversion requires a dissimilarity diagonal convention")
        order = self.sampling_unit_ids if order is None else tuple(order)
        self._validate_order(order)
        lookup = {label: index for index, label in enumerate(order)}
        # Standalone MLPE can retain float64 after its scoped x64 operation.
        # Preserve those leaves without changing the caller's precision policy.
        with jax.enable_x64(jax.config.read("jax_enable_x64") or self.values.dtype == jnp.float64):
            matrix = jnp.full((len(order), len(order)), jnp.nan, dtype=self.values.dtype)
            diagonal = jnp.arange(len(order))
            matrix = matrix.at[diagonal, diagonal].set(0)
            left = jnp.asarray([lookup[a] for a, _ in self.pairs])
            right = jnp.asarray([lookup[b] for _, b in self.pairs])
            return matrix.at[left, right].set(self.values).at[right, left].set(self.values)
