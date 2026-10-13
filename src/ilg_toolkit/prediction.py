"""Species-scoped ecological distances and original-scale genetic predictions."""

from collections.abc import Sequence
from typing import cast, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import ObservationPartition, TargetSpec


def _validate_order(order, pairs):
    if (
        not order
        or any(not isinstance(label, str) or not label for label in order)
        or len(set(order)) != len(order)
        or not {label for pair in pairs for label in pair}.issubset(order)
    ):
        raise ValueError("Matrix order requires unique labels and every predicted endpoint")


def _matrix(values, pairs, order):
    _validate_order(order, pairs)
    lookup = {label: index for index, label in enumerate(order)}
    with jax.enable_x64(jax.config.read("jax_enable_x64") or values.dtype == jnp.float64):
        matrix = jnp.full((len(order), len(order)), jnp.nan, dtype=values.dtype)
        diagonal = jnp.arange(len(order))
        matrix = matrix.at[diagonal, diagonal].set(0)
        left = jnp.asarray([lookup[a] for a, _ in pairs])
        right = jnp.asarray([lookup[b] for _, b in pairs])
        return matrix.at[left, right].set(values).at[right, left].set(values)


@final
class RawDistances(eqx.Module):
    """Finite ecological pair distances, without a genetic target contract."""

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    region_name: str
    species_name: str
    sampling_unit_ids: tuple[str, ...]

    def __post_init__(self):
        self.values = jnp.asarray(self.values)
        self.pairs = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in self.pairs))
        self.sampling_unit_ids = tuple(self.sampling_unit_ids)
        ObservationPartition(self.region_name, self.species_name, self.pairs, role="query")
        _validate_order(self.sampling_unit_ids, self.pairs)
        if self.values.shape != (len(self.pairs),):
            raise ValueError("Raw distances require one floating value per requested pair")
        if not jnp.issubdtype(self.values.dtype, jnp.floating):
            raise ValueError("Raw distances must have floating precision")
        if not bool(jnp.isfinite(self.values).all()) or bool((self.values < 0).any()):
            raise FloatingPointError("Distance model produced nonfinite or negative distances")

    def as_matrix(self, order: Sequence[str] | None = None) -> jax.Array:
        """Represent only stored pairs; unrequested off-diagonals remain NaN."""
        return _matrix(
            self.values, self.pairs, self.sampling_unit_ids if order is None else tuple(order)
        )


@final
class Prediction(eqx.Module):
    """Original genetic-scale values with region, species and pair provenance."""

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    region_name: str
    species_name: str
    target: TargetSpec
    sampling_unit_ids: tuple[str, ...]
    scale: str = "original"

    def __post_init__(self):
        self.values = jnp.asarray(self.values)
        self.pairs = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in self.pairs))
        self.sampling_unit_ids = tuple(self.sampling_unit_ids)
        ObservationPartition(self.region_name, self.species_name, self.pairs, role="query")
        _validate_order(self.sampling_unit_ids, self.pairs)
        if (
            not isinstance(self.target, TargetSpec)
            or self.scale != "original"
            or self.values.shape != (len(self.pairs),)
            or not jnp.issubdtype(self.values.dtype, jnp.floating)
            or bool(jnp.isinf(self.values).any())
        ):
            raise ValueError("Prediction requires aligned original-scale floating point values")

    def as_matrix(self, order: Sequence[str] | None = None) -> jax.Array:
        """Symmetric stored-pair view, with zero diagonal and missing-pair NaNs."""
        if self.target.kind != "dissimilarity":
            raise ValueError("Matrix conversion requires a dissimilarity diagonal convention")
        return _matrix(
            self.values, self.pairs, self.sampling_unit_ids if order is None else tuple(order)
        )
