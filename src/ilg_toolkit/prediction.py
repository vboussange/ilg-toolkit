"""Pair-aligned predictions and their symmetric matrix representation."""

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import _validate_pairs


@final
class Prediction(eqx.Module):
    """Prediction: values aligned with query pairs in one region.

    Landscape genetics models return genetic predictions on the original scale;
    distance models return ecological distances.

    Args:
        values: Floating-point value of each pair, without infinities.
        pairs: Query pairs, with their order and orientation.
        region_name: Name of the region holding the pairs.
        sampling_unit_ids: Unique sampling-unit identifiers ordering
            [`as_matrix`][ilg_toolkit.Prediction.as_matrix], including every
            sampling unit of ``pairs``.
        species: Predicted species, or ``None`` for one unnamed species.
        extras: Named real vectors with one value per query pair.

    Raises:
        ValueError: If the values, pairs, identifiers, species or extras are invalid.
    """

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    region_name: str
    sampling_unit_ids: tuple[str, ...]
    species: str | None = None
    extras: Mapping[str, jax.Array] = eqx.field(default_factory=dict)

    def __post_init__(self):
        self.pairs = _validate_pairs(self.pairs)
        self.sampling_unit_ids = tuple(self.sampling_unit_ids)
        self.extras = dict(self.extras)
        if not isinstance(self.region_name, str) or not self.region_name:
            raise ValueError("region_name must be a nonempty string")
        if self.values.shape != (len(self.pairs),) or not jnp.issubdtype(
            self.values.dtype, jnp.floating
        ):
            raise ValueError("Prediction requires aligned floating values")
        self.values = eqx.error_if(
            self.values, jnp.isinf(self.values).any(), "Prediction values cannot be infinite"
        )
        if (
            not self.sampling_unit_ids
            or any(not isinstance(label, str) or not label for label in self.sampling_unit_ids)
            or len(set(self.sampling_unit_ids)) != len(self.sampling_unit_ids)
            or not {label for pair in self.pairs for label in pair}.issubset(self.sampling_unit_ids)
        ):
            raise ValueError("Sampling-unit ids must be unique and contain every query pair")
        if self.species is not None and (not isinstance(self.species, str) or not self.species):
            raise ValueError("Species must be a nonempty name or None")
        for name, values in self.extras.items():
            if (
                not isinstance(name, str)
                or not name
                or not eqx.is_array(values)
                or values.shape != (len(self.pairs),)
                or not jnp.issubdtype(values.dtype, jnp.number)
                or jnp.issubdtype(values.dtype, jnp.complexfloating)
            ):
                raise ValueError("Each named extra must be a real array aligned with query pairs")

    def as_matrix(self, name: str | None = None) -> jax.Array:
        """Returns the values, or a named extra, as a symmetric matrix.

        Args:
            name: Name of an extra; ``None`` selects ``values``.

        Returns:
            Symmetric matrix in ``sampling_unit_ids`` order, with not-a-number (NaN)
            entries on the diagonal and for pairs not in ``pairs``.
        """
        values = self.values if name is None else self.extras[name]
        lookup = {label: index for index, label in enumerate(self.sampling_unit_ids)}
        dtype = jnp.result_type(values.dtype, jnp.float32)
        matrix = jnp.full((len(lookup), len(lookup)), jnp.nan, dtype=dtype)
        left = jnp.asarray([lookup[a] for a, _ in self.pairs])
        right = jnp.asarray([lookup[b] for _, b in self.pairs])
        return matrix.at[left, right].set(values).at[right, left].set(values)
