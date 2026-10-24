"""Pair-aligned predictions and their symmetric matrix representation."""

from collections.abc import Mapping
from typing import cast, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import ObservationPartition


@final
class Prediction(eqx.Module):
    """Values and named results aligned with query pairs in one region.

    Sampling-unit identifiers define the matrix order, and species optionally
    identifies the species being predicted. Each extra is a vector with one
    value per query pair.
    """

    values: jax.Array
    pairs: tuple[tuple[str, str], ...]
    region_name: str
    sampling_unit_ids: tuple[str, ...]
    species: str | None = None
    extras: Mapping[str, jax.Array] = eqx.field(default_factory=dict)

    def __post_init__(self):
        self.pairs = cast(tuple[tuple[str, str], ...], tuple(tuple(pair) for pair in self.pairs))
        self.sampling_unit_ids = tuple(self.sampling_unit_ids)
        self.extras = dict(self.extras)
        ObservationPartition(self.region_name, self.pairs, role="query")
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
        """Symmetric matrix of stored values or a named extra in sampling-unit order.

        The diagonal and unrequested pairs are not-a-number (NaN).
        """
        values = self.values if name is None else self.extras[name]
        lookup = {label: index for index, label in enumerate(self.sampling_unit_ids)}
        with jax.enable_x64(jax.config.read("jax_enable_x64") or values.dtype.itemsize == 8):
            dtype = jnp.result_type(values.dtype, jnp.float32)
            matrix = jnp.full((len(lookup), len(lookup)), jnp.nan, dtype=dtype)
            left = jnp.asarray([lookup[a] for a, _ in self.pairs])
            right = jnp.asarray([lookup[b] for _, b in self.pairs])
            return matrix.at[left, right].set(values).at[right, left].set(values)
