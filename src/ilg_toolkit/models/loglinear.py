"""Independent species coefficients on shared, fixed patch feature means."""

import math
from numbers import Integral, Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from .._pairs import PairSelection
from ..config import ResistanceSolverConfig
from .base import ConductanceModel, _pooled_resistance_pairs, validate_species
from .common import patchify


@final
class LogLinearConductance(ConductanceModel):
    """Bounded exponential of a species-specific predictor of patch feature means.

    ``coefficients[species_index, feature_index]`` follows ``species`` and the
    supplied raster feature order; ``intercepts[species_index]`` uses the same
    registry. A fitted LandscapeGeneticsModel also retains the input feature_names.
    For patch means x, the predictor is eta = x @ coefficients[s] + intercepts[s].
    Bounds a and b give clip(exp(min(eta, log(b - a))) + a, a, b). Defaults are
    a=0.01, b=1000 and patch_size=32. Zero coefficients/intercepts yield 1.01
    under these defaults, without a random key. Parameters use JAX's default
    floating precision; graph resistance requires x64 enabled. Fixed patch
    summaries carry no learned transfer between species; each row is pooled
    across regions.

    Input preparation owns feature scaling and categorical reference coding.
    With an intercept, avoid redundant full indicator sets and other collinear
    features; this model never drops a category. The floor and cap break exact
    conductance rescaling for nonuniform surfaces. Under standardized MLPE the
    intercept is not an independently identified absolute connectivity scale;
    a uniform surface remains scale-invariant.
    """

    species: tuple[str, ...] = eqx.field(static=True)
    coefficients: jax.Array
    intercepts: jax.Array
    in_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    max_conductance: float = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        species,
        patch_size: int = 32,
        min_conductance: float = 0.01,
        max_conductance: float = 1000.0,
        solver_config: ResistanceSolverConfig | None = None,
    ):
        self.species = validate_species(species)
        for name, value in (("in_channels", in_channels), ("patch_size", patch_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                for value in (min_conductance, max_conductance)
            )
            or not 0 < min_conductance < max_conductance
        ):
            raise ValueError(
                "Require finite conductance bounds 0 < min_conductance < max_conductance"
            )
        if solver_config is not None and not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.min_conductance = float(min_conductance)
        self.max_conductance = float(max_conductance)
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        self.coefficients = jnp.zeros((len(self.species), in_channels))
        self.intercepts = jnp.zeros(len(self.species))

    def _split_pooled(self):
        """Fixed summaries have no learned backbone and require no encoder VJP."""
        return None, self

    def _pooled_features(self, features, *, inference=True, key=None, patch_batch_size=None):
        """Fixed patch summaries reusable across selected species; no trainable cut."""
        features = jnp.asarray(features)
        if features.ndim != 3 or features.shape[-1] != self.in_channels:
            raise ValueError(f"Expected row, column, feature data with {self.in_channels} features")
        patches, grid_shape = patchify(features, self.patch_size)
        return jnp.mean(patches, axis=(2, 3)).reshape(*grid_shape, self.in_channels)

    def _conductance_from_pooled(self, pooled, species_index):
        predictor = pooled @ self.coefficients[species_index] + self.intercepts[species_index]
        ceiling = jnp.log(self.max_conductance - self.min_conductance)
        return jnp.clip(
            jnp.exp(jnp.minimum(predictor, ceiling)) + self.min_conductance,
            self.min_conductance,
            self.max_conductance,
        )

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        pooled = self._pooled_features(features, patch_batch_size=patch_batch_size)
        return self._conductance_from_pooled(pooled, species_index)

    def _pairs_from_pooled(self, pooled, selection: PairSelection, species_index, *, context=None):
        return _pooled_resistance_pairs(
            self._conductance_from_pooled,
            pooled,
            selection,
            species_index,
            self.solver_config,
            context=context,
        )

    def _pair_distances(
        self,
        features,
        selection,
        species_index,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        pooled = self._pooled_features(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        return self._pairs_from_pooled(pooled, selection, species_index, context=context)
