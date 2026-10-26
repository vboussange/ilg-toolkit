"""Log-linear conductance from patch-averaged landscape covariates."""

import math
from functools import partial
from numbers import Integral, Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config import ResistanceSolverConfig, validate_patch_microbatch_size
from .base import ConductanceModel
from .common import encode_patches


@final
class LogLinearConductance(ConductanceModel):
    """A conductance model with species-specific coefficients pooled across regions.

    Patch-averaged covariates form a linear predictor for each species index.
    Conductance is ``min_conductance + exp(predictor)``, capped at ``max_conductance``.
    Coefficient columns follow the raster feature order; coefficients and intercepts start at zero.
    """

    coefficients: jax.Array
    intercepts: jax.Array
    num_species: int = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    patch_microbatch_size: int | None = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    max_conductance: float = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        num_species: int = 1,
        patch_size: int = 32,
        patch_microbatch_size: int | None = None,
        min_conductance: float = 0.01,
        max_conductance: float = 1000.0,
        solver_config: ResistanceSolverConfig | None = None,
    ):
        for name, value in (
            ("in_channels", in_channels),
            ("num_species", num_species),
            ("patch_size", patch_size),
        ):
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
        validate_patch_microbatch_size(patch_microbatch_size)
        if solver_config is not None and not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        self.num_species = int(num_species)
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.patch_microbatch_size = patch_microbatch_size
        self.min_conductance = float(min_conductance)
        self.max_conductance = float(max_conductance)
        self.solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        self.coefficients = jnp.zeros((num_species, in_channels))
        self.intercepts = jnp.zeros(num_species)

    def backbone(self, features: jax.Array) -> jax.Array:
        """Patch-averaged landscape covariates."""
        if features.ndim != 3 or features.shape[-1] != self.in_channels:
            raise ValueError(f"Expected row, column, feature data with {self.in_channels} features")
        return encode_patches(self, features, partial(jnp.mean, axis=(1, 2)))

    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Conductance surface for one species from patch-averaged covariates."""
        predictor = latent @ self.coefficients[species_index] + self.intercepts[species_index]
        ceiling = jnp.log(self.max_conductance - self.min_conductance)
        return jnp.clip(
            jnp.exp(jnp.minimum(predictor, ceiling)) + self.min_conductance,
            self.min_conductance,
            self.max_conductance,
        )
