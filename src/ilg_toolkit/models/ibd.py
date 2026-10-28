"""Isolation-by-distance (IBD) model: an affine function of straight-line vertex distance."""

import math
from collections.abc import Sequence
from numbers import Real
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxscape import EuclideanDistance, GridGraph

from .._checks import _positive_integer
from .base import _species_names, AbstractDistanceModel


@final
class IBDModel(AbstractDistanceModel):
    r"""An isolation-by-distance (IBD) model with one affine map per species.

    The ecological distance between two sampling units is

    $$s_{ij} = a \, d_{ij} + b,$$

    where $d_{ij}$ is JAXScape's ``EuclideanDistance`` between the units' vertices
    on the conductance grid graph's vertex layout, and $a$ and $b$ are the
    species' non-negative slope and intercept, each the softplus of a trainable
    parameter. Each sampling unit sits at the vertex of its containing
    ``patch_size`` patch, as for the patch encoders, so with the same patch size
    an IBD model and a patch encoder measure distances between the same vertices.
    Distances are in vertex spacings, and the slope absorbs their units. The model
    ignores landscape features: ``backbone`` returns a vertex grid with no
    channels.

    ``train`` with ``Log1pLoss`` and ``optax.lbfgs()`` trains the IBD model
    jointly across regions, with any target transform, including ``"odds"``. An
    MLPE calibration head standardizes ecological distances, so its fit is the
    same for every slope and intercept: ``fit_calibration_head(...,
    loss=MLPELoss())`` on an untrained IBD model gives the MLPE baseline on
    geographic distance.

    Args:
        species: Species names, one slope and intercept each in parameter order;
            ``None`` (default) models one unnamed species.
        patch_size: Side length of square raster patches, default 32 as for
            ``LogLinearConductance``; 1 places each sampling unit at its own raster
            cell. Both raster dimensions must be divisible by it, and sampling
            units in the same patch share a vertex.
        slope: Positive initial slope of every species, in ecological distance per
            vertex spacing, default 1.
        intercept: Positive initial intercept of every species, default 0.01.

    Examples:
        >>> distance_model = IBDModel(patch_size=8)
        >>> config = TrainingConfig(loss=Log1pLoss(), optimizer=optax.lbfgs(),
        ...                         steps=200, gradient_tolerance=1e-9)
        >>> state = train(region, observations, distance_model=distance_model, config=config)
        >>> state.selected_model.distance_model.slopes
        >>> head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    """

    raw_slopes: jax.Array
    raw_intercepts: jax.Array
    species: tuple[str, ...] | None = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        species: Sequence[str] | None = None,
        patch_size: int = 32,
        slope: float = 1.0,
        intercept: float = 0.01,
    ):
        species = _species_names(species)
        patch_size = _positive_integer("patch_size", patch_size)
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in (slope, intercept)
        ) or not (slope > 0 and intercept > 0):
            raise ValueError("IBD slope and intercept must be finite positive real numbers")
        self.species = species
        self.patch_size = patch_size
        count = 1 if species is None else len(species)
        self.raw_slopes = jnp.full(count, _inverse_softplus(float(slope)), dtype=float)
        self.raw_intercepts = jnp.full(count, _inverse_softplus(float(intercept)), dtype=float)

    @property
    def slopes(self) -> jax.Array:
        """Non-negative slope of each species, in parameter order."""
        return jax.nn.softplus(self.raw_slopes)

    @property
    def intercepts(self) -> jax.Array:
        """Non-negative intercept of each species, in parameter order."""
        return jax.nn.softplus(self.raw_intercepts)

    def backbone(self, features: jax.Array) -> jax.Array:
        """Returns a vertex grid with no channels, which carries only the vertex layout.

        Args:
            features: Landscape features with shape ``(rows, columns, features)``;
                only their shape is read.

        Returns:
            An array of shape ``(rows // patch_size, columns // patch_size, 0)``.
        """
        if features.ndim != 3:
            raise ValueError(f"Expected row, column, feature data, got shape {features.shape}")
        rows, columns = features.shape[0] // self.patch_size, features.shape[1] // self.patch_size
        return jnp.zeros((rows, columns, 0), dtype=self.raw_slopes.dtype)

    def ecological_distances(
        self,
        latent: jax.Array,
        pairs: tuple[jax.Array, tuple[jax.Array, jax.Array]],
        species_index: int | jax.Array,
    ) -> jax.Array:
        """Computes affine straight-line distances between the vertices of pairs of sampling units.

        Args:
            latent: Backbone output of the region.
            pairs: ``(vertices, (left, right))``, as in
                ``AbstractDistanceModel.ecological_distances``.
            species_index: Parameter row of the species, an integer or a scalar
                integer array.

        Returns:
            The species' slope times the Euclidean vertex distance, plus its
            intercept, aligned with ``left`` and ``right``.
        """
        nodes, (left, right) = pairs
        graph = GridGraph(jnp.ones(latent.shape[:2], dtype=latent.dtype))
        # 64-bit vertex indices give float64 coordinate differences and distances.
        distances = EuclideanDistance()(graph, nodes=nodes.astype(jnp.int64))[left, right]
        return self.slopes[species_index] * distances + self.intercepts[species_index]


def _inverse_softplus(value: float) -> float:
    """Returns the parameter whose softplus is a positive ``value``, computed stably."""
    return value + math.log(-math.expm1(-value))
