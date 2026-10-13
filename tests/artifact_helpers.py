"""Small native fixtures shared by artifact and checkpoint public workflows."""

from collections.abc import Callable
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits, TargetSpec
from ilg_toolkit.models import (
    EmbeddingDistanceModel,
    LogLinearConductance,
    ResNet9Conductance,
    UNetEmbeddingDistance,
)


@final
class NativeEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True)
    weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)
    scale: float = 1.0
    activation: Callable = jax.nn.tanh

    def __init__(self, *, species=("fox", "lynx"), seed=3, scale=1.0):
        self.species = species
        self.weights = jax.random.normal(jax.random.key(seed), (len(species), 2)) * 0.1
        self.scale = scale
        self.activation = jax.nn.tanh

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return self.activation(features * jnp.exp(self.weights[species_index])) * self.scale


def distance_model(family="custom", seed=3, species=("fox", "lynx")):
    if family == "custom":
        return NativeEmbedding(seed=seed, species=species)
    if family == "unet":
        return UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=4,
            base_channels=2,
            embedding_dim=2,
            dropout=0.2,
            key=jax.random.key(seed),
        )
    if family == "resnet":
        return ResNet9Conductance(
            2, species=species, patch_size=4, patch_batch_size=3, key=jax.random.key(seed)
        )
    if family == "loglinear":
        return LogLinearConductance(2, species=species, patch_size=4)
    raise ValueError(family)


def checkpoint_problem(name="alpine", seed=14):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(8, 8, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 7], [7, 0], [7, 7]])
    ids = tuple(f"{name}-{i}" for i in range(4))
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        {species: SamplingUnits(ids, positions) for species in ("fox", "lynx")},
        feature_names=("elevation", "canopy"),
    )
    pairs = [(ids[i], ids[j]) for i, j in zip(*np.triu_indices(4, 1), strict=True)]
    observations = {
        species: PairwiseObservations.from_pairs(
            pairs,
            rng.uniform(0.2, 1.5, len(pairs)),
            target=TargetSpec("synthetic divergence", units="index"),
            sampling_unit_ids=ids,
        )
        for species in region.sampling_units
    }
    return region, observations
