"""Pool one U-Net backbone across synthetic regions of different extents."""

from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def synthetic_region(name, width, locations):
    rows, columns = jnp.mgrid[:4, :width]
    features = jnp.stack((rows / 4, columns / 6), axis=-1).astype(jnp.float32)
    identifiers = tuple(f"{name}-{index}" for index in range(len(locations)))
    return RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        {"fox": SamplingUnits(identifiers, jnp.asarray(locations))},
        feature_names=("north_covariate", "east_covariate"),
    )


regions = [
    synthetic_region("headwaters", 4, [[0, 0], [0, 3], [3, 3]]),
    synthetic_region("lowlands", 6, [[0, 0], [3, 5]]),
]
target = TargetSpec("synthetic divergence", units="index")
observations = {}
for region in regions:
    units = region.sampling_units["fox"]
    endpoint_features = region.feature_array[tuple(cast(jax.Array, units.grid_positions).T)]
    dissimilarities = jnp.sum(
        (endpoint_features[:, None] - endpoint_features[None, :]) ** 2, axis=-1
    )
    observations[region.name] = {
        "fox": PairwiseObservations.from_matrix(units.ids, dissimilarities, target=target)
    }
model = UNetEmbeddingDistance(
    2,
    species=("fox",),
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(4),
)
result = fit(
    regions,
    observations,
    distance_model=model,
    config=TrainingConfig(objective="direct_log1p", steps=4, learning_rate=0.001, seed=4),
)
print("Selected pooled model at step", result.selected_step)
print("Per-group training objectives:", result.history[-1].training_by_group)
print("Per-species objectives:", result.history[-1].training_by_species)
for region in regions:
    print(region.name, result.model.predict(region)["fox"].as_matrix(), result.model.target.units)
