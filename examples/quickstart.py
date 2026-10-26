"""Fit a conductance model to synthetic genetic dissimilarities, then save and reload it."""

from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import (
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    load_model,
    save_model,
    train,
)
from ilg_toolkit.models import ConductanceModel, ResNet9Conductance

# Effective resistance needs float64: enable x64 before creating any inputs.
jax.config.update("jax_enable_x64", True)

# An 8x8 raster with two covariates, and three sampling units at (row, column) cells.
region = RegionBatch(
    name="synthetic-valley",
    features=cx.field(
        jax.random.normal(jax.random.key(4), (8, 8, 2), dtype=jnp.float32),
        "row",
        "column",
        cx.LabeledAxis("feature", ["elevation", "habitat"]),
    ),
    sampling_unit_ids=("south", "north", "east"),
    grid_positions=[[0, 0], [0, 7], [7, 7]],
)
# Pairwise genetic observations with their meaning, units, and transform.
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic genetic dissimilarity", units="index", transform="identity"),
)
# patch_size=4 coarsens the 8x8 raster into a 2x2 conductance grid graph.
distance_model = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = train(
    region,
    observations,
    distance_model=distance_model,
    config=TrainingConfig(loss=Log1pLoss(), steps=4, learning_rate=0.001),
)
model = result.selected_model  # The selected landscape genetics model.

print(
    "Conductance surface:\n",
    cast(ConductanceModel, model.distance_model).conductance_surface(region),
)
print(
    "Ecological distances (effective resistance):\n",
    model.distance_model.predict(region).as_matrix(),
)
# predict returns one genetic prediction per distinct pair, in the target's units;
# as_matrix lays them out symmetrically in sampling-unit order.
prediction = model.predict(region)
print("Genetic predictions:", prediction.pairs, prediction.values)
print("Genetic prediction matrix:\n", prediction.as_matrix())

# Loading needs a template distance model with the same configuration.
save_model("model.eqx", model)
template = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(0)
)
restored = load_model("model.eqx", distance_model=template)
print("Restored predictions:\n", restored.predict(region).as_matrix())
