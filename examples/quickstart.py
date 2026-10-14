"""Fit a conductance model to synthetic genetic dissimilarities, then save and reload it."""

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
    load_model,
    save_model,
)
from ilg_toolkit.models import ResNet9Conductance

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
    sampling_units={"fox": SamplingUnits(("south", "north", "east"), [[0, 0], [0, 7], [7, 7]])},
)
# Pairwise genetic observations, with their meaning, units and transform declared.
observations = PairwiseObservations.from_matrix(
    region.sampling_units["fox"].ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic genetic dissimilarity", units="index", transform="identity"),
)
# patch_size=4 coarsens the 8x8 raster into a 2x2 conductance grid graph.
distance_model = ResNet9Conductance(
    in_channels=2, species=("fox",), patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = fit(
    [region],
    {region.name: {"fox": observations}},
    distance_model=distance_model,
    config=TrainingConfig(objective="direct_log1p", steps=4, learning_rate=0.001),
)
model = result.model  # The selected landscape genetics model.

print("Conductance surface:\n", model.conductance_surface(region)["fox"])
print("Ecological distances (effective resistance):\n", model.ecological_distances(region)["fox"])
# Outputs stay species-keyed, even when the sole species can be inferred.
# predict returns one calibrated value per distinct pair, in the target's units;
# as_matrix lays them out symmetrically in sampling-unit order.
prediction = model.predict(region)["fox"]
print("Genetic predictions:", prediction.pairs, prediction.values)
print("Genetic prediction matrix:\n", prediction.as_matrix())

# Loading checks the exact ordered registry and architecture/solver settings.
save_model("model.ilg", model)
template = ResNet9Conductance(
    in_channels=2, species=("fox",), patch_size=4, min_conductance=1e-6, key=jax.random.key(0)
)
restored = load_model("model.ilg", distance_model=template)
print("Restored predictions:\n", restored.predict(region)["fox"].as_matrix())
