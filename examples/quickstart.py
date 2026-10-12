"""Fit conductance from synthetic measurements: uv run python examples/quickstart.py."""

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import ResNet9Conductance

# Set precision before creating inputs; resistance calculations require float64.
jax.config.update("jax_enable_x64", True)

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
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic genetic dissimilarity", units="index", transform="identity"),
)
encoder = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = fit(
    region,
    observations,
    distance_model=encoder,
    config=TrainingConfig(objective="direct_log1p", epochs=4, learning_rate=0.001),
)
print("Conductance surface:\n", result.model.conductance_surface(region))
print("Resistance landscape scores:\n", result.model.landscape_scores(region))
print("Genetic predictions:\n", result.model.predict(region).as_matrix())
