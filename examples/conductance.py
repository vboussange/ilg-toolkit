"""Fit a tiny conductance graph: JAX_ENABLE_X64=true python examples/conductance.py."""

import coordax as cx
import jax
import jax.numpy as jnp

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ResNet9Conductance

region = RegionBatch(
    name="synthetic-valley",
    features=cx.field(
        jax.random.normal(jax.random.key(4), (8, 8, 2), dtype=jnp.float32),
        "row",
        "column",
        "feature",
    ),
    sampling_unit_ids=("south", "north", "east"),
    grid_positions=jnp.array([[0, 0], [0, 7], [7, 7]]),
)
# A 2x2 graph with conductance 2 has adjacent resistance 3/8 and opposite 1/2.
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic dissimilarity", units="index"),
)
encoder = ResNet9Conductance(
    in_channels=2,
    patch_size=4,
    min_conductance=1e-6,
    solver_config=ResistanceSolverConfig(rtol=1e-8),
    key=jax.random.key(6),
)
result = fit(
    region,
    observations,
    distance_model=encoder,
    config=TrainingConfig(
        objective="direct_log1p",
        epochs=4,
        learning_rate=0.001,
    ),
)
print("Training loss:", result.history[0].training_loss, "->", result.history[-1].training_loss)
# Neither operation requires genetic observations at the query locations.
print("Conductance surface:\n", result.model.conductance_surface(region))
print("Ecological distances (effective resistance):\n", result.model.ecological_distances(region))
print("Direct genetic predictions:\n", result.model.predict(region).as_matrix())
