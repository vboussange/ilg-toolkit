"""Fit a tiny conductance graph: JAX_ENABLE_X64=true python examples/conductance.py."""

import jax
import numpy as np

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
    features=np.random.default_rng(4).normal(size=(8, 8, 2)).astype(np.float32),
    sampling_unit_ids=("south", "north", "east"),
    grid_positions=np.array([[0, 0], [0, 7], [7, 7]]),
)
# A 2x2 graph with conductance 2 has adjacent resistance 3/8 and opposite 1/2.
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids,
    [[0, 0.375, 0.5], [0.375, 0, 0.375], [0.5, 0.375, 0]],
    target=TargetSpec("synthetic dissimilarity", units="index"),
)
encoder = ResNet9Conductance(
    in_channels=2, patch_size=4, min_conductance=1e-6, key=jax.random.key(6)
)
result = fit(
    region,
    observations,
    model=encoder,
    config=TrainingConfig(epochs=4, learning_rate=0.001, solver=ResistanceSolverConfig(rtol=1e-8)),
)
print("Training loss:", result.history[0].training_loss, "->", result.history[-1].training_loss)
# Neither operation requires genetic observations at the query locations.
print("Conductance surface:\n", result.model.conductance_surface(region))
print("Resistance landscape scores:\n", result.model.landscape_scores(region))
print("Direct genetic predictions:\n", result.model.predict(region).values)
