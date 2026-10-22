"""Fit genetic calibration for a new region with a frozen distance model."""

from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    MLPECalibrationModule,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import UNet

rng = np.random.default_rng(4)
region = RegionBatch(
    "original-catchment",
    cx.field(rng.normal(size=(4, 4, 2)).astype(np.float32), "row", "column", "feature"),
    tuple(f"population-{index}" for index in range(6)),
    np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]]),
    feature_names=("elevation", "canopy"),
)
model = UNet(2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(4))
initial_distances = model.predict(region).as_matrix()
effects = np.array([0.2, -0.1, 0.15, -0.2, 0.25, -0.05])
noise = rng.normal(scale=0.08, size=(6, 6))
targets = 2 + 0.5 * initial_distances + effects[:, None] + effects[None, :] + (noise + noise.T) / 2
targets = targets.at[jnp.diag_indices(len(region.sampling_unit_ids))].set(0)
observations = PairwiseObservations.from_matrix(
    region.sampling_unit_ids, targets, target=TargetSpec("synthetic divergence", units="index")
)
result = fit(
    region,
    observations,
    distance_model=model,
    config=TrainingConfig(objective="direct_log1p", steps=3, learning_rate=0.0001, seed=4),
)
calibrated = fit(
    region,
    observations,
    distance_model=result.selected_model.distance_model,
    config=TrainingConfig(objective="mlpe", steps=0),
).selected_model
new_region = replace(
    region,
    name="new-catchment",
    features=cx.field(region.feature_array * 1.1, "row", "column", "feature"),
)
print(
    "Label-free ecological distances in new region:\n",
    calibrated.distance_model.predict(new_region).as_matrix(),
)
try:
    calibrated.predict(new_region)
except ValueError as error:
    print("Before regional calibration:", error)
new_observations = PairwiseObservations.from_matrix(
    new_region.sampling_unit_ids, observations.values * 2, target=observations.target
)
transferred = fit(
    new_region,
    new_observations,
    distance_model=calibrated.distance_model,
    config=TrainingConfig(objective="mlpe", steps=0),
).selected_model
assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
assert isinstance(calibrated.calibration, MLPECalibrationModule)
assert isinstance(transferred.calibration, MLPECalibrationModule)
assert tuple(transferred.calibration.regions) == (new_region.name,)
print("Calibrated new-region predictions:\n", transferred.predict(new_region).as_matrix())
assert isinstance(transferred.calibration, MLPECalibrationModule)
print(
    "Declared data roles:", set(transferred.calibration.regions[new_region.name].calibration_roles)
)
