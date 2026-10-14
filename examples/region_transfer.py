"""Freeze the complete distance model and calibrate a compatible new region."""

from itertools import combinations

import coordax as cx
import equinox as eqx
import jax
import numpy as np

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    calibrate,
    fit,
)
from ilg_toolkit.models import UNetEmbeddingDistance

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(4)
positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
features = rng.normal(size=(4, 4, 2)).astype(np.float32)
labels = tuple(f"original-{i}" for i in range(6))
region = RegionBatch(
    "original-catchment",
    cx.field(features, "row", "column", "feature"),
    {"fox": SamplingUnits(labels, positions)},
    feature_names=("elevation", "canopy"),
)
distance_model = UNetEmbeddingDistance(
    2,
    species=("fox",),
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(4),
)
pairs = tuple(combinations(labels, 2))
distances = distance_model.predict_raw_distances(region, pairs={"fox": pairs})["fox"].values
left, right = np.triu_indices(len(labels), 1)
effects = np.array([0.2, -0.1, 0.15, -0.2, 0.25, -0.05])
measured = (
    2 + 0.5 * distances + effects[left] + effects[right] + rng.normal(scale=0.08, size=len(pairs))
)
observations = PairwiseObservations.from_pairs(
    pairs,
    measured,
    sampling_unit_ids=labels,
    target=TargetSpec("synthetic divergence", units="index"),
)
result = fit(
    [region],
    {region.name: {"fox": observations}},
    distance_model=distance_model,
    config=TrainingConfig(objective="direct_log1p", steps=3, learning_rate=0.0001, seed=4),
)
# Recorded training access permits calibration of the original group.
calibrated = calibrate(result.model, [region], {region.name: {"fox": observations}})
# Different biological populations get different identities within this species.
new_labels = tuple(f"new-{i}" for i in range(6))
new_region = RegionBatch(
    "new-catchment",
    cx.field(features * 1.1, "row", "column", "feature"),
    {"fox": SamplingUnits(new_labels, positions)},
    feature_names=region.feature_names,
)
print("Label-free ecological distances:\n", calibrated.ecological_distances(new_region)["fox"])
try:
    calibrated.predict(new_region)
except ValueError as error:
    print("Before group calibration:", error)
else:
    raise AssertionError("An unseen group requires its own MLPE calibration")
new_observations = PairwiseObservations.from_pairs(
    tuple(combinations(new_labels, 2)),
    observations.observed_values * 2,
    sampling_unit_ids=new_labels,
    target=observations.target,
)
development = ObservationPartition(
    new_region.name, "fox", new_observations.observed_pairs, role="calibration"
)
transferred = calibrated.calibrate(
    new_region,
    {"fox": new_observations},
    partitions={"fox": development},
)
assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
assert isinstance(calibrated.calibration, UnpooledMLPECalibrationModule)
assert isinstance(transferred.calibration, UnpooledMLPECalibrationModule)
assert eqx.tree_equal(
    transferred.calibration.for_group(region.name, "fox"),
    calibrated.calibration.for_group(region.name, "fox"),
)
print("Calibrated new-region predictions:\n", transferred.predict(new_region)["fox"].as_matrix())
print(
    "Declared data roles:",
    set(transferred.calibration.for_group(new_region.name, "fox").calibration_roles),
)
