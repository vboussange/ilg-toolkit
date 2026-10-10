"""Calibrate declared landscape scores separately from the frozen encoder."""

import jax
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, calibrate_mlpe
from ilg_toolkit.models import UNetEmbeddingDistance

rng = np.random.default_rng(5)
region = RegionBatch(
    "synthetic-valley",
    rng.normal(size=(4, 4, 2)),
    tuple(f"population-{i}" for i in range(8)),
    np.array([[0, 0], [0, 1], [0, 3], [1, 1], [1, 3], [2, 0], [3, 1], [3, 3]]),
)
encoder = UNetEmbeddingDistance(
    2,
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(3),
)
raw_matrix = np.asarray(encoder.predict_distances(region.feature_array, region.pixel_nodes))
left, right = np.triu_indices(len(region.sampling_unit_ids), 1)
raw_scores = np.asarray(raw_matrix[left, right], dtype=np.float64)
effects = rng.normal(scale=0.04, size=8)
measured = (
    1
    + 0.15 * (raw_scores - raw_scores.mean()) / raw_scores.std(ddof=1)
    + effects[left]
    + effects[right]
    + rng.normal(scale=0.02, size=len(left))
)
pairs = tuple(
    (region.sampling_unit_ids[i], region.sampling_unit_ids[j])
    for i, j in zip(left, right, strict=True)
)
observations = PairwiseObservations.from_pairs(
    pairs,
    measured,
    sampling_unit_ids=region.sampling_unit_ids,
    target=TargetSpec("synthetic divergence", units="index"),
)
head = calibrate_mlpe(raw_scores, observations, region_name=region.name)
prediction = head.predict_marginal(raw_scores, observations.observed_pairs)
print("Landscape scores:", raw_scores[:4])
print("Calibrated genetic predictions:", prediction.values[:4], prediction.target.units)
print("Signed fitted slope:", head.slope)
