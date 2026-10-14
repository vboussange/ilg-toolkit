"""Fit one MLPE relationship to requested distances from a frozen U-Net."""

from itertools import combinations

import coordax as cx
import jax
import numpy as np

from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits, TargetSpec, calibrate_mlpe
from ilg_toolkit.models import UNetEmbeddingDistance

jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(5)
labels = tuple(f"population-{i}" for i in range(8))
region = RegionBatch(
    "synthetic-valley",
    cx.field(rng.normal(size=(4, 4, 2)).astype(np.float32), "row", "column", "feature"),
    {
        "fox": SamplingUnits(
            labels, [[0, 0], [0, 1], [0, 3], [1, 1], [1, 3], [2, 0], [3, 1], [3, 3]]
        )
    },
    feature_names=("elevation", "habitat"),
)
encoder = UNetEmbeddingDistance(
    2,
    species=("fox",),
    patch_size=1,
    base_channels=2,
    embedding_dim=2,
    dropout=0,
    key=jax.random.key(3),
)
pairs = tuple(combinations(labels, 2))
# Pair vectors preserve requested order; no full distance matrix is needed.
distances = encoder.predict_raw_distances(region, pairs={"fox": pairs})["fox"].values.astype(
    "float64"
)
left, right = np.triu_indices(len(labels), 1)
effects = rng.normal(scale=0.04, size=len(labels))
measured = (
    1
    + 0.15 * (distances - distances.mean()) / distances.std(ddof=1)
    + effects[left]
    + effects[right]
    + rng.normal(scale=0.02, size=len(pairs))
)
observations = PairwiseObservations.from_pairs(
    pairs,
    measured,
    sampling_unit_ids=labels,
    target=TargetSpec("synthetic divergence", units="index"),
)
# Calibration consumes distances aligned with the observation's canonical pair order.
raw = encoder.predict_raw_distances(region, pairs={"fox": observations.observed_pairs})["fox"]
head = calibrate_mlpe(raw.values, observations, region_name=region.name, species_name="fox")
prediction = head.predict_marginal(raw.values, observations.observed_pairs)
assert np.isfinite(prediction.prediction.values).all()
print("Ecological distances:", raw.values[:4])
print(
    "Calibrated predictions:", prediction.prediction.values[:4], prediction.prediction.target.units
)
print("Signed fitted slope:", head.slope)
