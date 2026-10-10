"""Fit independent synthetic MLPE members; inspect calibrated means and spread."""

import jax
import numpy as np

from ilg_toolkit import TrainingConfig, PairwiseObservations, RegionBatch, TargetSpec, fit_ensemble
from ilg_toolkit.models import UNetEmbeddingDistance


def model_factory(key):
    # Every invocation initializes a new model from its member-specific key.
    return UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=key
    )


with jax.enable_x64():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    labels = tuple(f"population-{i}" for i in range(len(positions)))
    region = RegionBatch(
        "synthetic", features, labels, positions, feature_names=("elevation", "habitat")
    )
    embeddings = features[tuple(positions.T)] * np.array([0.3, 1.5])
    scores = ((embeddings[:, None] - embeddings[None, :]) ** 2).sum(-1)
    left, right = np.triu_indices(len(labels), 1)
    targets = 1 + 0.1 * scores[left, right] + rng.normal(0, 0.02, len(left))
    observations = PairwiseObservations.from_pairs(
        [(labels[i], labels[j]) for i, j in zip(left, right, strict=True)],
        targets,
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=labels,
    )
    ensemble = fit_ensemble(
        region,
        observations,
        n_folds=2,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(13, 29),
        model_factory=model_factory,
        config=TrainingConfig(objective="mlpe", epochs=2, learning_rate=0.001),
    )
    if ensemble.failures:
        raise RuntimeError(ensemble.failures)
    prediction = ensemble.predict(region)
    print("Independent members:", prediction.member_ids)
    print("Calibrated deployment mean, original units:", prediction.target.units)
    print(prediction.values)
    print("Descriptive member SD (not a confidence interval):")
    print(prediction.member_spread)
    print("Query-only holdouts; this deployment mean is not out-of-fold evaluation.")
