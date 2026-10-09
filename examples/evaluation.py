"""Actual independent MLPE fits followed by coverage-aware OOF scoring."""

import jax
import numpy as np

from ilg_toolkit import (
    FitConfig,
    PairwiseObservations,
    PreparedRegion,
    TargetSpec,
    evaluate_ensemble,
    fit_ensemble,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def model_factory(key):
    return UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=key
    )


with jax.enable_x64():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    labels = tuple(f"population-{i}" for i in range(len(positions)))
    region = PreparedRegion(
        "synthetic", features, labels, positions, feature_names=("elevation", "habitat")
    )
    embedding = features[tuple(positions.T)] * np.array([0.3, 1.5])
    scores = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)
    left, right = np.triu_indices(len(labels), 1)
    observed = PairwiseObservations.from_pairs(
        [(labels[i], labels[j]) for i, j in zip(left, right, strict=True)],
        1 + 0.1 * scores[left, right] + rng.normal(0, 0.02, len(left)),
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=labels,
    )
    ensemble = fit_ensemble(
        region,
        observed,
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(13, 29),
        config=FitConfig(objective="mlpe", epochs=1),
        model_factory=model_factory,
    )
    result = evaluate_ensemble(ensemble, region, observed)
    print("Covered unique pairs:", result.n_pairs, "of", result.n_query_pairs)
    print("Coverage:", result.coverage)
    print("Eligible member counts:", result.predictions.eligible_counts)
    print("Original-scale MSE, RMSE, MAE:", result.mse, result.rmse, result.mae)
    print(
        "Fit/prediction failures:",
        result.predictions.member_failures,
        result.predictions.prediction_failures,
    )
    print("All targets accessed during training/selection/calibration are excluded.")
