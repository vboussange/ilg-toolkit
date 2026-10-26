"""Train selected observations, score held-out pairs, and average fitted models.

Observation selection and scoring are ordinary caller-owned Python code.
Run with ``JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python examples/ensemble.py``.
"""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import coordax as cx
import jax
import numpy as np

from ilg_toolkit import (
    Ensemble,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def distance_model(seed):
    return UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(seed)
    )


with jax.enable_x64():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    labels = tuple(f"population-{i}" for i in range(len(positions)))
    region = RegionBatch(
        "synthetic",
        cx.field(features, "row", "column", "feature"),
        labels,
        positions,
        feature_names=("elevation", "habitat"),
    )
    embeddings = features[tuple(positions.T)] * np.array([0.3, 1.5])
    distances = ((embeddings[:, None] - embeddings[None, :]) ** 2).sum(-1)
    left, right = np.triu_indices(len(labels), 1)
    observations = PairwiseObservations.from_pairs(
        [(labels[i], labels[j]) for i, j in zip(left, right, strict=True)],
        1 + 0.1 * distances[left, right] + rng.normal(0, 0.02, len(left)),
        target=TargetSpec("synthetic divergence", units="index", transform="log1p"),
        sampling_unit_ids=labels,
    )
    held_out = set(labels[-2:])
    training = observations.select_pairs(
        [pair for pair in observations.observed_pairs if not held_out.intersection(pair)]
    )
    query = observations.select_pairs([tuple(labels[-2:])])
    states = [
        train(
            region,
            training,
            distance_model=distance_model(seed),
            config=TrainingConfig(loss=MLPELoss(), steps=1, learning_rate=0.001, seed=seed),
        )
        for seed in (13, 29)
    ]
    ensemble = Ensemble([state.selected_model for state in states])
    prediction = ensemble.predict(region, pairs=query.observed_pairs)
    print("Member positions:", prediction.member_ids)
    print("Original-scale predictions:", prediction.prediction.values)
    print("Descriptive member standard deviation:", prediction.member_spread)
    rmse = np.sqrt(np.mean((prediction.prediction.values - query.observed_values) ** 2))
    print("Held-out pair RMSE (index):", rmse)

    # Continue one member from its latest parameters and optimizer/RNG state.
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "member-0.eqx"
        save_checkpoint(checkpoint, states[0])
        restored = load_checkpoint(checkpoint, distance_model=distance_model(13))
        continued = train(
            region, training, state=restored, config=replace(restored.config, steps=2)
        )
        print("Continued first member to step", continued.step)
