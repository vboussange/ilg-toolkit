"""In-memory direct-regression training independent of benchmark orchestration."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig
from .data import ObservationPartition, PairwiseObservations, PreparedRegion
from .models import EmbeddingDistanceModel, UNetEmbeddingDistance
from .predictor import Predictor


@dataclass(frozen=True)
class EpochRecord:
    """Inference-mode objective values after a completed update (zero is initialization)."""

    epoch: int
    training_loss: float
    validation_loss: float | None = None


@dataclass(frozen=True)
class FitResult:
    """Selected predictor and an auditable optimization history."""

    predictor: Predictor
    history: tuple[EpochRecord, ...]
    selected_epoch: int
    selection: str


def _prepared(region, observations, partition=None, *, role="training"):
    if partition is not None and partition.role != role:
        raise ValueError(f"Expected a {role} partition, got {partition.role}")
    pairs, values = observations.aligned_pairs(region, partition)
    if observations.target.kind != "dissimilarity" or np.any(values < 0):
        raise ValueError(
            "direct_log1p requires nonnegative dissimilarities; relatedness is unsupported"
        )
    return (
        jnp.asarray(region.features),
        jnp.asarray(region.pixel_nodes),
        pairs,
        jnp.asarray(observations.target.forward(values)),
    )


def fit(
    region: PreparedRegion,
    observations: PairwiseObservations,
    *,
    model: EmbeddingDistanceModel | None = None,
    config: FitConfig | None = None,
    validation: tuple[PreparedRegion, PairwiseObservations] | None = None,
    partition: ObservationPartition | None = None,
    validation_partition: ObservationPartition | None = None,
) -> FitResult:
    """Fit nonnegative dissimilarities with direct log1p-MSE.

    Without validation, the final fixed-budget state is returned. When validation
    is supplied, its inference objective selects among initialization and every
    updated encoder. Validation targets never enter optimization. Prediction is
    on the original declared scale, without clipping targets or an MLPE head.
    """
    config = config or FitConfig()
    training_data = _prepared(region, observations, partition)
    if validation is None and validation_partition is not None:
        raise ValueError("validation_partition requires validation observations")
    validation_data = None
    if validation is not None:
        validation_region, validation_observations = validation
        if validation_observations.target != observations.target:
            raise ValueError("Training and validation target scales must match")
        if validation_region.features.shape[-1] != region.features.shape[-1]:
            raise ValueError("Training and validation feature channels must match")
        if validation_region.feature_names != region.feature_names:
            raise ValueError(
                "Training and validation feature contracts must match meanings and order"
            )
        validation_data = _prepared(
            validation_region, validation_observations, validation_partition, role="validation"
        )
        if region.name == validation_region.name:

            def labelled_pairs(prepared_region, data):
                return {
                    frozenset(
                        (prepared_region.sampling_unit_ids[i], prepared_region.sampling_unit_ids[j])
                    )
                    for i, j in zip(*data[2], strict=True)
                }

            if labelled_pairs(region, training_data) & labelled_pairs(
                validation_region, validation_data
            ):
                raise ValueError(
                    "Training and validation observations overlap within the same region"
                )
    init_key, random_key = jax.random.split(jax.random.key(config.seed))
    model = (
        model
        if model is not None
        else UNetEmbeddingDistance(
            region.features.shape[-1],
            patch_size=1,
            base_channels=8,
            embedding_dim=4,
            dropout=0,
            key=init_key,
        )
    )
    optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def loss(encoder, data, *, inference=True, key=None):
        features, nodes, pairs, target = data
        predictions = encoder.predict_distances(features, nodes, inference=inference, key=key)[
            pairs
        ]
        return jnp.mean(jnp.square(jnp.log1p(predictions) - jnp.log1p(target)))

    def step(encoder, state, key):
        objective, gradients = eqx.filter_value_and_grad(loss)(
            encoder, training_data, inference=False, key=key
        )
        updates, state = optimizer.update(gradients, state, encoder)
        return eqx.apply_updates(encoder, updates), state, objective

    evaluate = eqx.filter_jit(loss) if config.jit else loss
    update = eqx.filter_jit(step) if config.jit else step
    history = []
    selected = model
    selected_epoch = 0
    best_loss = float("inf")
    for epoch in range(config.epochs + 1):
        if epoch:
            random_key, update_key = jax.random.split(random_key)
            model, optimizer_state, objective = update(model, optimizer_state, update_key)
            if not np.isfinite(float(objective)):
                raise FloatingPointError(f"Nonfinite training objective at epoch {epoch}")
        training_loss = float(evaluate(model, training_data))
        validation_loss = (
            None if validation_data is None else float(evaluate(model, validation_data))
        )
        if not np.isfinite(training_loss) or (
            validation_loss is not None and not np.isfinite(validation_loss)
        ):
            raise FloatingPointError(f"Nonfinite inference objective at epoch {epoch}")
        history.append(EpochRecord(epoch, training_loss, validation_loss))
        if validation_loss is None or validation_loss < best_loss:
            selected = model
            selected_epoch = epoch
            best_loss = training_loss if validation_loss is None else validation_loss
    return FitResult(
        Predictor(
            selected,
            observations.target,
            region.features.shape[-1],
            feature_names=region.feature_names,
        ),
        tuple(history),
        selected_epoch,
        "final" if validation is None else "validation",
    )
