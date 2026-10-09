"""In-memory direct-regression training independent of benchmark orchestration."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig
from .data import PairwiseObservations, PreparedRegion
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


def _prepared(region, observations):
    if observations.target.kind != "dissimilarity" or np.any(observations.values < 0):
        raise ValueError(
            "direct_log1p requires nonnegative dissimilarities; relatedness is unsupported"
        )
    values = observations.aligned_values(region)
    pairs = np.triu_indices(len(region.sampling_unit_ids), 1)
    return (
        jnp.asarray(region.features),
        jnp.asarray(region.pixel_nodes),
        pairs,
        jnp.asarray(values[pairs]),
    )


def fit(
    region: PreparedRegion,
    observations: PairwiseObservations,
    *,
    model: EmbeddingDistanceModel | None = None,
    config: FitConfig | None = None,
    validation: tuple[PreparedRegion, PairwiseObservations] | None = None,
) -> FitResult:
    """Fit nonnegative dissimilarities with direct log1p-MSE.

    Without validation, the final fixed-budget state is returned. When validation
    is supplied, its inference objective selects among initialization and every
    updated encoder. Validation targets never enter optimization. Prediction is
    on the original declared scale, without clipping targets or an MLPE head.
    """
    config = config or FitConfig()
    training_data = _prepared(region, observations)
    validation_data = None
    if validation is not None:
        validation_region, validation_observations = validation
        if validation_observations.target != observations.target:
            raise ValueError("Training and validation target scales must match")
        if validation_region.features.shape[-1] != region.features.shape[-1]:
            raise ValueError("Training and validation feature channels must match")
        validation_data = _prepared(validation_region, validation_observations)
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
        Predictor(selected, observations.target, region.features.shape[-1]),
        tuple(history),
        selected_epoch,
        "final" if validation is None else "validation",
    )
