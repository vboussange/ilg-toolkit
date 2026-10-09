"""In-memory direct-regression training independent of benchmark orchestration."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import FitConfig
from .data import PairwiseObservations, PreparedRegion
from .models import ConductanceModel, EmbeddingDistanceModel, UNetEmbeddingDistance
from .predictor import Predictor
from .solver import build_solver_context


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
    model: ConductanceModel | EmbeddingDistanceModel | None = None,
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

    context = None
    validation_context = None
    if isinstance(model, ConductanceModel):

        def regional_context(prepared_region):
            height, width = prepared_region.features.shape[:2]
            if height % model.patch_size or width % model.patch_size:
                raise ValueError("Raster dimensions must be divisible by model patch_size")
            return build_solver_context(
                (height // model.patch_size, width // model.patch_size), config.solver
            )

        context = regional_context(region)
        if validation is not None:
            validation_context = regional_context(validation[0])

    def loss(encoder, data, *, context=None, inference=True, key=None):
        features, nodes, pairs, target = data
        options = dict(inference=inference, key=key)
        if isinstance(encoder, ConductanceModel):
            options["context"] = context
        predictions = encoder.predict_distances(features, nodes, **options)[pairs]
        return jnp.mean(jnp.square(jnp.log1p(predictions) - jnp.log1p(target)))

    def step(encoder, state, key):
        objective, gradients = eqx.filter_value_and_grad(loss)(
            encoder, training_data, context=context, inference=False, key=key
        )
        bad_gradient = jnp.array(False)
        for leaf in jax.tree.leaves(gradients):
            if eqx.is_inexact_array(leaf):
                bad_gradient = bad_gradient | jnp.any(~jnp.isfinite(leaf))
        gradients = eqx.error_if(gradients, bad_gradient, "Nonfinite encoder gradient")
        updates, state = optimizer.update(gradients, state, encoder)
        return eqx.apply_updates(encoder, updates), state, objective

    evaluate = eqx.filter_jit(loss) if config.jit else loss
    update = eqx.filter_jit(step) if config.jit else step
    history = []
    selected = model
    selected_epoch = 0
    best_loss = float("inf")

    def checked_call(function, *args, regional_name, epoch, **kwargs):
        try:
            return jax.block_until_ready(function(*args, **kwargs))
        except RuntimeError as error:
            if not isinstance(model, ConductanceModel):
                raise
            raise RuntimeError(
                f"{regional_name}: resistance calculation failed at epoch {epoch}; "
                f"solver={config.solver}"
            ) from error

    for epoch in range(config.epochs + 1):
        if epoch:
            random_key, update_key = jax.random.split(random_key)
            model, optimizer_state, objective = checked_call(
                update, model, optimizer_state, update_key, regional_name=region.name, epoch=epoch
            )
            if not np.isfinite(float(objective)):
                raise FloatingPointError(f"Nonfinite training objective at epoch {epoch}")
        training_loss = float(
            checked_call(
                evaluate,
                model,
                training_data,
                context=context,
                regional_name=region.name,
                epoch=epoch,
            )
        )
        validation_loss = (
            None
            if validation_data is None
            else float(
                checked_call(
                    evaluate,
                    model,
                    validation_data,
                    context=validation_context,
                    regional_name=validation[0].name,
                    epoch=epoch,
                )
            )
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
            selected, observations.target, region.features.shape[-1], solver_config=config.solver
        ),
        tuple(history),
        selected_epoch,
        "final" if validation is None else "validation",
    )
