"""Training checkpoints saved as Equinox files for continuing a fit."""

from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax

from .config import TrainingConfig
from .mlpe import MLPEConfig
from .models import AbstractDistanceModel
from .persistence import (
    ArtifactError,
    _model_metadata,
    _model_template,
    _read_header,
    _write_native,
)
from .training import StepRecord, TrainingState, _validate_state


def save_checkpoint(path, state: TrainingState) -> None:
    """Save the latest training state and selected model in an Equinox checkpoint.

    ``fit(..., on_step=lambda state: save_checkpoint(path, state))`` saves after
    initialization and every optimizer update. A failed replacement leaves the prior file intact.
    """
    if not isinstance(state, TrainingState):
        raise ArtifactError("save_checkpoint requires a TrainingState returned by fit")
    _validate_state(state)
    metadata = {
        "config": asdict(state.config),
        "history_length": len(state.history),
        "latest_model": _model_metadata(state.latest_model),
        "selected_model": _model_metadata(state.selected_model),
    }
    _write_native(path, state, kind="training_checkpoint", metadata=metadata)


def load_checkpoint(path, *, distance_model: AbstractDistanceModel) -> TrainingState:
    """Load a training state using a caller-constructed distance model.

    The template supplies the model class, static settings, and numerical shapes
    and dtypes. Reproducing uninterrupted training requires the original inputs
    and execution conditions. Only the total step budget may increase.
    """
    try:
        with Path(path).open("rb") as stream:
            metadata = _read_header(stream, kind="training_checkpoint")
            options = dict(metadata["config"])
            options["mlpe"] = MLPEConfig(**options["mlpe"])
            config = TrainingConfig(**options)
            latest = _model_template(metadata["latest_model"], distance_model)
            selected = _model_template(metadata["selected_model"], distance_model)
            raw = (
                {name: jnp.zeros(2, dtype=jnp.float64) for name in latest.training_pairs}
                if config.objective == "mlpe"
                else None
            )
            optimizer_state = optax.adam(config.learning_rate).init(
                eqx.filter((latest.distance_model, raw), eqx.is_inexact_array)
            )
            history_length = metadata["history_length"]
            if not 1 <= history_length <= config.steps + 1:
                raise ArtifactError("Checkpoint history length must be within the step budget")
            history = tuple(
                StepRecord(
                    step=0,
                    training_loss=0.0,
                    validation_loss=0.0 if latest.validation_pairs else None,
                    training_by_region=dict.fromkeys(latest.training_pairs, 0.0),
                    validation_by_region=dict.fromkeys(latest.validation_pairs, 0.0),
                )
                for _ in range(history_length)
            )
            template = TrainingState(
                raw_variances=raw,
                optimizer_state=optimizer_state,
                rng_key=jnp.zeros(2, dtype=jnp.uint32),
                config=config,
                history=history,
                latest_model=latest,
                selected_model=selected,
                selected_step=0,
                step=0,
            )
            state = eqx.tree_deserialise_leaves(stream, template)
            _validate_state(state)
            return state
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load checkpoint with the supplied distance_model template: {error}"
        ) from error
