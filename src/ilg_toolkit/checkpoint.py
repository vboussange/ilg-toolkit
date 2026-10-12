"""Atomic native Equinox continuation checkpoints with caller encoder templates."""

from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax

from .config import TrainingConfig
from .models import AbstractDistanceModel
from .persistence import (
    ArtifactError,
    _model_metadata,
    _model_template,
    _read_header,
    _write_native,
)
from .training import EpochRecord, TrainingState


def save_checkpoint(path, state: TrainingState) -> None:
    """Atomically save the actual latest/best state, including custom encoders.

    ``fit(..., on_epoch=lambda state: save_checkpoint(path, state))`` saves after
    every complete epoch. A failed replacement leaves the prior file intact.
    """
    if not isinstance(state, TrainingState):
        raise ArtifactError("save_checkpoint requires a TrainingState returned by fit")
    metadata = {
        "config": asdict(state.config),
        "input_contract": state.input_contract,
        "history_length": len(state.history),
        "latest_model": _model_metadata(state.latest_model),
        "best_model": _model_metadata(state.best_model),
    }
    _write_native(path, state, kind="training_checkpoint", metadata=metadata)


def load_checkpoint(path, *, encoder: AbstractDistanceModel) -> TrainingState:
    """Restore native state using the exact full encoder configuration and precision.

    Continue with the original inputs and partitions. Only the total epoch budget
    may increase; unchanged measurements and execution conditions remain caller
    responsibilities. Numerical leaf compatibility is checked by Equinox.
    """
    try:
        with Path(path).open("rb") as stream:
            metadata = _read_header(stream, kind="training_checkpoint")
            options = dict(metadata["config"])
            config = TrainingConfig(**options)
            latest = _model_template(metadata["latest_model"], encoder)
            best = _model_template(metadata["best_model"], encoder)
            raw = (
                jnp.zeros((len(latest.training_pairs), 2), dtype=jnp.float64)
                if config.objective == "mlpe"
                else None
            )
            optimizer_state = optax.adam(config.learning_rate).init(
                eqx.filter((latest.encoder, raw), eqx.is_inexact_array)
            )
            history_length = metadata["history_length"]
            if history_length < 1:
                raise ArtifactError("Checkpoint history is incomplete")
            history = tuple(
                EpochRecord(
                    epoch=0,
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
                input_contract=metadata["input_contract"],
                history=history,
                latest_model=latest,
                best_model=best,
                selected_epoch=0,
            )
            return eqx.tree_deserialise_leaves(stream, template)
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load checkpoint with the supplied encoder template: {error}"
        ) from error
