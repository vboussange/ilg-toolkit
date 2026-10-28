"""Checkpoints: saved training states for continuing training."""

import os
from dataclasses import asdict, fields
from pathlib import Path

import equinox as eqx
import jax
import optax

from ._checks import _require_float64
from .config import TrainingConfig
from .losses import _loss_from_metadata, _loss_metadata
from .models import AbstractDistanceModel
from .persistence import (
    _model_metadata,
    _model_template,
    _read_header,
    _read_native,
    _write_native,
    ArtifactError,
)
from .training import _optimizer_state_signature, _validate_state, StepRecord, TrainingState


def save_checkpoint(path: str | os.PathLike[str], state: TrainingState) -> None:
    """Saves a training state to a checkpoint file atomically.

    The checkpoint holds the latest parameters, statistical parameter state,
    optimizer and random state, history, training configuration, convergence,
    training and validation groups, and the latest and selected models. It holds
    no optimizer, observations or rasters: [`load_checkpoint`][ilg_toolkit.load_checkpoint]
    takes the optimizer, and continuation takes the original inputs. The write
    goes to a temporary file that atomically replaces the destination, so a
    failed write keeps the previous file.
    ``train(..., on_step=lambda state: save_checkpoint(path, state))`` saves after
    initialization and after each update.

    Args:
        path: Destination file in an existing directory.
        state: Training state returned by [`train`][ilg_toolkit.train].

    Raises:
        ArtifactError: If ``state`` is not a training state.
        ValueError: If the state's steps, history or convergence are inconsistent,
            or its loss is not [`Log1pLoss`][ilg_toolkit.Log1pLoss] or
            [`MLPELoss`][ilg_toolkit.MLPELoss].

    Examples:
        >>> save_checkpoint("training.eqx", state)
        >>> restored = load_checkpoint(
        ...     "training.eqx", distance_model=template, optimizer=optax.adam(1e-3)
        ... )
        >>> continued = train(region, observations, state=restored,
        ...                   config=dataclasses.replace(restored.config, steps=1000))
    """
    if not isinstance(state, TrainingState):
        raise ArtifactError("save_checkpoint requires a TrainingState returned by train")
    _validate_state(state)
    options = {
        field.name: getattr(state.config, field.name)
        for field in fields(state.config)
        if field.name != "optimizer"
    }
    options["loss"] = _loss_metadata(state.config.loss)
    metadata = {
        "config": options,
        "optimizer_state": _optimizer_state_signature(state.optimizer_state),
        "loss_state": state.config.loss._state_metadata(state.loss_state),
        "history": [asdict(record) for record in state.history],
        "training_groups": state.training_groups,
        "validation_groups": state.validation_groups,
        "latest_model": _model_metadata(state.latest_model),
        "selected_model": _model_metadata(state.selected_model),
    }
    _write_native(path, state, kind="training_checkpoint", metadata=metadata)


def load_checkpoint[DistanceModel: AbstractDistanceModel](
    path: str | os.PathLike[str],
    *,
    distance_model: DistanceModel,
    optimizer: optax.GradientTransformation,
) -> TrainingState[DistanceModel]:
    """Loads a training state from a checkpoint file.

    The distance-model template supplies the distance model's class, static
    settings, array shapes and dtypes: build it with the original constructor
    settings (patch size, patch microbatch size, dropout, species and activation
    functions). The optimizer rebuilds the optimizer-state template, whose state
    types, leaf paths, shapes and dtypes match the saved optimizer state, and becomes
    the optimizer of the restored training configuration. Continuing with the
    original inputs, validation arguments, library versions, device and execution
    settings reproduces the uninterrupted run; continuation accepts a larger step
    budget and an optimizer whose state has the saved structure. Continuing with
    the saved step budget returns the saved state.

    Args:
        path: File written by [`save_checkpoint`][ilg_toolkit.save_checkpoint].
        distance_model: Distance-model template matching the saved model's class,
            static settings, array shapes and dtypes.
        optimizer: Optax optimizer whose state matches the saved optimizer state,
            such as the optimizer the checkpoint was trained with.

    Returns:
        The saved training state.

    Raises:
        ArtifactError: If the file, the template or the optimizer is incompatible.
        RuntimeError: If float64 is disabled.
    """
    _require_float64()
    try:
        with Path(path).open("rb") as stream:
            metadata, weak_leaves = _read_header(stream, kind="training_checkpoint")
            options = dict(metadata["config"])
            options["loss"] = _loss_from_metadata(options["loss"])
            config = TrainingConfig(**options, optimizer=optimizer)
            latest = _model_template(metadata["latest_model"], distance_model)
            selected = _model_template(metadata["selected_model"], distance_model)
            loss_state = config.loss._state_template(metadata["loss_state"])
            optimizer_state = optimizer.init(
                eqx.filter((latest.distance_model, loss_state), eqx.is_inexact_array)
            )
            if _optimizer_state_signature(optimizer_state) != metadata["optimizer_state"]:
                raise ArtifactError(
                    "The supplied optimizer's state does not match the checkpoint's optimizer "
                    "state (state types, leaf paths, shapes and dtypes); supply an optimizer "
                    "with the same state structure, such as the optimizer the checkpoint was "
                    "trained with"
                )
            history = tuple(StepRecord(**record) for record in metadata["history"])
            if not 1 <= len(history) <= config.steps + 1:
                raise ArtifactError("Checkpoint history length must be within the step budget")
            template = TrainingState(
                loss_state=loss_state,
                optimizer_state=optimizer_state,
                rng_key=jax.random.key(0),
                config=config,
                history=history,
                latest_model=latest,
                selected_model=selected,
                selected_step=0,
                step=0,
                converged=False,
                training_groups=tuple(tuple(group) for group in metadata["training_groups"]),
                validation_groups=tuple(tuple(group) for group in metadata["validation_groups"]),
            )
            state = _read_native(stream, template, weak_leaves)
            _validate_state(state)
            return state
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load checkpoint with the supplied distance_model template and optimizer: "
            f"{error}"
        ) from error
