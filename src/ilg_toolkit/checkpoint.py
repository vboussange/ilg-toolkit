"""Training checkpoints saved as Equinox files for continuing optimization."""

from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import optax

from .config import TrainingConfig
from .losses import _loss_from_metadata, _loss_metadata
from .models import AbstractDistanceModel
from .persistence import (
    ArtifactError,
    _model_metadata,
    _model_template,
    _read_header,
    _read_native,
    _write_native,
)
from .training import StepRecord, TrainingState, _validate_state


def save_checkpoint(path, state: TrainingState) -> None:
    """Atomic storage of the latest training state and selected fitted model.

    ``train(..., on_step=lambda state: save_checkpoint(path, state))`` saves after
    initialization and each update. Statistical parameters, optimizer/random state,
    and the selected model retain their respective training steps. History,
    configuration, loss-specific state, and training/validation group identities
    are included. The parent directory must exist; writes atomically replace the
    destination after flushing a temporary file.

    ``state.latest_model`` matches the current distance and statistical parameters;
    continuation uses those parameters with optimizer and random state.
    ``state.selected_model`` is the evaluated fitted model retained for prediction,
    and can come from an earlier step. Supply the original inputs and validation
    arguments when resuming.

    Args:
        path (str | Path): Destination checkpoint file in an existing directory.
        state: Training state returned by ``train``.

    Raises:
        ArtifactError: The state is incompatible with checkpoint persistence.

    Notes:
        No observation data or rasters are embedded in the checkpoint.

    Examples:
        >>> save_checkpoint("training.eqx", state)
        >>> restored = load_checkpoint("training.eqx", distance_model=distance_model_template)
        >>> continued = train(region, observations, state=restored,
        ...                   config=dataclasses.replace(restored.config, steps=1000))
    """
    if not isinstance(state, TrainingState):
        raise ArtifactError("save_checkpoint requires a TrainingState returned by train")
    _validate_state(state)
    options = asdict(state.config)
    options["loss"] = _loss_metadata(state.config.loss)
    metadata = {
        "config": options,
        "loss_state": state.config.loss._state_metadata(state.loss_state),
        "history": [asdict(record) for record in state.history],
        "training_groups": state.training_groups,
        "validation_groups": state.validation_groups,
        "latest_model": _model_metadata(state.latest_model),
        "selected_model": _model_metadata(state.selected_model),
    }
    _write_native(path, state, kind="training_checkpoint", metadata=metadata)


def load_checkpoint(path, *, distance_model: AbstractDistanceModel) -> TrainingState:
    """A saved training state restored with a caller-constructed distance model template.

    The template supplies the model class, static settings, numerical shapes, and
    dtypes; float64 checkpoints require float64 enabled before loading. Continuation
    uses unchanged inputs and execution settings, allowing only the total step
    budget to increase. The built-in loss reconstructs its own statistical-state
    template from checkpoint metadata. Checkpoint persistence supports built-in
    losses; custom distance models still use caller-constructed templates.

    Match the original model's static settings (including patch size, species count,
    dropout, and patch microbatch size). Reproducible continuation also requires
    the same inputs, observations, validation arguments, library versions, device,
    and execution settings. An already-exhausted unchanged budget returns the saved
    state. See ``save_checkpoint`` and the [ensemble workflow](../examples/ensemble.md)
    for continuation examples.

    Args:
        path (str | Path): Checkpoint file previously written by ``save_checkpoint``.
        distance_model: Template matching the saved model's class, static settings,
            array shapes, and dtypes.

    Returns:
        Training state containing its latest parameters, selected model, and history.

    Raises:
        ArtifactError: The artifact or supplied template is incompatible.
    """
    try:
        with Path(path).open("rb") as stream:
            metadata, weak_leaves = _read_header(stream, kind="training_checkpoint")
            options = dict(metadata["config"])
            options["loss"] = _loss_from_metadata(options["loss"])
            config = TrainingConfig(**options)
            latest = _model_template(metadata["latest_model"], distance_model)
            selected = _model_template(metadata["selected_model"], distance_model)
            loss_state = config.loss._state_template(metadata["loss_state"])
            optimizer_state = optax.adam(config.learning_rate).init(
                eqx.filter((latest.distance_model, loss_state), eqx.is_inexact_array)
            )
            history = tuple(StepRecord(**record) for record in metadata["history"])
            if not 1 <= len(history) <= config.steps + 1:
                raise ArtifactError("Checkpoint history length must be within the step budget")
            template = TrainingState(
                loss_state=loss_state,
                optimizer_state=optimizer_state,
                rng_key=jnp.zeros(2, dtype=jnp.uint32),
                config=config,
                history=history,
                latest_model=latest,
                selected_model=selected,
                selected_step=0,
                step=0,
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
            f"Cannot load checkpoint with the supplied distance_model template: {error}"
        ) from error
