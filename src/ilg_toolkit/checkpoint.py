"""Atomic native continuation checkpoints with independent latest/selected state."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

from ._artifact import describe, pack, require_equal, unpack
from .config import TrainingConfig
from .models import AbstractDistanceModel
from .persistence import (
    ArtifactError,
    _model_metadata,
    _model_template,
    _read_header,
    _read_native,
    _write_native,
)
from .training import StepRecord, TrainingState


def _metadata(state):
    return json.loads(
        json.dumps(
            {
                "config": asdict(state.config),
                "input_contract": pack(state.input_contract),
                "history": [pack(asdict(record)) for record in state.history],
                "step": state.step,
                "selected_step": state.selected_step,
                "distance_model": describe(state.latest_distance_model),
                "best_model": _model_metadata(state.best_model),
            },
            allow_nan=False,
        )
    )


def _validate_state(state):
    if any(type(value) is not int for value in (state.step, state.selected_step)) or not (
        0 <= state.selected_step <= state.step <= state.config.steps
    ):
        raise ArtifactError("Checkpoint step/selected_step progress is inconsistent")
    steps = tuple(record.step for record in state.history)
    if (
        not steps
        or steps[0] != 0
        or any(type(step) is not int for step in steps)
        or any(a >= b for a, b in zip(steps, steps[1:], strict=False))
        or steps[-1] > state.step
        or state.selected_step not in steps
    ):
        raise ArtifactError("Checkpoint sparse history and selected step are inconsistent")
    contract = state.input_contract
    if set(contract) != {
        "species",
        "training",
        "validation",
        "training_regions",
        "validation_regions",
    }:
        raise ArtifactError("Incomplete checkpoint input contract")
    if contract["species"] != state.latest_distance_model.species:
        raise ArtifactError("Checkpoint input contract species registry changed")
    require_equal(describe(state.latest_distance_model), describe(state.best_model.distance_model))
    expected_access = []
    for role in ("training", "validation"):
        entries = contract[role]
        if not isinstance(entries, tuple):
            raise ArtifactError("Checkpoint group catalogue must retain its ordered tuple")
        identities = [(entry["region"], entry["species"]) for entry in entries]
        if len(set(identities)) != len(identities) or identities != sorted(
            identities, key=lambda identity: (identity[0], contract["species"].index(identity[1]))
        ):
            raise ArtifactError("Checkpoint nuisance/input group order is inconsistent")
        access = {}
        for entry in entries:
            pairs = entry["pairs"] if entry["selected"] is None else entry["selected"]
            unit_indices = {label: index for index, label in enumerate(entry["units"])}
            aligned = sorted(
                pairs, key=lambda pair: tuple(sorted(unit_indices[label] for label in pair))
            )
            access.setdefault(entry["region"], {})[entry["species"]] = tuple(
                tuple(sorted(pair)) for pair in aligned
            )
            if entry["target"] != asdict(state.best_model.target):
                raise ArtifactError("Checkpoint input target differs from selected model")
        expected_access.append(access)
        if access != getattr(state.best_model, f"{role}_pairs"):
            raise ArtifactError(f"Checkpoint {role} access differs from selected model")
    training, validation = expected_access
    if not training:
        raise ArtifactError("Checkpoint training catalogue is empty")
    missing_groups = tuple(
        (region, species)
        for region, groups in training.items()
        for species in groups
        if species not in validation.get(region, {})
    )
    validation_species = {name for groups in validation.values() for name in groups}
    missing_species = tuple(name for name in contract["species"] if name not in validation_species)
    for record in state.history:
        if (
            record.validation_missing_groups != missing_groups
            or record.validation_missing_species != missing_species
        ):
            raise ArtifactError("Checkpoint history has inconsistent validation coverage")
        for role, access in (("training", training), ("validation", validation)):
            by_group = getattr(record, f"{role}_by_group")
            by_species = getattr(record, f"{role}_by_species")
            if {region: set(groups) for region, groups in by_group.items()} != {
                region: set(groups) for region, groups in access.items()
            } or set(by_species) != {name for groups in access.values() for name in groups}:
                raise ArtifactError("Checkpoint history group/species metrics are inconsistent")
            loss = getattr(record, f"{role}_loss")
            if (loss is None) != (not access):
                raise ArtifactError("Checkpoint history loss/coverage is inconsistent")
            values = [
                *by_species.values(),
                *(value for groups in by_group.values() for value in groups.values()),
            ]
            if loss is not None:
                values.append(loss)
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in values
            ):
                raise ArtifactError("Checkpoint history metrics must be finite")
    raw_values = (state.raw_variances, state.selected_raw_variances)
    if state.config.objective == "direct_log1p":
        if any(value is not None for value in raw_values):
            raise ArtifactError("Direct checkpoints cannot have MLPE nuisance parameters")
    else:
        shape = (len(contract["training"]), 2)
        if any(
            value is None
            or value.shape != shape
            or value.dtype != np.dtype("float64")
            or not np.isfinite(np.asarray(value)).all()
            for value in raw_values
        ):
            raise ArtifactError("MLPE nuisance parameters require aligned float64 group rows")
    if state.rng_key.shape != (2,) or state.rng_key.dtype != np.dtype("uint32"):
        raise ArtifactError("Checkpoint sampling/dropout random key is incompatible")
    if int(state.optimizer_state[0].count) != state.step:
        raise ArtifactError("Checkpoint Adam progress differs from optimizer step")


def save_checkpoint(path, state: TrainingState) -> None:
    """Atomically save latest parameters and the independently selected calibration.

    Use ``fit(..., on_step=lambda state: save_checkpoint(path, state))`` after
    any completed update, including updates between full evaluations.
    """
    if not isinstance(state, TrainingState):
        raise ArtifactError("save_checkpoint requires a TrainingState returned by fit")
    _validate_state(state)
    _write_native(path, state, kind="training_checkpoint", metadata=_metadata(state))


def load_checkpoint(path, *, distance_model: AbstractDistanceModel) -> TrainingState:
    """Restore exact native optimizer, random, input and sparse evaluation state.

    Continue with identical inputs and policy; only the total step budget may
    increase. The caller's complete distance model configuration must match.
    """
    try:
        with Path(path).open("rb") as stream:
            header = _read_header(stream, kind="training_checkpoint")
            metadata = header["metadata"]
            config = TrainingConfig(**metadata["config"])
            require_equal(metadata["distance_model"], describe(distance_model))
            best = _model_template(metadata["best_model"], distance_model)
            contract = unpack(metadata["input_contract"])
            raw = (
                jnp.zeros((len(contract["training"]), 2), dtype=jnp.float64)
                if config.objective == "mlpe"
                else None
            )
            parameters = distance_model if raw is None else (distance_model, raw)
            optimizer_state = optax.adam(config.learning_rate).init(
                eqx.filter(parameters, eqx.is_inexact_array)
            )
            history = tuple(StepRecord(**unpack(record)) for record in metadata["history"])
            template = TrainingState(
                step=metadata["step"],
                selected_step=metadata["selected_step"],
                latest_distance_model=distance_model,
                raw_variances=raw,
                selected_raw_variances=raw,
                optimizer_state=optimizer_state,
                rng_key=jnp.zeros(2, dtype=jnp.uint32),
                config=config,
                input_contract=contract,
                history=history,
                best_model=best,
            )
            state = _read_native(stream, template, header)
            require_equal(metadata, _metadata(state), "checkpoint")
            _validate_state(state)
            return state
    except (
        KeyError,
        IndexError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        EOFError,
    ) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load checkpoint with the supplied distance_model template: {error}"
        ) from error
