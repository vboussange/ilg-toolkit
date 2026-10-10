"""Complete, atomic continuation checkpoints using the shared artifact format."""

from dataclasses import asdict
from importlib.metadata import version

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ._archive import ArrayWriter, ArtifactError, read_archive, runtime_versions, write_archive
from ._codecs import decode_model, decode_tree_leaves, encode_model, encode_tree_leaves
from .config import ResistanceSolverConfig, TrainingConfig
from .training import EpochRecord, TrainingState


def _runtime():
    return {
        **runtime_versions(),
        "lineax": version("lineax"),
        "backend": jax.default_backend(),
        "devices": [device.device_kind for device in jax.devices()],
        "x64": jax.config.x64_enabled,
    }


def _validate_state(state):
    if not isinstance(state, TrainingState):
        raise ArtifactError("Checkpoint requires a TrainingState returned by fit")
    if state.model_state is not None:
        raise ArtifactError("Checkpoint supports the shipped stateless encoders only")
    if not isinstance(state.config, TrainingConfig):
        raise ArtifactError("Checkpoint has invalid training configuration")
    if (
        type(state.epoch) is not int
        or type(state.step) is not int
        or not 0 <= state.epoch <= state.config.epochs
        or state.step != state.epoch
        or type(state.selected_epoch) is not int
        or not 0 <= state.selected_epoch <= state.epoch
    ):
        raise ArtifactError("Checkpoint has inconsistent epoch/step/selection progress")
    names = state.region_names
    if (
        not isinstance(names, tuple)
        or not names
        or any(not isinstance(n, str) or not n for n in names)
        or names != tuple(sorted(set(names)))
    ):
        raise ArtifactError("Checkpoint region identities must be unique and sorted")
    if (
        not isinstance(state.data_identity, str)
        or len(state.data_identity) != 64
        or any(c not in "0123456789abcdef" for c in state.data_identity)
    ):
        raise ArtifactError("Checkpoint has invalid data identity")
    if state.schedule_state != {"kind": "fixed", "learning_rate": state.config.learning_rate}:
        raise ArtifactError("Unsupported checkpoint learning-rate policy")
    if state.stopping_state != {"kind": "fixed_budget", "epochs": state.config.epochs}:
        raise ArtifactError("Unsupported checkpoint stopping policy")
    latest, best = state.latest_model, state.best_model
    if not bool(eqx.tree_equal(state.encoder, latest.encoder)):
        raise ArtifactError("Latest checkpoint encoder and model disagree")
    for model in (latest, best):
        if (
            model.objective != state.config.objective
            or model.solver_config != state.config.solver
            or set(model.training_pairs) != set(names)
            or model.target != latest.target
            or model.feature_names != latest.feature_names
            or model.feature_count != latest.feature_count
            or model.training_pairs != latest.training_pairs
            or model.validation_pairs != latest.validation_pairs
            or type(model.encoder) is not type(latest.encoder)
        ):
            raise ArtifactError("Checkpoint models have incompatible training contracts")
        if model.objective == "mlpe" and set(model.calibrations) != set(names):
            raise ArtifactError("Checkpoint is missing regional MLPE calibration")
    validation_names = set(latest.validation_pairs)
    selection = "validation" if validation_names else "final"
    if state.selection != selection or (
        selection == "final" and state.selected_epoch != state.epoch
    ):
        raise ArtifactError("Checkpoint selection policy disagrees with progress")
    if not isinstance(state.history, tuple) or len(state.history) != state.epoch + 1:
        raise ArtifactError("Checkpoint history is incomplete")
    for index, record in enumerate(state.history):
        if (
            not isinstance(record, EpochRecord)
            or type(record.epoch) is not int
            or record.epoch != index
            or not np.isfinite(record.training_loss)
            or not isinstance(record.training_by_region, dict)
            or set(record.training_by_region) != set(names)
            or not all(np.isfinite(v) for v in record.training_by_region.values())
            or not isinstance(record.validation_by_region, dict)
            or set(record.validation_by_region) != validation_names
            or not all(np.isfinite(v) for v in record.validation_by_region.values())
            or (record.validation_loss is None) != (selection == "final")
            or (record.validation_loss is not None and not np.isfinite(record.validation_loss))
        ):
            raise ArtifactError("Checkpoint history has invalid objectives or regional progress")
    selected_record = state.history[state.selected_epoch]
    selected_loss = (
        selected_record.training_loss if selection == "final" else selected_record.validation_loss
    )
    if not np.isfinite(state.best_loss) or state.best_loss != selected_loss:
        raise ArtifactError("Checkpoint best loss disagrees with its selected epoch")
    key = np.asarray(state.rng_key)
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ArtifactError("Checkpoint requires the legacy uint32[2] training RNG key")
    if state.config.objective == "mlpe":
        raw = np.asarray(state.raw_variances)
        if raw.shape != (len(names), 2) or raw.dtype != np.float64 or not np.isfinite(raw).all():
            raise ArtifactError("Checkpoint has invalid regional MLPE nuisance parameters")
    elif state.raw_variances is not None:
        raise ArtifactError("Direct checkpoint cannot contain MLPE nuisance parameters")
    if int(state.optimizer_state[0].count) != state.step:
        raise ArtifactError("Checkpoint Adam count disagrees with update progress")
    for leaf in jax.tree.leaves(state.optimizer_state):
        if eqx.is_inexact_array(leaf) and not np.isfinite(np.asarray(leaf)).all():
            raise ArtifactError("Checkpoint contains nonfinite optimizer state")


def save_checkpoint(path, state: TrainingState) -> None:
    """Atomically save one complete latest/best continuation generation.

    ``fit(..., on_epoch=lambda state: save_checkpoint(path, state))`` writes each
    completed epoch. A failed replacement leaves the previous file intact.
    Custom encoders and mutable model state are currently unsupported.
    """
    try:
        _validate_state(state)
        arrays = ArrayWriter()
        payload = {
            "checkpoint_version": 1,
            "resume_runtime": _runtime(),
            "config": asdict(state.config),
            "epoch": state.epoch,
            "step": state.step,
            "region_names": list(state.region_names),
            "data_identity": state.data_identity,
            "history": [asdict(record) for record in state.history],
            # Schema-one field names are independent of the public Python names.
            "latest_predictor": encode_model(state.latest_model, arrays),
            "best_predictor": encode_model(state.best_model, arrays),
            "raw_variances": None
            if state.raw_variances is None
            else arrays.add(state.raw_variances),
            "optimizer_state": encode_tree_leaves(state.optimizer_state, arrays),
            "rng_key": arrays.add(state.rng_key),
            "selected_epoch": state.selected_epoch,
            "best_loss": state.best_loss,
            "selection": state.selection,
            "model_state": None,
            "schedule_state": state.schedule_state,
            "stopping_state": state.stopping_state,
        }
        write_archive(path, kind="training_checkpoint", payload=payload, arrays=arrays)
    except ArtifactError:
        raise
    except (TypeError, KeyError, AttributeError, IndexError, ValueError) as error:
        raise ArtifactError(f"Invalid training checkpoint state: {error}") from error


def load_checkpoint(path) -> TrainingState:
    """Load complete state without a caller template; pass it to ``fit(state=...)``.

    Continuation requires the recorded numerical library versions, device kind,
    backend, and explicit JAX x64 setting. Identical data/configuration is checked
    by fit; only an equal or increased total epoch budget is accepted.
    """
    try:
        with read_archive(path, expected_kind="training_checkpoint") as archive:
            record = archive.payload
            expected = {
                "checkpoint_version",
                "resume_runtime",
                "config",
                "epoch",
                "step",
                "region_names",
                "data_identity",
                "history",
                "latest_predictor",
                "best_predictor",
                "raw_variances",
                "optimizer_state",
                "rng_key",
                "selected_epoch",
                "best_loss",
                "selection",
                "model_state",
                "schedule_state",
                "stopping_state",
            }
            if (
                not isinstance(record, dict)
                or set(record) != expected
                or record["checkpoint_version"] != 1
            ):
                raise ArtifactError("Unsupported or incomplete training checkpoint schema")
            if record["resume_runtime"] != _runtime():
                raise ArtifactError(
                    "Checkpoint continuation runtime differs: use the saved numerical library "
                    "versions, device/backend and JAX x64 setting"
                )
            config_record = dict(record["config"])
            config_record["solver"] = ResistanceSolverConfig(**config_record["solver"])
            config = TrainingConfig(**config_record)
            latest = decode_model(record["latest_predictor"], archive)
            best = decode_model(record["best_predictor"], archive)
            raw = (
                None
                if record["raw_variances"] is None
                else jnp.asarray(
                    archive.array(
                        record["raw_variances"],
                        shape=(len(record["region_names"]), 2),
                        dtype="float64",
                    )
                )
            )
            parameters = eqx.filter((latest.encoder, raw), eqx.is_inexact_array)
            template = optax.adam(config.learning_rate).init(parameters)
            optimizer_state = decode_tree_leaves(template, record["optimizer_state"], archive)
            for loaded, initialized in zip(
                jax.tree.leaves(optimizer_state), jax.tree.leaves(template), strict=True
            ):
                if loaded.dtype != initialized.dtype:
                    raise ArtifactError("Optimizer leaf dtype differs from its parameter template")
            state = TrainingState(
                encoder=latest.encoder,
                raw_variances=raw,
                optimizer_state=optimizer_state,
                rng_key=jnp.asarray(archive.array(record["rng_key"], shape=(2,), dtype="uint32")),
                epoch=record["epoch"],
                step=record["step"],
                config=config,
                region_names=tuple(record["region_names"]),
                data_identity=record["data_identity"],
                history=tuple(EpochRecord(**item) for item in record["history"]),
                latest_model=latest,
                best_model=best,
                selected_epoch=record["selected_epoch"],
                best_loss=record["best_loss"],
                selection=record["selection"],
                model_state=record["model_state"],
                schedule_state=record["schedule_state"],
                stopping_state=record["stopping_state"],
            )
            _validate_state(state)
            archive.finish()
            return state
    except ArtifactError:
        raise
    except (TypeError, KeyError, AttributeError, IndexError, ValueError) as error:
        raise ArtifactError(f"Invalid training checkpoint: {error}") from error
