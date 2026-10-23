"""Landscape genetics models saved as Equinox files with prediction metadata."""

import json
import os
import tempfile
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .calibration import DirectCalibrationModule, GroupKey, MLPECalibrationModule
from .config import ResistanceSolverConfig
from .data import TargetSpec
from .mlpe import MLPEConfig, RegionalMLPECalibration
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel, ConductanceModel


class ArtifactError(ValueError):
    """An incompatible saved model, checkpoint, or loading template."""


def _write_native(path, tree, *, kind, metadata):
    """Atomically replace a header and native leaves; retain the previous file on failure."""
    header = {
        "format": "ilg-toolkit-equinox",
        "kind": kind,
        "requires_x64": any(
            isinstance(value, jax.Array)
            and np.dtype(value.dtype)
            in (np.dtype("float64"), np.dtype("complex128"), np.dtype("int64"), np.dtype("uint64"))
            for value in jax.tree.leaves(tree)
        ),
        "metadata": metadata,
    }
    destination = Path(path)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        with os.fdopen(descriptor, "w+b") as stream:
            stream.write(json.dumps(header, allow_nan=False).encode("utf-8") + b"\n")
            eqx.tree_serialise_leaves(stream, tree)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _read_header(stream, *, kind):
    """Read the file format and precision requirements before loading model values."""
    try:
        header = json.loads(stream.readline())
    except (ValueError, UnicodeError) as error:
        raise ArtifactError("Unsupported or incomplete native Equinox artifact header") from error
    if not isinstance(header, dict) or header.get("format") != "ilg-toolkit-equinox":
        raise ArtifactError("Unsupported native Equinox artifact format")
    if header.get("kind") != kind:
        raise ArtifactError(f"Expected {kind} artifact, found {header.get('kind')!r}")
    if type(header.get("requires_x64")) is not bool:
        raise ArtifactError("Missing native artifact precision requirement")
    if header["requires_x64"] and not jax.config.read("jax_enable_x64"):
        raise ArtifactError("Saved JAX leaves require JAX_ENABLE_X64=true before loading")
    return header["metadata"]


def _model_metadata(model):
    """Target, feature, calibration, and observation-use metadata for a saved model."""
    if isinstance(model.calibration, DirectCalibrationModule):
        calibration = {"kind": "direct"}
    elif isinstance(model.calibration, MLPECalibrationModule):
        calibration = {
            "kind": "mlpe",
            "regions": _encode_groups(
                {
                    name: {
                        "population_ids": value.population_ids,
                        "calibration_pairs": value.calibration_pairs,
                        "calibration_roles": value.calibration_roles,
                        "config": asdict(value.config),
                        "optimizer_message": value.optimizer_message,
                    }
                    for name, value in model.calibration.regions.items()
                }
            ),
        }
    else:
        raise ArtifactError("Native persistence requires a direct or MLPE calibration module")
    return {
        "target": asdict(model.target),
        "feature_count": model.feature_count,
        "feature_names": model.feature_names,
        "distance_solver": (
            asdict(model.distance_model.solver_config)
            if isinstance(model.distance_model, ConductanceModel)
            else None
        ),
        "training_objective": model.training_objective,
        "species": model.species,
        "training_pairs": _encode_groups(model.training_pairs),
        "validation_pairs": _encode_groups(model.validation_pairs),
        "calibration": calibration,
    }


def _encode_groups(values):
    if all(isinstance(key, str) for key in values):
        return values
    return [
        {"region": region, "species_index": index, "value": value}
        for (region, index), value in values.items()
    ]


def _decode_groups(values) -> dict[GroupKey, Any]:
    if isinstance(values, dict):
        return values
    return {(item["region"], item["species_index"]): item["value"] for item in values}


def _head_template(name, metadata, target):
    config = MLPEConfig(**metadata["config"])
    population_ids = tuple(metadata["population_ids"])
    count = len(population_ids)
    return RegionalMLPECalibration(
        region_name=name if isinstance(name, str) else name[0],
        target=target,
        population_ids=population_ids,
        distance_center=0.0,
        distance_scale=max(1.0, 2 * config.min_distance_scale),
        intercept=0.0,
        slope=0.0,
        unit_variance=config.variance_floor,
        residual_variance=config.variance_floor,
        effect_mean=jnp.zeros(count, dtype=jnp.float64),
        effect_covariance=jnp.eye(count, dtype=jnp.float64),
        effect_precision_cholesky=jnp.eye(count, dtype=jnp.float64),
        calibration_pairs=tuple(tuple(pair) for pair in metadata["calibration_pairs"]),
        calibration_roles=tuple(metadata["calibration_roles"]),
        ml_log_likelihood=0.0,
        config=config,
        optimizer_iterations=0,
        optimizer_message=metadata["optimizer_message"],
        converged=False,
    )


def _model_template(metadata, distance_model):
    target = TargetSpec(**metadata["target"])
    solver_settings = metadata["distance_solver"]
    if isinstance(distance_model, ConductanceModel):
        if solver_settings is None:
            raise ArtifactError("Saved distance model requires a non-conductance template")
        distance_model = copy(distance_model)
        object.__setattr__(
            distance_model, "solver_config", ResistanceSolverConfig(**solver_settings)
        )
    elif solver_settings is not None:
        raise ArtifactError("Saved conductance model requires a conductance template")
    saved_calibration = metadata["calibration"]
    if saved_calibration["kind"] == "direct":
        calibration = DirectCalibrationModule(target)
    elif saved_calibration["kind"] == "mlpe":
        calibration = MLPECalibrationModule(
            target,
            {
                name: _head_template(name, regional, target)
                for name, regional in _decode_groups(saved_calibration["regions"]).items()
            },
        )
    else:
        raise ArtifactError("Unsupported calibration module in native artifact")
    return LandscapeGeneticsModel(
        distance_model,
        calibration,
        metadata["feature_count"],
        feature_names=(
            None if metadata["feature_names"] is None else tuple(metadata["feature_names"])
        ),
        training_objective=metadata["training_objective"],
        species=None if metadata["species"] is None else tuple(metadata["species"]),
        training_pairs=_decode_groups(metadata["training_pairs"]),
        validation_pairs=_decode_groups(metadata["validation_pairs"]),
    )


def save_model(path, model: LandscapeGeneticsModel) -> None:
    """Atomically save a calibrated model, including compatible custom distance models."""
    if not isinstance(model, LandscapeGeneticsModel):
        raise ArtifactError("save_model requires a fitted LandscapeGeneticsModel")
    _write_native(path, model, kind="model", metadata=_model_metadata(model))


def load_model(path, *, distance_model: AbstractDistanceModel) -> LandscapeGeneticsModel:
    """Load a landscape genetics model using a caller-constructed distance model.

    The template supplies the model class, static settings, and compatible
    numerical shapes and dtypes. Saved parameters and conductance solver settings
    replace the template's values. Solver preparation is rebuilt for each region.
    """
    try:
        with Path(path).open("rb") as stream:
            metadata = _read_header(stream, kind="model")
            template = _model_template(metadata, distance_model)
            return eqx.tree_deserialise_leaves(stream, template)
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load model with the supplied distance_model template: {error}"
        ) from error
