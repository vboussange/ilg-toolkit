"""Native Equinox model leaves with the metadata needed for a load template."""

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._artifact import ArtifactError, describe, require_equal
from .calibration import DirectCalibrationModule, UnpooledMLPECalibrationModule
from .data import TargetSpec
from .mlpe import MLPECalibration, MLPEConfig
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel


def _json_bytes(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ArtifactError(f"Invalid artifact configuration or metadata: {error}") from error


def _manifest(tree):
    result = []
    for value in jax.tree.leaves(tree):
        if isinstance(value, (jax.Array, np.ndarray)):
            result.append(describe(value))
        elif isinstance(value, (bool, int, float, complex, np.generic)):
            result.append({"scalar": f"{type(value).__module__}.{type(value).__qualname__}"})
    return result


def _digest(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.digest()


def _write_native(path, tree, *, kind, metadata):
    """Write a checksummed generation; only a complete fsynced file replaces the old one."""
    header = {
        "format": "ilg-toolkit-equinox",
        "schema_version": 2,
        "kind": kind,
        "requires_x64": any(
            isinstance(value, jax.Array)
            and np.dtype(value.dtype)
            in (
                np.dtype("float64"),
                np.dtype("complex128"),
                np.dtype("int64"),
                np.dtype("uint64"),
            )
            for value in jax.tree.leaves(tree)
        ),
        "metadata": metadata,
        "leaves": _manifest(tree),
    }
    _json_bytes(header)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.TemporaryFile(dir=destination.parent) as payload:
            eqx.tree_serialise_leaves(payload, tree)
            header["payload_bytes"] = payload.tell()
            payload.seek(0)
            header["sha256"] = hashlib.sha256(_json_bytes(header) + _digest(payload)).hexdigest()
            payload.seek(0)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            with os.fdopen(descriptor, "w+b") as stream:
                stream.write(_json_bytes(header) + b"\n")
                shutil.copyfileobj(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ArtifactError("Duplicate artifact metadata keys")
    return result


def _invalid_constant(value):
    raise ArtifactError(f"Nonfinite artifact metadata: {value}")


def _read_header(stream, *, kind):
    """Validate schema, integrity and precision before deserializing JAX leaves."""
    try:
        header = json.loads(
            stream.readline(), object_pairs_hook=_unique_object, parse_constant=_invalid_constant
        )
    except (ValueError, UnicodeError) as error:
        raise ArtifactError("Unsupported or incomplete native Equinox artifact header") from error
    expected = {
        "format",
        "schema_version",
        "kind",
        "requires_x64",
        "metadata",
        "leaves",
        "payload_bytes",
        "sha256",
    }
    if not isinstance(header, dict) or header.get("format") != "ilg-toolkit-equinox":
        raise ArtifactError("Unsupported native Equinox artifact format")
    if type(header.get("schema_version")) is not int or header["schema_version"] != 2:
        raise ArtifactError("Unsupported native artifact schema version")
    if set(header) != expected or not isinstance(header["metadata"], dict):
        raise ArtifactError("Missing or invalid native artifact metadata")
    if header["kind"] != kind:
        raise ArtifactError(f"Expected {kind} artifact, found {header['kind']!r}")
    if type(header["requires_x64"]) is not bool:
        raise ArtifactError("Missing native artifact precision requirement")
    if type(header["payload_bytes"]) is not int or header["payload_bytes"] < 0:
        raise ArtifactError("Invalid artifact payload length")
    if not isinstance(header["sha256"], str) or len(header["sha256"]) != 64:
        raise ArtifactError("Invalid artifact integrity digest")
    if not isinstance(header["leaves"], list):
        raise ArtifactError("Invalid artifact native leaf manifest")
    payload_start = stream.tell()
    stream.seek(0, os.SEEK_END)
    if stream.tell() - payload_start != header["payload_bytes"]:
        raise ArtifactError("Incomplete or appended native artifact payload")
    stream.seek(payload_start)
    digest_header = {key: value for key, value in header.items() if key != "sha256"}
    actual_digest = hashlib.sha256(_json_bytes(digest_header) + _digest(stream)).hexdigest()
    if actual_digest != header["sha256"]:
        raise ArtifactError("Corrupted native artifact metadata or payload (integrity mismatch)")
    stream.seek(payload_start)
    if header["requires_x64"] and not jax.config.read("jax_enable_x64"):
        raise ArtifactError("Saved JAX leaves require JAX_ENABLE_X64=true before loading")
    return header


def _read_native(stream, template, header):
    require_equal(header["leaves"], _manifest(template), "native leaves")
    tree = eqx.tree_deserialise_leaves(stream, template)
    if stream.read(1):
        raise ArtifactError("Unexpected trailing native leaves")
    return tree


def _validate_model(model):
    if type(model.feature_count) is not int or model.feature_count < 1:
        raise ArtifactError("Model feature_count must be a positive integer")
    names = model.feature_names
    if names is not None and (
        not isinstance(names, tuple)
        or len(names) != model.feature_count
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ArtifactError("Model feature_names must match the feature count and order")
    registry = model.species
    if (
        not isinstance(registry, tuple)
        or not registry
        or len(set(registry)) != len(registry)
        or any(not isinstance(name, str) or not name for name in registry)
    ):
        raise ArtifactError("Model requires an exact nonempty species registry")
    if model.training_objective not in (None, "direct_log1p", "mlpe"):
        raise ArtifactError("Invalid model training objective")
    assignments = {}

    def record(region, species, labels):
        if not isinstance(region, str) or not region or species not in registry:
            raise ArtifactError("Invalid region/species provenance identity")
        for label in labels:
            if not isinstance(label, str) or not label:
                raise ArtifactError("Invalid sampling-unit provenance identity")
            prior = assignments.setdefault((species, label), region)
            if prior != region:
                raise ArtifactError("Biological sampling identity appears in multiple regions")

    def pairs(region, species, selected):
        canonical = []
        for pair in selected:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2 or pair[0] == pair[1]:
                raise ArtifactError("Invalid model provenance pair")
            record(region, species, pair)
            canonical.append(tuple(sorted(pair)))
        if len(set(canonical)) != len(canonical):
            raise ArtifactError("Duplicate model provenance pairs")

    for access in (model.training_pairs, model.validation_pairs):
        for region, groups in access.items():
            for species, selected in groups.items():
                pairs(region, species, selected)
    if isinstance(model.calibration, UnpooledMLPECalibrationModule):
        for region, groups in model.calibration.groups.items():
            for species, head in groups.items():
                if (head.region_name, head.species_name, head.target) != (
                    region,
                    species,
                    model.target,
                ):
                    raise ArtifactError("MLPE head identity/target differs from its group")
                record(region, species, head.population_ids)
                pairs(region, species, head.calibration_pairs)
                if any(
                    label not in head.population_ids
                    for pair in head.calibration_pairs
                    for label in pair
                ):
                    raise ArtifactError("MLPE calibration endpoints must match its populations")
                if any(
                    role not in {"training", "validation", "calibration"}
                    for role in head.calibration_roles
                ):
                    raise ArtifactError("MLPE calibration contains invalid target-access roles")
                if (
                    type(head.optimizer_iterations) is not int
                    or head.optimizer_iterations < 0
                    or type(head.converged) is not bool
                ):
                    raise ArtifactError("Invalid MLPE optimizer diagnostics")
                head.__post_init__()
    elif not isinstance(model.calibration, DirectCalibrationModule):
        raise ArtifactError("Native persistence requires direct or MLPE calibration")


def _model_metadata(model):
    """Nonnumerical contracts and tree structure; learned values remain native leaves."""
    _validate_model(model)
    if isinstance(model.calibration, DirectCalibrationModule):
        calibration = {"kind": "direct"}
    elif isinstance(model.calibration, UnpooledMLPECalibrationModule):
        calibration = {
            "kind": "mlpe",
            "groups": {
                region: {
                    name: {
                        "population_ids": value.population_ids,
                        "calibration_pairs": value.calibration_pairs,
                        "calibration_roles": value.calibration_roles,
                        "config": asdict(value.config),
                        "optimizer_message": value.optimizer_message,
                    }
                    for name, value in groups.items()
                }
                for region, groups in model.calibration.groups.items()
            },
        }
    else:
        raise ArtifactError("Native persistence requires a direct or MLPE calibration module")
    return {
        "target": asdict(model.target),
        "feature_count": model.feature_count,
        "feature_names": model.feature_names,
        "distance_model": describe(model.distance_model),
        "training_objective": model.training_objective,
        "training_pairs": model.training_pairs,
        "validation_pairs": model.validation_pairs,
        "calibration": calibration,
    }


def _head_template(region, name, metadata, target):
    config = MLPEConfig(**metadata["config"])
    population_ids = tuple(metadata["population_ids"])
    count = len(population_ids)
    return MLPECalibration(
        region_name=region,
        species_name=name,
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
    require_equal(metadata["distance_model"], describe(distance_model))
    saved_calibration = metadata["calibration"]
    if saved_calibration["kind"] == "direct":
        calibration = DirectCalibrationModule(target)
    elif saved_calibration["kind"] == "mlpe":
        calibration = UnpooledMLPECalibrationModule(
            target,
            {
                region: {
                    name: _head_template(region, name, group, target)
                    for name, group in groups.items()
                }
                for region, groups in saved_calibration["groups"].items()
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
        training_pairs=metadata["training_pairs"],
        validation_pairs=metadata["validation_pairs"],
    )


def save_model(path, model: LandscapeGeneticsModel) -> None:
    """Atomically save a calibrated model, including compatible custom distance models."""
    if not isinstance(model, LandscapeGeneticsModel):
        raise ArtifactError("save_model requires a fitted LandscapeGeneticsModel")
    try:
        _write_native(path, model, kind="model", metadata=_model_metadata(model))
    except (TypeError, ValueError, AttributeError, RecursionError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"Cannot save model configuration: {error}") from error


def load_model(path, *, distance_model: AbstractDistanceModel) -> LandscapeGeneticsModel:
    """Load native leaves into the caller's exact full distance model configuration.

    The caller supplies the distance model class, static settings and compatible leaves.
    Saved leaves replace its numerical values. Registry, structure, dtype and
    execution settings must match exactly; solver contexts are rebuilt as needed.
    """
    try:
        with Path(path).open("rb") as stream:
            header = _read_header(stream, kind="model")
            metadata = header["metadata"]
            template = _model_template(metadata, distance_model)
            model = _read_native(stream, template, header)
            require_equal(metadata["distance_model"], describe(model.distance_model))
            _validate_model(model)
            return model
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
            f"Cannot load model with the supplied distance_model template: {error}"
        ) from error
