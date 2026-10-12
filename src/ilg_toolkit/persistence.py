"""Native Equinox model leaves with the metadata needed for a load template."""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._archive import ArtifactError
from .config import ResistanceSolverConfig
from .data import TargetSpec
from .mlpe import MLPEConfig, MLPEHead
from .model import CalibratedModel
from .models import ConductanceModel, EmbeddingDistanceModel


def _write_native(path, tree, *, kind, metadata):
    """Atomically replace a header and native leaves; retain the previous file on failure."""
    header = {
        "format": "ilg-toolkit-equinox",
        "version": 1,
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
    """Read the format boundary before native loading can convert any JAX leaves."""
    try:
        header = json.loads(stream.readline())
    except (ValueError, UnicodeError) as error:
        raise ArtifactError("Unsupported or incomplete native Equinox artifact header") from error
    if (
        not isinstance(header, dict)
        or header.get("format") != "ilg-toolkit-equinox"
        or type(header.get("version")) is not int
        or header["version"] != 1
    ):
        raise ArtifactError("Unsupported native Equinox artifact format or version")
    if header.get("kind") != kind:
        raise ArtifactError(f"Expected {kind} artifact, found {header.get('kind')!r}")
    if type(header.get("requires_x64")) is not bool:
        raise ArtifactError("Missing native artifact precision requirement")
    if header["requires_x64"] and not jax.config.read("jax_enable_x64"):
        raise ArtifactError("Saved JAX leaves require JAX_ENABLE_X64=true before loading")
    return header["metadata"]


def _model_metadata(model):
    """Nonnumerical contracts and tree structure; learned values remain native leaves."""
    return {
        "target": asdict(model.target),
        "feature_count": model.feature_count,
        "feature_names": model.feature_names,
        "solver_config": asdict(model.solver_config),
        "objective": model.objective,
        "training_pairs": model.training_pairs,
        "validation_pairs": model.validation_pairs,
        "calibrations": {
            name: {
                "population_ids": head.population_ids,
                "calibration_pairs": head.calibration_pairs,
                "calibration_roles": head.calibration_roles,
                "config": asdict(head.config),
                "optimizer_message": head.optimizer_message,
            }
            for name, head in model.calibrations.items()
        },
    }


def _head_template(name, metadata, target):
    config = MLPEConfig(**metadata["config"])
    population_ids = tuple(metadata["population_ids"])
    count = len(population_ids)
    return MLPEHead(
        region_name=name,
        target=target,
        population_ids=population_ids,
        score_center=0.0,
        score_scale=max(1.0, 2 * config.min_score_scale),
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


def _model_template(metadata, encoder):
    target = TargetSpec(**metadata["target"])
    return CalibratedModel(
        encoder,
        target,
        metadata["feature_count"],
        feature_names=(
            None if metadata["feature_names"] is None else tuple(metadata["feature_names"])
        ),
        solver_config=ResistanceSolverConfig(**metadata["solver_config"]),
        objective=metadata["objective"],
        training_pairs=metadata["training_pairs"],
        validation_pairs=metadata["validation_pairs"],
        calibrations={
            name: _head_template(name, head, target)
            for name, head in metadata["calibrations"].items()
        },
    )


def save_model(path, model: CalibratedModel) -> None:
    """Atomically save a calibrated model, including compatible custom encoders."""
    if not isinstance(model, CalibratedModel):
        raise ArtifactError("save_model requires a fitted CalibratedModel")
    _write_native(path, model, kind="model", metadata=_model_metadata(model))


def load_model(path, *, encoder: ConductanceModel | EmbeddingDistanceModel) -> CalibratedModel:
    """Load native leaves into the caller's exact full encoder configuration.

    The caller supplies the encoder class, static settings and compatible leaves.
    Saved leaves replace its numerical values. Static settings cannot be inferred
    from numerical shapes and remain the caller's responsibility.
    """
    try:
        with Path(path).open("rb") as stream:
            metadata = _read_header(stream, kind="model")
            template = _model_template(metadata, encoder)
            return eqx.tree_deserialise_leaves(stream, template)
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load model with the supplied encoder template: {error}"
        ) from error
