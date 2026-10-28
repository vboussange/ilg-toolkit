"""Landscape genetics models saved as Equinox files with prediction metadata.

Calibration is direct or maximum-likelihood population effects (MLPE).
"""

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

from ._checks import _require_float64
from ._training_utils import RegionSpeciesKey
from .calibration import DirectCalibrationModule, MLPECalibrationModule
from .config import ResistanceSolverConfig
from .data import TargetSpec
from .losses import MLPELoss
from .mlpe import RegionalMLPECalibration
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel, ConductanceModel


class ArtifactError(ValueError):
    """An incompatible saved model, checkpoint, or loading template."""


def _is_typed_key(leaf) -> bool:
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)


def _key_data(tree):
    """The tree with typed random keys replaced by their raw key data, which NumPy can store."""
    return jax.tree.map(
        lambda leaf: jax.random.key_data(leaf) if _is_typed_key(leaf) else leaf, tree
    )


def _write_native(path, tree, *, kind, metadata):
    """Atomic storage of metadata and model arrays, retaining the previous file if writing fails."""
    tree = _key_data(tree)
    arrays = [
        (index, leaf)
        for index, leaf in enumerate(jax.tree.leaves(tree))
        if isinstance(leaf, jax.Array)
    ]
    header = {
        "format": "ilg-toolkit-equinox",
        "kind": kind,
        # NumPy storage drops JAX weak types, which change dtype promotion.
        "weak_leaves": [index for index, leaf in arrays if leaf.weak_type],
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


def _read_native(stream, template, weak_leaves):
    """Saved leaves loaded into a template, with the weak types recorded when saving."""
    loaded = eqx.tree_deserialise_leaves(stream, _key_data(template))
    leaves, structure = jax.tree.flatten(loaded)
    for index in weak_leaves:
        if not 0 <= index < len(leaves) or not isinstance(leaves[index], jax.Array):
            raise ArtifactError(f"Recorded weakly typed leaf {index} is not a saved JAX array")
        leaves[index] = _weakly_typed(leaves[index])
    return jax.tree.map(
        lambda leaf, reference: (
            jax.random.wrap_key_data(leaf, impl=jax.random.key_impl(reference))
            if _is_typed_key(reference)
            else leaf
        ),
        jax.tree.unflatten(structure, leaves),
        template,
    )


def _weakly_typed(leaf):
    # Only Python scalars create weak types publicly: write the values into
    # weakly typed zeros of the same kind, whose dtype follows the current precision.
    zeros = jax.lax.full(leaf.shape, leaf.dtype.type(0).item())
    if not zeros.weak_type or zeros.dtype != leaf.dtype:
        raise ArtifactError(
            f"Cannot restore a weakly typed {leaf.dtype} leaf under the current JAX precision"
        )
    return zeros.at[...].set(leaf)


def _read_header(stream, *, kind):
    """Metadata and weakly typed leaves, after checking the format and file kind."""
    try:
        header = json.loads(stream.readline())
    except (ValueError, UnicodeError) as error:
        raise ArtifactError("Unsupported or incomplete saved-model or checkpoint header") from error
    if not isinstance(header, dict) or header.get("format") != "ilg-toolkit-equinox":
        raise ArtifactError("Unsupported saved-model or checkpoint format")
    if header.get("kind") != kind:
        raise ArtifactError(f"Expected {kind} file, found {header.get('kind')!r}")
    weak_leaves = header.get("weak_leaves")
    if not isinstance(weak_leaves, list) or any(type(index) is not int for index in weak_leaves):
        raise ArtifactError("Missing or invalid saved-model or checkpoint weak-type record")
    return header["metadata"], weak_leaves


def _model_metadata(model):
    """Target, feature, and fitted calibration metadata for a saved model."""
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
                        "loss": asdict(value.loss),
                        "optimizer_message": value.optimizer_message,
                    }
                    for name, value in model.calibration.regions.items()
                }
            ),
        }
    else:
        raise ArtifactError("Saving a model requires a direct or MLPE calibration module")
    return {
        "target": asdict(model.target),
        "feature_names": model.feature_names,
        "distance_solver": (
            asdict(model.distance_model.solver_config)
            if isinstance(model.distance_model, ConductanceModel)
            else None
        ),
        "species": model.species,
        "calibration": calibration,
    }


def _encode_groups(values):
    return [
        {"region": region, "species_index": index, "value": value}
        for (region, index), value in values.items()
    ]


def _decode_groups(values) -> dict[RegionSpeciesKey, Any]:
    return {(item["region"], item["species_index"]): item["value"] for item in values}


def _head_template(name, metadata, target):
    loss = MLPELoss(**metadata["loss"])
    population_ids = tuple(metadata["population_ids"])
    count = len(population_ids)
    return RegionalMLPECalibration(
        region_name=name[0],
        target=target,
        population_ids=population_ids,
        distance_center=0.0,
        distance_scale=max(1.0, 2 * loss.min_distance_scale),
        intercept=0.0,
        slope=0.0,
        unit_variance=loss.variance_floor,
        residual_variance=loss.variance_floor,
        effect_mean=jnp.zeros(count, dtype=jnp.float64),
        effect_covariance=jnp.eye(count, dtype=jnp.float64),
        effect_precision_cholesky=jnp.eye(count, dtype=jnp.float64),
        calibration_pairs=tuple(tuple(pair) for pair in metadata["calibration_pairs"]),
        ml_log_likelihood=0.0,
        loss=loss,
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
        raise ArtifactError("Unsupported calibration module in the saved model")
    return LandscapeGeneticsModel(
        distance_model,
        calibration,
        tuple(metadata["feature_names"]),
        species=None if metadata["species"] is None else tuple(metadata["species"]),
    )


def save_model(path, model: LandscapeGeneticsModel) -> None:
    """Save a fitted model and its prediction contract atomically.

    The Equinox file retains distance parameters, direct or MLPE calibration,
    target semantics, species order, feature names, and MLPE calibration-pair
    identities. Compatible custom distance models use the same template protocol.
    Files contain a small JSON metadata header followed by serialized array values.

    The destination parent directory must exist. Writing and flushing a temporary
    file precedes atomic replacement; a failed write preserves the previous file.
    Save each ensemble member separately. See ``save_checkpoint`` when optimizer
    and random state are needed for continuation.

    Args:
        path (str | Path): Destination model file in an existing directory.
        model: Fitted landscape genetics model to save.

    Raises:
        ArtifactError: The supplied object is not a landscape genetics model.

    Examples:
        >>> save_model("analysis.eqx", state.selected_model)
        >>> restored = load_model("analysis.eqx", distance_model=matching_template)
        >>> prediction = restored.predict(region, pairs=query_pairs)

        The [multispecies workflow](../examples/multispecies.md) constructs a
        matching template for model loading; the
        [ensemble workflow](../examples/ensemble.md) demonstrates checkpoint continuation.
    """
    if not isinstance(model, LandscapeGeneticsModel):
        raise ArtifactError("save_model requires a fitted LandscapeGeneticsModel")
    _write_native(path, model, kind="model", metadata=_model_metadata(model))


def load_model(path, *, distance_model: AbstractDistanceModel) -> LandscapeGeneticsModel:
    """A saved landscape genetics model loaded using a caller-constructed distance model.

    The template supplies the model class, static settings, and compatible
    numerical shapes and dtypes. Match patch size, patch microbatch size, dropout,
    species count, and custom activation functions. Initial parameter values may
    differ. Saved parameters and conductance solver settings replace the template's
    values; solver preparation is rebuilt for each region.

    Equinox checks array structure, shapes, and dtypes; incompatible
    artifacts or templates raise ``ArtifactError``. Static settings supplied by
    the caller remain part of the model contract, so retain the original constructor
    settings alongside the artifact.

    Args:
        path (str | Path): Model file previously written by ``save_model``.
        distance_model: Template matching the saved model's class, static settings,
            array shapes, and dtypes.

    Returns:
        The saved model with its genetic calibration and prediction contract.

    Raises:
        ArtifactError: The artifact or supplied template is incompatible.
        RuntimeError: Float64 is disabled.
    """
    _require_float64()
    try:
        with Path(path).open("rb") as stream:
            metadata, weak_leaves = _read_header(stream, kind="model")
            template = _model_template(metadata, distance_model)
            return _read_native(stream, template, weak_leaves)
    except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            f"Cannot load model with the supplied distance_model template: {error}"
        ) from error
