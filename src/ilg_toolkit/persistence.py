"""Saved landscape genetics models: Equinox files with a JSON metadata header."""

import json
import os
import tempfile
from copy import copy
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from ._checks import _require_float64
from .calibration import DirectCalibrationHead, MLPECalibrationHead
from .config import ResistanceSolverConfig
from .data import TargetSpec
from .losses import MLPELoss
from .mlpe import MLPECalibration
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel, ConductanceModel
from .models.base import _species_names, _species_rows


class ArtifactError(ValueError):
    """Error raised for an incompatible saved model, checkpoint or loading template."""


def _is_typed_key(leaf) -> bool:
    """Returns whether ``leaf`` is a typed JAX random key."""
    return isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)


def _key_data(tree):
    """Returns the tree with typed random keys replaced by their raw key data."""
    return jax.tree.map(
        lambda leaf: jax.random.key_data(leaf) if _is_typed_key(leaf) else leaf, tree
    )


def _write_native(path, tree, *, kind, metadata):
    """Writes the metadata header and the tree's arrays to ``path`` atomically.

    A failed write keeps the previous file.
    """
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
    """Reads saved leaves into a template, restoring the weak types recorded when saving."""
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
    """Returns ``leaf`` as a weakly typed array of the same dtype.

    Python scalars are the only public way to create weak types, so the values are
    written into weakly typed zeros, whose dtype follows the current precision.
    """
    zeros = jax.lax.full(leaf.shape, leaf.dtype.type(0).item())
    if not zeros.weak_type or zeros.dtype != leaf.dtype:
        raise ArtifactError(
            f"Cannot restore a weakly typed {leaf.dtype} leaf under the current JAX precision"
        )
    return zeros.at[...].set(leaf)


def _read_header(stream, *, kind):
    """Returns the metadata and weakly typed leaf indices after checking the file kind."""
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
    """Returns the target, feature, species and calibration-head metadata of a model."""
    head = model.calibration_head
    if isinstance(head, DirectCalibrationHead):
        calibration_head = {"kind": "direct"}
    elif isinstance(head, MLPECalibrationHead):
        calibration_head = {
            "kind": "mlpe",
            "groups": [
                {
                    "region": region,
                    "species": species,
                    "population_ids": calibration.population_ids,
                    "calibration_pairs": calibration.calibration_pairs,
                    "loss": asdict(calibration.loss),
                    "optimizer_message": calibration.optimizer_message,
                }
                for (region, species), calibration in head.calibrations.items()
            ],
        }
    else:
        raise ArtifactError("Saving a model requires a direct or MLPE calibration head")
    return {
        "target": asdict(model.target),
        "feature_names": model.feature_names,
        "distance_solver": (
            asdict(model.distance_model.solver_config)
            if isinstance(model.distance_model, ConductanceModel)
            else None
        ),
        "species": model.species,
        "calibration_head": calibration_head,
    }


def _calibration_template(metadata) -> MLPECalibration:
    """Returns an MLPE calibration template with the saved identities and settings."""
    loss = MLPELoss(**metadata["loss"])
    population_ids = tuple(metadata["population_ids"])
    count = len(population_ids)
    return MLPECalibration(
        population_ids=population_ids,
        distance_center=0.0,
        distance_scale=max(1.0, 2 * loss.min_distance_scale),
        intercept=0.0,
        slope=0.0,
        unit_variance=loss.variance_floor,
        residual_variance=loss.variance_floor,
        effect_mean=jnp.zeros(count, dtype=jnp.float64),
        effect_covariance=jnp.eye(count, dtype=jnp.float64),
        calibration_pairs=tuple(tuple(pair) for pair in metadata["calibration_pairs"]),
        ml_log_likelihood=0.0,
        loss=loss,
        optimizer_iterations=0,
        optimizer_message=metadata["optimizer_message"],
        converged=False,
    )


def _model_template[DistanceModel: AbstractDistanceModel](
    metadata, distance_model: DistanceModel
) -> LandscapeGeneticsModel[DistanceModel]:
    """Returns a landscape genetics model template from saved metadata and a distance model."""
    target = TargetSpec(**metadata["target"])
    species = None if metadata["species"] is None else tuple(metadata["species"])
    if _species_rows(distance_model.species) != _species_rows(species):
        raise ArtifactError(
            f"Saved model has species {species!r}; the distance_model template has "
            f"species {distance_model.species!r}"
        )
    if distance_model.species != species:
        distance_model = copy(distance_model)
        object.__setattr__(distance_model, "species", _species_names(species))
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
    saved_head = metadata["calibration_head"]
    if saved_head["kind"] == "direct":
        head = DirectCalibrationHead(target)
    elif saved_head["kind"] == "mlpe":
        head = MLPECalibrationHead(
            target,
            {
                (group["region"], group["species"]): _calibration_template(group)
                for group in saved_head["groups"]
            },
        )
    else:
        raise ArtifactError("Unsupported calibration head in the saved model")
    return LandscapeGeneticsModel(
        distance_model,
        head,
        tuple(metadata["feature_names"]),
    )


def save_model(path: str | os.PathLike[str], model: LandscapeGeneticsModel) -> None:
    """Saves a landscape genetics model to an Equinox file atomically.

    The file holds a JSON metadata header followed by the model's arrays: distance
    model parameters, the direct or maximum-likelihood population effects (MLPE)
    calibration head with its target contract, species, feature names, and the
    region, species and calibration pairs of each MLPE calibration. The write goes
    to a temporary file that atomically replaces the destination, so a failed write
    keeps the previous file. Custom distance models are saved the same way. Each
    ensemble member is saved separately;
    [`save_checkpoint`][ilg_toolkit.save_checkpoint] also saves the optimizer and
    random state for continuing training.

    Args:
        path: Destination file in an existing directory.
        model: Landscape genetics model to save.

    Raises:
        ArtifactError: If ``model`` is not a landscape genetics model or holds an
            unsupported calibration head.

    Examples:
        >>> save_model("analysis.eqx", state.selected_model)
        >>> restored = load_model("analysis.eqx", distance_model=template)
        >>> prediction = restored.predict(region, query_pairs)

        The [multispecies example](../examples/multispecies.md) builds the
        distance-model template for loading.
    """
    if not isinstance(model, LandscapeGeneticsModel):
        raise ArtifactError("save_model requires a LandscapeGeneticsModel")
    _write_native(path, model, kind="model", metadata=_model_metadata(model))


def load_model[DistanceModel: AbstractDistanceModel](
    path: str | os.PathLike[str], *, distance_model: DistanceModel
) -> LandscapeGeneticsModel[DistanceModel]:
    """Loads a saved landscape genetics model into a distance-model template.

    The template supplies the distance model's class, static settings, array shapes
    and dtypes: build it with the original constructor settings (patch size, patch
    microbatch size, dropout, species and activation functions). Its
    parameter values may differ. The saved parameters, species names and solver
    settings replace the template's.

    Args:
        path: File written by [`save_model`][ilg_toolkit.save_model].
        distance_model: Distance-model template matching the saved model's class,
            static settings, array shapes and dtypes.

    Returns:
        The saved landscape genetics model.

    Raises:
        ArtifactError: If the file or the template is incompatible.
        RuntimeError: If float64 is disabled.
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
