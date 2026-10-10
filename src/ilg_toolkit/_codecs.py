"""Explicit shipped-model codecs; no serialized classes or pytree definitions."""

from dataclasses import asdict, fields

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._archive import ArtifactError, require_fields
from .config import ResistanceSolverConfig, TrainingConfig
from .data import TargetSpec
from .mlpe import MLPEConfig, MLPEHead
from .model import CalibratedModel
from .models import ResNet9Conductance, UNetEmbeddingDistance


def _arraylike(value):
    return eqx.is_array(value) or isinstance(value, jax.ShapeDtypeStruct)


def encode_tree_leaves(tree, arrays):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    records = []
    for path, value in leaves:
        if _arraylike(value):
            kind, encoded = "array", arrays.add(value)
        elif type(value) in (bool, int, float):
            kind, encoded = type(value).__name__, value
        else:
            raise ArtifactError(f"Unsupported tree leaf at {jax.tree_util.keystr(path)}")
        records.append({"path": jax.tree_util.keystr(path), "kind": kind, "value": encoded})
    return {"leaves": records}


def decode_tree_leaves(template, record, archive):
    require_fields(record, {"leaves"}, "Tree payload")
    expected, definition = jax.tree_util.tree_flatten_with_path(template)
    records = record["leaves"]
    if not isinstance(records, list) or len(records) != len(expected):
        raise ArtifactError("Tree payload leaf count does not match its template")
    restored = []
    for (path, value), leaf in zip(expected, records, strict=True):
        require_fields(leaf, {"path", "kind", "value"}, "Tree leaf")
        if leaf["path"] != jax.tree_util.keystr(path):
            raise ArtifactError("Tree payload leaf paths do not match its template")
        if _arraylike(value):
            if leaf["kind"] != "array":
                raise ArtifactError("Expected a numeric array leaf")
            array = archive.array(leaf["value"], shape=value.shape)
            if array.dtype.kind != np.dtype(value.dtype).kind:
                raise ArtifactError("Tree payload dtype category does not match its template")
            if array.dtype.itemsize > (8 if array.dtype.kind == "c" else 4) and not jax.config.read(
                "jax_enable_x64"
            ):
                raise ArtifactError(
                    "64-bit JAX payload requires JAX_ENABLE_X64=true; no truncation"
                )
            restored.append(jnp.asarray(array))
        else:
            if leaf["kind"] != type(value).__name__ or type(leaf["value"]) is not type(value):
                raise ArtifactError("Tree scalar leaf type does not match its template")
            restored.append(leaf["value"])
    return jax.tree_util.tree_unflatten(definition, restored)


def _template(record):
    require_fields(record, {"kind", "version", "constructor", "leaves"}, "Model")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ArtifactError("Unsupported model codec version")
    constructors = {
        "unet_embedding": (
            UNetEmbeddingDistance,
            {"in_channels", "patch_size", "base_channels", "embedding_dim", "dropout"},
        ),
        "resnet9_conductance": (
            ResNet9Conductance,
            {"in_channels", "patch_size", "min_conductance"},
        ),
    }
    if record["kind"] not in constructors:
        raise ArtifactError(
            "Unsupported model architecture; only shipped UNet and ResNet9 are supported"
        )
    cls, fields = constructors[record["kind"]]
    options = record["constructor"]
    require_fields(options, fields, "Model constructor")
    for name in fields - {"dropout", "min_conductance"}:
        if type(options[name]) is not int or options[name] < 1:
            raise ArtifactError(f"Model constructor {name} must be a positive integer")
    # Shape-only initialization prevents allocation based on unchecked model dimensions.
    return eqx.filter_eval_shape(lambda: cls(**options, key=jax.random.PRNGKey(0)))


def encode_encoder(model, arrays):
    if type(model) is UNetEmbeddingDistance:
        area = model.patch_size**2
        inputs = model.patch_embedding.in_features
        if not isinstance(inputs, int) or inputs % area:
            raise ArtifactError("Model patch inputs do not match its declared patch size")
        kind = "unet_embedding"
        options = {
            "in_channels": inputs // area,
            "patch_size": model.patch_size,
            "base_channels": model.base_channels,
            "embedding_dim": model.embedding_dim,
            "dropout": model.encoder1.dropout.p,
        }
    elif type(model) is ResNet9Conductance:
        kind = "resnet9_conductance"
        options = {
            "in_channels": model.conv1.in_channels,
            "patch_size": model.patch_size,
            "min_conductance": model.min_conductance,
        }
    else:
        raise ArtifactError("Unsupported custom model architecture; use a shipped UNet or ResNet9")
    record = {"kind": kind, "version": 1, "constructor": options, "leaves": None}
    template = _template(record)
    if jax.tree_util.tree_structure(model) != jax.tree_util.tree_structure(template):
        raise ArtifactError("Model has unsupported static architecture changes")
    record["leaves"] = encode_tree_leaves(model, arrays)
    return record


def decode_encoder(record, archive):
    model = decode_tree_leaves(_template(record), record["leaves"], archive)
    if type(model) is UNetEmbeddingDistance:
        for block in (
            model.encoder1,
            model.encoder2,
            model.bottleneck,
            model.decoder2,
            model.decoder1,
        ):
            if not 0 <= block.dropout.p < 1 or type(block.dropout.inference) is not bool:
                raise ArtifactError("Invalid U-Net dropout settings")
    return model


def _target(record):
    require_fields(record, set(TargetSpec.__dataclass_fields__), "Target")
    if any(not isinstance(value, str) or not value for value in record.values()):
        raise ArtifactError("Target fields must be nonempty strings")
    return TargetSpec(**record)


def decode_training_config(record):
    """Validate the same explicit optimization record in checkpoints and runs."""
    require_fields(record, set(TrainingConfig.__dataclass_fields__), "Training config")
    values = dict(record)
    require_fields(
        values["solver"], set(ResistanceSolverConfig.__dataclass_fields__), "Solver config"
    )
    values["solver"] = ResistanceSolverConfig(**values["solver"])
    return TrainingConfig(**values)


def _pairs(value, context):
    if not isinstance(value, (list, tuple)):
        raise ArtifactError(f"{context}: pairs must be a sequence")
    result, seen = [], set()
    for pair in value:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(not isinstance(label, str) or not label for label in pair)
            or pair[0] == pair[1]
        ):
            raise ArtifactError(f"{context}: pairs require distinct nonempty labels")
        canonical = tuple(sorted(pair))
        if canonical in seen:
            raise ArtifactError(f"{context}: duplicate unordered pair identities")
        seen.add(canonical)
        result.append(tuple(pair))
    return tuple(result)


def _access(value):
    if not isinstance(value, dict) or any(not isinstance(name, str) or not name for name in value):
        raise ArtifactError("Pair provenance must map nonempty region names to labelled pairs")
    return {name: _pairs(pairs, "Pair provenance") for name, pairs in value.items()}


_PREPARATION = {
    "features": "prepared_hwc",
    "locations": "native_row_column_grid",
    "normalization": "caller_prepared",
}


def encode_model(model, arrays):
    return {
        "encoder": encode_encoder(model.encoder, arrays),
        "model_state": None,
        "target": asdict(model.target),
        "objective": model.objective,
        "feature_count": model.feature_count,
        "feature_names": model.feature_names,
        "preparation": _PREPARATION,
        "prediction_scale": "original",
        "solver": asdict(model.solver_config),
        "calibrations": {
            name: encode_head(head, arrays) for name, head in model.calibrations.items()
        },
        "training_pairs": model.training_pairs,
        "validation_pairs": model.validation_pairs,
    }


def decode_model(record, archive):
    require_fields(
        record,
        {
            "encoder",
            "model_state",
            "target",
            "objective",
            "feature_count",
            "feature_names",
            "preparation",
            "prediction_scale",
            "solver",
            "calibrations",
            "training_pairs",
            "validation_pairs",
        },
        "CalibratedModel",
    )
    if record["model_state"] is not None or record["preparation"] != _PREPARATION:
        raise ArtifactError("Unsupported model state or prepared-feature contract")
    if record["prediction_scale"] != "original":
        raise ArtifactError("Unsupported prediction target scale")
    count, names = record["feature_count"], record["feature_names"]
    if type(count) is not int or count < 1:
        raise ArtifactError("CalibratedModel feature_count must be a positive integer")
    if names is not None and (
        not isinstance(names, (list, tuple))
        or len(names) != count
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != count
    ):
        raise ArtifactError("CalibratedModel feature names must declare each channel uniquely")
    model = decode_encoder(record["encoder"], archive)
    if record["encoder"]["constructor"]["in_channels"] != count:
        raise ArtifactError("Encoder input channels do not match model feature_count")
    require_fields(record["solver"], {"rtol", "atol", "max_steps", "use_amg"}, "Solver config")
    if not isinstance(record["calibrations"], dict):
        raise ArtifactError("Regional calibrations must be a mapping")
    heads = {name: decode_head(head, archive) for name, head in record["calibrations"].items()}
    training = _access(record["training_pairs"])
    validation = _access(record["validation_pairs"])
    if record["objective"] == "mlpe" and not heads:
        raise ArtifactError("MLPE model is missing required regional calibration")
    return CalibratedModel(
        encoder=model,
        target=_target(record["target"]),
        feature_count=count,
        feature_names=None if names is None else tuple(names),
        solver_config=ResistanceSolverConfig(**record["solver"]),
        objective=record["objective"],
        calibrations=heads,
        training_pairs=training,
        validation_pairs=validation,
    )


_HEAD_FIELDS = set(MLPEHead.__dataclass_fields__)
_POSTERIOR_FIELDS = {"effect_mean", "effect_covariance", "effect_precision_cholesky"}


def encode_head(head, arrays):
    # Preserve immutable metadata directly; asdict would first deep-copy the
    # entire posterior only to replace it with numeric archive references.
    record = {field.name: getattr(head, field.name) for field in fields(head)}
    record["target"], record["config"] = asdict(head.target), asdict(head.config)
    for name in _POSTERIOR_FIELDS:
        record[name] = arrays.add(np.asarray(getattr(head, name), dtype=np.float64))
    return record


def decode_head(record, archive):
    require_fields(record, _HEAD_FIELDS, "MLPE head")
    values = dict(record)
    labels = values["population_ids"]
    if (
        not isinstance(labels, (list, tuple))
        or len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ArtifactError("MLPE population identities must be distinct nonempty labels")
    values["population_ids"] = tuple(labels)
    values["target"] = _target(values["target"])
    if values["target"].kind != "dissimilarity":
        raise ArtifactError("Population MLPE artifacts require a dissimilarity target")
    values["calibration_pairs"] = _pairs(values["calibration_pairs"], "MLPE calibration")
    if not values["calibration_pairs"] or any(
        endpoint not in labels for pair in values["calibration_pairs"] for endpoint in pair
    ):
        raise ArtifactError("MLPE calibration endpoints must belong to its population identities")
    roles = values["calibration_roles"]
    if (
        not isinstance(roles, (list, tuple))
        or len(roles) != len(values["calibration_pairs"])
        or any(role not in {"training", "validation", "calibration"} for role in roles)
    ):
        raise ArtifactError("MLPE calibration roles are missing or invalid")
    values["calibration_roles"] = tuple(roles)
    require_fields(
        values["config"],
        set(MLPEConfig.__dataclass_fields__),
        "MLPE config",
    )
    if any(isinstance(value, bool) for value in values["config"].values()):
        raise ArtifactError("MLPE numerical settings cannot be booleans")
    values["config"] = MLPEConfig(**values["config"])
    if (
        type(values["converged"]) is not bool
        or type(values["optimizer_iterations"]) is not int
        or values["optimizer_iterations"] < 0
        or not isinstance(values["optimizer_message"], str)
    ):
        raise ArtifactError("MLPE optimizer diagnostics have invalid types")
    for name in (
        _HEAD_FIELDS
        - _POSTERIOR_FIELDS
        - {
            "region_name",
            "target",
            "population_ids",
            "calibration_pairs",
            "calibration_roles",
            "config",
            "optimizer_iterations",
            "optimizer_message",
            "converged",
        }
    ):
        if type(values[name]) not in (float, int):
            raise ArtifactError(f"MLPE scalar {name} must be numeric")
    n = len(labels)
    for name in _POSTERIOR_FIELDS:
        shape = (n,) if name == "effect_mean" else (n, n)
        array = archive.array(values[name], shape=shape, dtype=np.float64)
        values[name] = tuple(array) if name == "effect_mean" else tuple(map(tuple, array))
    covariance = np.asarray(values["effect_covariance"])
    factor = np.asarray(values["effect_precision_cholesky"])
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ArtifactError("MLPE posterior covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ArtifactError("MLPE posterior covariance must be positive definite") from error
    if not np.array_equal(factor, np.tril(factor)) or (np.diag(factor) <= 0).any():
        raise ArtifactError("MLPE posterior precision factor must be lower triangular and positive")
    if not isinstance(values["region_name"], str) or not values["region_name"]:
        raise ArtifactError("MLPE region_name must be a nonempty string")
    return MLPEHead(**values)
