"""Inference artifacts preserve public predictions using a caller encoder template."""

from typing import final

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_checkpoint import initial_state as initial_state
from test_species_conditioning import conditioned_species as conditioned_species

from ilg_toolkit import (
    DirectCalibrationModule,
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
    load_model,
    save_model,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def problem(name="inference"):
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    labels = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        cx.Field(jnp.asarray(features), dims=("row", "column", "feature")),
        {"fox": SamplingUnits(labels, np.array(list(np.ndindex(2, 3))))},
        feature_names=("east_covariate", "canopy_covariate"),
    )
    distances = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    observed = 2 + 0.7 * distances + effects[:, None] + effects[None, :]
    noise = np.random.default_rng(17).normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(
        labels,
        observed,
        target=TargetSpec("synthetic divergence", units="index", transform="log1p"),
    )
    return region, observations


def test_direct_model_reload_uses_caller_encoder_and_preserves_predictions(tmp_path):
    region, observations = problem()

    def encoder(seed):
        return UNetEmbeddingDistance(
            2,
            species=("fox",),
            patch_size=1,
            base_channels=2,
            embedding_dim=3,
            dropout=0.23,
            key=jax.random.key(seed),
        )

    model = fit(
        [region],
        {region.name: {"fox": observations}},
        distance_model=encoder(3),
        config=TrainingConfig(steps=0, objective="direct_log1p"),
    ).model
    path = tmp_path / "direct.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=encoder(7))
    np.testing.assert_array_equal(
        restored.ecological_distances(region)["fox"], model.ecological_distances(region)["fox"]
    )
    np.testing.assert_array_equal(
        restored.predict(region)["fox"].values, model.predict(region)["fox"].values
    )
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names
    assert restored.training_pairs == model.training_pairs
    assert restored.training_objective == "direct_log1p"
    assert isinstance(restored.distance_model, UNetEmbeddingDistance)
    assert restored.distance_model.backbone.encoder1.dropout.p == 0.23


def test_custom_mlpe_reload_preserves_posteriors_and_conditioning(tmp_path, conditioned_species):
    import equinox as eqx

    model, region, _, query, support, partitions = conditioned_species
    path = tmp_path / "custom-mlpe.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=model.distance_model)
    assert eqx.tree_equal(restored, model)
    names = ("fox", "lynx")
    pairs = dict.fromkeys(names, query)
    for name in names:
        np.testing.assert_array_equal(
            restored.predict(region, species=names)[name].values,
            model.predict(region, species=names)[name].values,
        )
    for operation in (
        lambda value: value.predict_known_effects(region, species=names, pairs=pairs),
        lambda value: value.predict_with_support(
            region, support, species=names, pairs=pairs, support_partition=partitions
        ),
    ):
        assert eqx.tree_equal(operation(restored), operation(model))


@pytest.mark.parametrize(
    "saved_dtype,template_dtype",
    [("float64", "float32"), ("complex128", "complex64"), ("int64", "int32"), ("uint64", "uint32")],
)
def test_model_reload_refuses_jax_precision_truncation(tmp_path, saved_dtype, template_dtype):
    import equinox as eqx
    import pytest

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel
    from ilg_toolkit.models import EmbeddingDistanceModel

    @final
    class CustomEmbedding(EmbeddingDistanceModel):
        weight: jax.Array
        species: tuple[str, ...] = eqx.field(static=True, default=("fox",))
        patch_size: int = eqx.field(static=True, default=1)

        def __init__(self, weight: jax.Array):
            self.weight = weight

        def _embedding_grid(
            self, features, species_index, *, inference=True, key=None, patch_batch_size=None
        ):
            return features * self.weight

    path = tmp_path / "precision.ilg"
    with jax.enable_x64():
        encoder = CustomEmbedding(jnp.asarray([1.1234567890123456], dtype=saved_dtype))
        model = LandscapeGeneticsModel(
            encoder,
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            training_objective="direct_log1p",
        )
        save_model(path, model)
        restored = load_model(path, distance_model=CustomEmbedding(jnp.zeros(1, dtype=saved_dtype)))
        assert isinstance(restored.distance_model, CustomEmbedding)
        np.testing.assert_array_equal(restored.distance_model.weight, encoder.weight)
    with jax.enable_x64(False):
        template = CustomEmbedding(jnp.zeros(1, dtype=template_dtype))
        with pytest.raises(ArtifactError, match="JAX_ENABLE_X64"):
            load_model(path, distance_model=template)


def test_conductance_reload_preserves_surface_distances_and_solver_settings(tmp_path):
    from ilg_toolkit import LandscapeGeneticsModel, ResistanceSolverConfig
    from ilg_toolkit.models import ResNet9Conductance

    region = RegionBatch(
        "graph-reload",
        cx.Field(
            jnp.asarray(np.random.default_rng(1).normal(size=(8, 8, 2))),
            dims=("row", "column", "feature"),
        ),
        {"fox": SamplingUnits(("west", "north", "east"), np.array([[0, 0], [0, 7], [7, 7]]))},
        feature_names=("elevation", "canopy"),
    )

    def encoder(seed):
        return ResNet9Conductance(
            2,
            species=("fox",),
            patch_size=4,
            min_conductance=0.02,
            patch_batch_size=3,
            solver_config=ResistanceSolverConfig(rtol=1e-9, atol=1e-10),
            key=jax.random.key(seed),
        )

    model = LandscapeGeneticsModel(
        encoder(3),
        DirectCalibrationModule(TargetSpec("divergence")),
        feature_count=2,
        feature_names=region.feature_names,
        training_objective="direct_log1p",
    )
    path = tmp_path / "conductance.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=encoder(9))
    np.testing.assert_array_equal(
        restored.conductance_surface(region)["fox"], model.conductance_surface(region)["fox"]
    )
    np.testing.assert_array_equal(
        restored.ecological_distances(region)["fox"], model.ecological_distances(region)["fox"]
    )
    np.testing.assert_array_equal(
        restored.predict(region)["fox"].values, model.predict(region)["fox"].values
    )
    assert isinstance(restored.distance_model, ResNet9Conductance)
    assert isinstance(model.distance_model, ResNet9Conductance)
    assert restored.distance_model.solver_config == model.distance_model.solver_config
    assert restored.distance_model.patch_batch_size == 3
    assert restored.distance_model.min_conductance == 0.02


def test_model_reload_uses_native_shape_and_dtype_compatibility(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNetEmbeddingDistance(
        2, species=("fox",), patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
    )
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
        training_objective="direct_log1p",
    )
    path = tmp_path / "model.ilg"
    save_model(path, model)
    wrong_shape = UNetEmbeddingDistance(
        2, species=("fox",), patch_size=1, base_channels=2, embedding_dim=3, key=jax.random.key(0)
    )
    with pytest.raises(ArtifactError, match="shape|embedding_dim"):
        load_model(path, distance_model=wrong_shape)
    wrong_dtype = jax.tree.map(
        lambda value: value.astype(jnp.float64) if isinstance(value, jax.Array) else value,
        encoder,
    )
    with pytest.raises(ArtifactError, match="dtype"):
        load_model(path, distance_model=wrong_dtype)


def test_model_reload_rejects_truncated_leaves(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    encoder = UNetEmbeddingDistance(
        2, species=("fox",), patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(0)
    )
    model = LandscapeGeneticsModel(
        encoder,
        DirectCalibrationModule(TargetSpec("divergence")),
        2,
        training_objective="direct_log1p",
    )
    path = tmp_path / "model.ilg"
    save_model(path, model)
    truncated = tmp_path / "truncated.ilg"
    truncated.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ArtifactError, match="payload|supplied distance_model template"):
        load_model(truncated, distance_model=encoder)


def test_interrupted_model_replacement_preserves_previous_predictions(tmp_path, monkeypatch):
    import os

    from ilg_toolkit import LandscapeGeneticsModel

    region, _ = problem()

    def model(seed):
        return LandscapeGeneticsModel(
            UNetEmbeddingDistance(
                2,
                species=("fox",),
                patch_size=1,
                base_channels=2,
                embedding_dim=2,
                key=jax.random.key(seed),
            ),
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            feature_names=region.feature_names,
            training_objective="direct_log1p",
        )

    first, replacement = model(3), model(7)
    path = tmp_path / "model.ilg"
    save_model(path, first)

    def interrupt(source, destination):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError, match="interrupted replacement"):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(
        restored.predict(region)["fox"].values, first.predict(region)["fox"].values
    )
    assert set(tmp_path.iterdir()) == {path}


def test_template_rejects_same_shape_species_reassociation(tmp_path):
    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    def distance(species):
        return UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            key=jax.random.key(3),
        )

    model = LandscapeGeneticsModel(
        distance(("fox", "lynx")), DirectCalibrationModule(TargetSpec("divergence")), 2
    )
    path = tmp_path / "registry.ilg"
    save_model(path, model)
    for species in (("lynx", "fox"), ("fox", "deer")):
        with pytest.raises(ArtifactError, match="species"):
            load_model(path, distance_model=distance(species))


def test_bound_callable_policy_settings_must_match(tmp_path):
    from dataclasses import dataclass

    import equinox as eqx
    from artifact_helpers import NativeEmbedding

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    @dataclass(frozen=True)
    class Policy:
        factor: float

        def apply(self, values):
            return values * self.factor

    def template(factor):
        return eqx.tree_at(lambda value: value.activation, NativeEmbedding(), Policy(factor).apply)

    model = LandscapeGeneticsModel(template(2.0), DirectCalibrationModule(TargetSpec("d")), 2)
    path = tmp_path / "policy.ilg"
    save_model(path, model)
    load_model(path, distance_model=template(2.0))
    with pytest.raises(ArtifactError, match="factor"):
        load_model(path, distance_model=template(3.0))


@pytest.mark.parametrize("family", ["custom", "unet"])
def test_native_templates_are_stable_in_separate_processes(tmp_path, family):
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = """
import sys
# Only the test helper directory is added; toolkit resolution stays with the caller.
sys.path.insert(0, sys.argv.pop(1))
import numpy as np
from artifact_helpers import distance_model, checkpoint_problem
from ilg_toolkit import (
    LandscapeGeneticsModel, DirectCalibrationModule, TargetSpec, save_model, load_model
)
family, mode, directory = sys.argv[1:]
region, _ = checkpoint_problem()
path = directory + '/model.ilg'
if mode == 'save':
    model = LandscapeGeneticsModel(
        distance_model(family), DirectCalibrationModule(TargetSpec('d')), 2, region.feature_names
    )
    save_model(path, model)
    np.save(directory + '/prediction.npy', model.predict(region, species='fox')['fox'].values)
else:
    model = load_model(path, distance_model=distance_model(family, seed=99))
    np.testing.assert_array_equal(
        model.predict(region, species='fox')['fox'].values, np.load(directory + '/prediction.npy')
    )
"""
    for mode, hash_seed in (("save", "11"), ("load", "947")):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(Path(__file__).resolve().parent),
                family,
                mode,
                str(tmp_path),
            ],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "family,setting,value",
    [
        ("custom", "scale", 2.0),
        ("unet", "dropout", 0.4),
        ("resnet", "patch_batch_size", 1),
        ("resnet", "min_conductance", 0.2),
        ("loglinear", "max_conductance", 200.0),
        ("loglinear", "patch_size", 2),
    ],
)
def test_same_shape_execution_settings_must_match(tmp_path, family, setting, value):
    from copy import copy

    from artifact_helpers import distance_model

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    original = distance_model(family)
    altered = copy(original)
    object.__setattr__(altered, setting, value)
    model = LandscapeGeneticsModel(original, DirectCalibrationModule(TargetSpec("d")), 2)
    path = tmp_path / "setting.ilg"
    save_model(path, model)
    with pytest.raises(ArtifactError, match=setting):
        load_model(path, distance_model=altered)


def test_nested_dropout_and_solver_policy_must_match(tmp_path):
    from copy import copy
    from dataclasses import replace

    import equinox as eqx
    from artifact_helpers import distance_model

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    unet = distance_model("unet")
    changed = eqx.tree_at(lambda model: model.backbone.encoder1.dropout.p, unet, 0.3)
    resnet = distance_model("resnet")
    from ilg_toolkit.models import ResNet9Conductance

    assert isinstance(resnet, ResNet9Conductance)
    changed_solver = copy(resnet)
    object.__setattr__(changed_solver, "solver_config", replace(resnet.solver_config, rtol=1e-8))
    for model, template, match in (
        (unet, changed, "dropout"),
        (resnet, changed_solver, "solver_config"),
    ):
        path = tmp_path / "nested.ilg"
        save_model(path, LandscapeGeneticsModel(model, DirectCalibrationModule(TargetSpec("d")), 2))
        with pytest.raises(ArtifactError, match=match):
            load_model(path, distance_model=template)


def test_artifacts_reject_altered_or_appended_payload_and_metadata(tmp_path):
    import json

    from artifact_helpers import NativeEmbedding

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    model = LandscapeGeneticsModel(NativeEmbedding(), DirectCalibrationModule(TargetSpec("d")), 2)
    path = tmp_path / "integrity.ilg"
    save_model(path, model)
    original = path.read_bytes()
    header_bytes, leaves = original.split(b"\n", 1)
    metadata = json.loads(header_bytes)
    metadata["metadata"]["target"]["name"] = "different target"
    altered = json.dumps(metadata).encode() + b"\n" + leaves
    for contents in (
        original + b"unexpected bytes",
        original[:-1],
        original[:-1] + bytes([original[-1] ^ 1]),
        altered,
    ):
        path.write_bytes(contents)
        with pytest.raises(ArtifactError):
            load_model(path, distance_model=model.distance_model)


@pytest.mark.parametrize(
    "setting,value",
    [("feature_count", 0), ("feature_count", True), ("feature_names", ("only-one",))],
)
def test_invalid_model_contract_cannot_be_saved(tmp_path, setting, value):
    from copy import copy

    from artifact_helpers import NativeEmbedding

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    model = LandscapeGeneticsModel(NativeEmbedding(), DirectCalibrationModule(TargetSpec("d")), 2)
    invalid = copy(model)
    object.__setattr__(invalid, setting, value)
    with pytest.raises(ArtifactError, match="feature"):
        save_model(tmp_path / "invalid.ilg", invalid)


@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
@pytest.mark.parametrize("kind", ["model", "checkpoint"])
def test_atomic_generations_survive_all_write_failures(
    tmp_path, monkeypatch, initial_state, failure, kind
):
    import os

    import equinox as eqx

    from ilg_toolkit import load_checkpoint, save_checkpoint

    state, _, _ = initial_state
    tree = state.model if kind == "model" else state
    save = save_model if kind == "model" else save_checkpoint
    load = load_model if kind == "model" else load_checkpoint
    path = tmp_path / "atomic.ilg"
    save(path, tree)
    original = path.read_bytes()

    def interrupt(*args, **kwargs):
        if failure == "serialize":
            args[0].write(b"partial generation")
        raise OSError("interrupted artifact write")

    owner, name = (eqx, "tree_serialise_leaves") if failure == "serialize" else (os, failure)
    with monkeypatch.context() as patch:
        patch.setattr(owner, name, interrupt)
        with pytest.raises(OSError, match="interrupted artifact"):
            save(path, tree)
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}
    assert eqx.tree_equal(load(path, distance_model=state.distance_model), tree)


@pytest.mark.parametrize("header", [b"", b"not json\n", b"[]\n", b'{"format":"unknown"}\n'])
def test_malformed_artifact_headers_fail_actionably(tmp_path, header):
    from artifact_helpers import NativeEmbedding

    from ilg_toolkit import ArtifactError, load_checkpoint

    path = tmp_path / "invalid.ilg"
    path.write_bytes(header)
    for load in (load_model, load_checkpoint):
        with pytest.raises(ArtifactError):
            load(path, distance_model=NativeEmbedding())


def test_unsupported_opaque_model_configuration_is_actionable(tmp_path):
    import equinox as eqx
    from artifact_helpers import NativeEmbedding

    from ilg_toolkit import ArtifactError, LandscapeGeneticsModel

    class OpaquePolicy:
        def apply(self, values):
            return values

    distance = eqx.tree_at(lambda model: model.activation, NativeEmbedding(), OpaquePolicy().apply)
    model = LandscapeGeneticsModel(distance, DirectCalibrationModule(TargetSpec("d")), 2)
    with pytest.raises(ArtifactError, match="configuration"):
        save_model(tmp_path / "opaque.ilg", model)
