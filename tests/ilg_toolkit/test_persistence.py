"""Tests of saving and loading landscape genetics models and checkpoints."""

import json
from dataclasses import replace
from typing import cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    ArtifactError,
    DirectCalibrationHead,
    LandscapeGeneticsModel,
    load_checkpoint,
    load_model,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    save_checkpoint,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import ConductanceModel


# Direct and MLPE calibration heads, with unnamed and named species, save different trees.
@pytest.mark.parametrize(("loss", "species"), [(Log1pLoss(), None), (MLPELoss(), ("frog", "newt"))])
def test_saved_precision_is_preserved(tmp_path, training_problem, loss, species, float64_required):
    with jax.enable_x64():
        region, observations, distance_model = training_problem("embedding", species=species)
        state = train(
            region,
            observations,
            distance_model=distance_model,
            config=TrainingConfig(loss=loss, steps=0),
        )
        for kind in ("model", "checkpoint"):
            path = tmp_path / f"{kind}.eqx"
            if kind == "model":
                save_model(path, state.selected_model)
                model = load_model(path, distance_model=distance_model)
                assert eqx.tree_equal(model, state.selected_model)
            else:
                save_checkpoint(path, state)
                restored = load_checkpoint(
                    path, distance_model=distance_model, optimizer=state.config.optimizer
                )
                assert eqx.tree_equal(restored, state)
                model = restored.selected_model
            query = None if species is None else dict.fromkeys(species)
            expected = state.selected_model.predict(region, query)
            assert eqx.tree_equal(model.predict(region, query), expected)
            for values in jax.tree.leaves(eqx.filter(expected, eqx.is_inexact_array)):
                assert values.dtype == jnp.float64
            with jax.enable_x64(False):
                narrow = jax.tree.map(
                    lambda leaf: leaf.astype(jnp.float32) if eqx.is_inexact_array(leaf) else leaf,
                    distance_model,
                )
            with float64_required():
                if kind == "model":
                    load_model(path, distance_model=narrow)
                else:
                    load_checkpoint(path, distance_model=narrow, optimizer=state.config.optimizer)


def test_species_names_round_trip_and_predictions_are_identical(tmp_path, training_problem):
    names = ("stoat", "hare")
    region, observations, distance_model = training_problem("embedding", species=names)
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=MLPELoss(), steps=1),
    )
    renamed = type(distance_model)(jnp.zeros((2, 2)), species=("frog", "newt"))
    model_path, checkpoint_path = tmp_path / "model.eqx", tmp_path / "checkpoint.eqx"
    save_model(model_path, state.selected_model)
    save_checkpoint(checkpoint_path, state)
    model = load_model(model_path, distance_model=renamed)
    checkpoint = load_checkpoint(
        checkpoint_path, distance_model=renamed, optimizer=state.config.optimizer
    )
    for restored, saved in (
        (model, state.selected_model),
        (checkpoint.selected_model, state.selected_model),
        (checkpoint.latest_model, state.latest_model),
    ):
        assert restored.species == restored.distance_model.species == names
        assert eqx.tree_equal(restored.distance_model, saved.distance_model)
        assert eqx.tree_equal(restored.calibration_head, saved.calibration_head)
        assert set(cast(MLPECalibrationHead, restored.calibration_head).calibrations) == {
            (region.name, name) for name in names
        }
        expected = saved.predict(region, dict.fromkeys(names))
        for name, prediction in restored.predict(region, dict.fromkeys(names)).items():
            assert prediction.species == name
            np.testing.assert_array_equal(prediction.values, expected[name].values)
    unnamed = type(distance_model)(jnp.zeros(2))
    with pytest.raises(ArtifactError, match="species"):
        load_model(model_path, distance_model=unnamed)
    with pytest.raises(ArtifactError, match="species"):
        load_checkpoint(checkpoint_path, distance_model=unnamed, optimizer=state.config.optimizer)


# NumPy storage drops weak types, and eqx.tree_equal ignores them.
def test_saved_weak_types_are_restored_whatever_the_template(
    tmp_path, training_problem, ScalarEmbedding
):
    with jax.enable_x64():
        region, observations, _ = training_problem("embedding")
        # The initial parameter is weakly typed; after one Adam step only its
        # first moment still is.
        initial, trained = (
            train(
                region,
                observations,
                distance_model=ScalarEmbedding(jnp.asarray(1.0)),
                config=TrainingConfig(loss=Log1pLoss(), steps=steps),
            )
            for steps in (0, 1)
        )
        template = ScalarEmbedding(jnp.zeros(()))
        assert not template.weight.weak_type
        save_model(tmp_path / "model.eqx", initial.selected_model)
        save_checkpoint(tmp_path / "checkpoint.eqx", trained)
        model = load_model(tmp_path / "model.eqx", distance_model=template)
        restored = load_checkpoint(
            tmp_path / "checkpoint.eqx", distance_model=template, optimizer=trained.config.optimizer
        )

        def weak_types(tree):
            # Typed random keys have no weak type.
            return [
                leaf.weak_type
                for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))
                if not jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key)
            ]

        assert weak_types(model) == weak_types(initial.selected_model) == [True]
        assert 0 < sum(weak_types(trained)) < len(weak_types(trained))
        assert weak_types(restored) == weak_types(trained)
        # A weakly typed float64 parameter keeps float32 features in float32.
        predictions = initial.selected_model.predict(region).values
        assert predictions.dtype == jnp.float32
        np.testing.assert_array_equal(model.predict(region).values, predictions)
        assert model.predict(region).values.dtype == jnp.float32

        lines = (tmp_path / "model.eqx").read_bytes().split(b"\n", 1)
        for record in (None, [1]):
            header = json.loads(lines[0])
            header.pop("weak_leaves") if record is None else header.update(weak_leaves=record)
            corrupt = tmp_path / "corrupt.eqx"
            corrupt.write_bytes(json.dumps(header).encode() + b"\n" + lines[1])
            with pytest.raises(ArtifactError, match="weak"):
                load_model(corrupt, distance_model=template)

        # Python scalars give float32 weak types only while float64 is disabled.
        with jax.enable_x64(False):
            narrow = eqx.tree_at(
                lambda model: model.distance_model.weight,
                initial.selected_model,
                jnp.asarray(1.0),
            )
            save_model(tmp_path / "narrow.eqx", narrow)
        with pytest.raises(ArtifactError, match="current JAX precision"):
            load_model(
                tmp_path / "narrow.eqx",
                distance_model=ScalarEmbedding(jnp.zeros((), jnp.float32)),
            )


def test_odds_target_model_reloads_and_predicts_in_original_units(
    training_problem, WeightedEmbedding, tmp_path
):
    region, observations, distance_model = training_problem("embedding")
    target = TargetSpec("Fst", units="proportion", transform="odds")
    observations = replace(observations, values=observations.values / 2, target=target)
    state = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=Log1pLoss(), steps=3, optimizer=optax.adam(0.05)),
    )
    assert state.history[-1].training_loss < state.history[0].training_loss
    model = state.selected_model
    path = tmp_path / "odds.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.ones(2)))
    distances = np.asarray(model.distance_model.predict(region).values)
    prediction = restored.predict(region)
    assert restored.target == target
    np.testing.assert_array_equal(prediction.values, model.predict(region).values)
    np.testing.assert_allclose(prediction.values, distances / (1 + distances))
    assert np.all((np.asarray(prediction.values) >= 0) & (np.asarray(prediction.values) < 1))


def test_direct_model_reload_uses_caller_template_and_preserves_predictions(
    training_problem, WeightedEmbedding, tmp_path
):
    region, observations, distance_model = training_problem("embedding")
    observations = replace(observations, target=replace(observations.target, transform="log1p"))
    model = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(steps=0, loss=Log1pLoss()),
    ).selected_model
    path = tmp_path / "direct.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.asarray([0.5, -0.5])))
    np.testing.assert_array_equal(
        restored.distance_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert isinstance(restored.calibration_head, DirectCalibrationHead)
    assert eqx.tree_equal(restored.calibration_head, model.calibration_head)
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names


def test_custom_mlpe_reload_preserves_mlpe_calibrations_and_support_conditioning(
    persistence_problem, WeightedEmbedding, tmp_path
):
    from dataclasses import replace

    region, observations = persistence_problem()
    other = replace(region, name="second-calibration")
    other_observations = PairwiseObservations(
        observations.sampling_unit_ids, observations.values * 2.0, target=observations.target
    )
    labels = set(region.sampling_unit_ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    model = train(
        (region, other),
        (observations.select_pairs(pairs[:-1]), other_observations.select_pairs(pairs[:-1])),
        distance_model=WeightedEmbedding(jnp.asarray([0.2, -0.7])),
        config=TrainingConfig(steps=0, loss=MLPELoss()),
        validation=(region, observations.select_pairs(pairs[-1:])),
    ).selected_model
    path = tmp_path / "custom-mlpe.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=WeightedEmbedding(jnp.asarray([4.0, 3.0])))
    assert type(restored.distance_model) is WeightedEmbedding
    assert eqx.tree_equal(restored.calibration_head, model.calibration_head)
    assert set(cast(MLPECalibrationHead, restored.calibration_head).calibrations) == {
        (region.name, None),
        (other.name, None),
    }
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict(region, query, known_effects=True)
    after = restored.predict(region, query, known_effects=True)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    before = model.predict(region, query, support=support)
    after = restored.predict(region, query, support=support)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    assert (
        cast(MLPECalibrationHead, restored.calibration_head)
        .calibrations[(region.name, None)]
        .converged
        is False
    )


def test_conductance_reload_preserves_surface_distances_and_solver_settings(
    PointConductance, tmp_path
):
    features = np.random.default_rng(1).uniform(0.5, 1.5, size=(8, 8, 2))
    region = RegionBatch(
        "graph-reload",
        cx.field(
            jnp.asarray(features),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation", "canopy"])),
        ),
        ("west", "north", "east"),
        np.array([[0, 0], [0, 7], [7, 7]]),
    )
    solver_config = ResistanceSolverConfig(rtol=1e-9, atol=1e-10)
    assert solver_config != ResistanceSolverConfig()
    model = LandscapeGeneticsModel(
        PointConductance(jnp.asarray(0.3), solver_config=solver_config),
        DirectCalibrationHead(TargetSpec("divergence")),
        region.feature_names,
    )
    path = tmp_path / "conductance.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=PointConductance(jnp.asarray(-1.0)))
    restored_model = cast(ConductanceModel, restored.distance_model)
    np.testing.assert_array_equal(
        restored_model.conductance_surface(region).data,
        cast(ConductanceModel, model.distance_model).conductance_surface(region).data,
    )
    # Repeated conjugate gradient solves on a GPU differ by rounding.
    np.testing.assert_allclose(
        restored_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        restored.predict(region).values, model.predict(region).values, rtol=1e-9
    )
    assert restored_model.solver_config == solver_config


def _saved_conductance_model(PointConductance, training_problem, solver_config, path):
    """Saves a conductance model with a direct calibration head, and returns it and its region."""
    region, _, _ = training_problem("conductance")
    model = LandscapeGeneticsModel(
        PointConductance(jnp.asarray(0.3), solver_config=solver_config),
        DirectCalibrationHead(TargetSpec("divergence")),
        region.feature_names,
    )
    save_model(path, model)
    return model, region


def test_saved_models_record_each_solver_choice(
    solver_choice, PointConductance, training_problem, tmp_path
):
    solver_config = ResistanceSolverConfig(solver=solver_choice)
    path = tmp_path / "conductance.eqx"
    model, region = _saved_conductance_model(
        PointConductance, training_problem, solver_config, path
    )
    restored = load_model(path, distance_model=PointConductance(jnp.asarray(-1.0)))
    assert cast(ConductanceModel, restored.distance_model).solver_config == solver_config
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)


@pytest.mark.parametrize("use_amg", [False, True])
def test_loading_a_saved_use_amg_setting_names_its_replacement(
    PointConductance, training_problem, tmp_path, use_amg
):
    path = tmp_path / "conductance.eqx"
    _saved_conductance_model(PointConductance, training_problem, ResistanceSolverConfig(), path)
    line, arrays = path.read_bytes().split(b"\n", 1)
    header = json.loads(line)
    header["metadata"]["distance_solver"] = dict(
        rtol=1e-6, atol=1e-6, max_steps=1000, use_amg=use_amg
    )
    path.write_bytes(json.dumps(header).encode() + b"\n" + arrays)
    with pytest.raises(ArtifactError, match=r"use_amg.*solver='cg'.*solver='amg'"):
        load_model(path, distance_model=PointConductance(jnp.asarray(-1.0)))


def test_model_reload_uses_native_shape_and_dtype_compatibility(WeightedEmbedding, tmp_path):
    with jax.enable_x64():
        distance_model = WeightedEmbedding(jnp.asarray([0.2, -0.4], dtype=jnp.float32))
        model = LandscapeGeneticsModel(
            distance_model,
            DirectCalibrationHead(TargetSpec("divergence")),
            ("elevation", "canopy"),
        )
        path = tmp_path / "model.eqx"
        save_model(path, model)
        assert eqx.tree_equal(load_model(path, distance_model=distance_model), model)
        with pytest.raises(ArtifactError):
            load_model(path, distance_model=WeightedEmbedding(jnp.zeros(3, dtype=jnp.float32)))
        with pytest.raises(ArtifactError):
            load_model(path, distance_model=WeightedEmbedding(jnp.zeros(2, dtype=jnp.float64)))

        truncated = tmp_path / "truncated.eqx"
        truncated.write_bytes(path.read_bytes()[:-10])
        with pytest.raises(ArtifactError):
            load_model(truncated, distance_model=distance_model)


def test_interrupted_model_replacement_preserves_previous_predictions(
    persistence_problem, WeightedEmbedding, tmp_path, monkeypatch
):
    import os

    region, _ = persistence_problem()

    def model(log_weights):
        return LandscapeGeneticsModel(
            WeightedEmbedding(jnp.asarray(log_weights)),
            DirectCalibrationHead(TargetSpec("divergence")),
            region.feature_names,
        )

    first, replacement = model([0.2, -0.7]), model([1.0, 0.5])
    path = tmp_path / "model.eqx"
    save_model(path, first)

    def interrupt(source, destination):
        raise OSError("interrupted replacement")

    monkeypatch.setattr(os, "replace", interrupt)
    with pytest.raises(OSError):
        save_model(path, replacement)
    restored = load_model(path, distance_model=replacement.distance_model)
    np.testing.assert_array_equal(restored.predict(region).values, first.predict(region).values)
    assert not np.array_equal(replacement.predict(region).values, first.predict(region).values)
    assert set(tmp_path.iterdir()) == {path}


def test_loading_requires_float64(tmp_path, observation_problem, float64_required):
    region, target, distance_model = observation_problem()
    path = tmp_path / "model.eqx"
    save_model(
        path,
        LandscapeGeneticsModel(distance_model, DirectCalibrationHead(target), region.feature_names),
    )
    with float64_required():
        load_model(path, distance_model=distance_model)
