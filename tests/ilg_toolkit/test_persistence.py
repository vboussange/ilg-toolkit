"""Scientific behavior of persistence."""

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
    DirectCalibrationModule,
    LandscapeGeneticsModel,
    load_checkpoint,
    load_model,
    Log1pLoss,
    MLPECalibrationModule,
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


# Direct and MLPE calibrations, without and with species, save different trees.
@pytest.mark.parametrize(("loss", "num_species"), [(Log1pLoss(), 1), (MLPELoss(), 2)])
def test_saved_precision_is_preserved(
    tmp_path, training_problem, loss, num_species, float64_required
):
    with jax.enable_x64():
        region, observations, distance_model = training_problem(
            "embedding", num_species=num_species
        )
        species = distance_model.species
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
            for name in species or (None,):
                predictions = state.selected_model.predict(region, species=name).values
                np.testing.assert_array_equal(
                    model.predict(region, species=name).values, predictions
                )
                assert predictions.dtype == jnp.float64
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
    region, observations, distance_model = training_problem("embedding", num_species=2)
    names = ("species-0", "species-1")
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
    for restored in (model, checkpoint.selected_model, checkpoint.latest_model):
        assert restored.species == restored.distance_model.species == names
        assert eqx.tree_equal(restored.distance_model, state.selected_model.distance_model)
        for name, prediction in restored.predict(region, species=names).items():
            assert prediction.species == name
            np.testing.assert_array_equal(
                prediction.values, state.selected_model.predict(region, species=name).values
            )
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
    assert restored.target == model.target
    assert restored.feature_names == model.feature_names


def test_custom_mlpe_reload_preserves_regional_posteriors_and_conditioning(
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
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
        assert eqx.tree_equal(
            cast(MLPECalibrationModule, restored.calibration).regions[(prepared.name, 0)],
            cast(MLPECalibrationModule, model.calibration).regions[(prepared.name, 0)],
        )
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict(region, pairs=query, known_effects=True)
    after = restored.predict(region, pairs=query, known_effects=True)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    before = model.predict(region, pairs=query, support=support)
    after = restored.predict(region, pairs=query, support=support)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(after.extras["model_variance"], before.extras["model_variance"])
    assert (
        cast(MLPECalibrationModule, restored.calibration).regions[(region.name, 0)].converged
        is False
    )


def test_conductance_reload_preserves_surface_distances_and_solver_settings(
    PointConductance, tmp_path
):
    features = np.random.default_rng(1).uniform(0.5, 1.5, size=(8, 8, 2))
    region = RegionBatch(
        "graph-reload",
        cx.Field(jnp.asarray(features), dims=("row", "column", "feature")),
        ("west", "north", "east"),
        np.array([[0, 0], [0, 7], [7, 7]]),
        feature_names=("elevation", "canopy"),
    )
    solver_config = ResistanceSolverConfig(rtol=1e-9, atol=1e-10)
    assert solver_config != ResistanceSolverConfig()
    model = LandscapeGeneticsModel(
        PointConductance(jnp.asarray(0.3), solver_config=solver_config),
        DirectCalibrationModule(TargetSpec("divergence")),
        feature_count=2,
        feature_names=region.feature_names,
    )
    path = tmp_path / "conductance.eqx"
    save_model(path, model)
    restored = load_model(path, distance_model=PointConductance(jnp.asarray(-1.0)))
    restored_model = cast(ConductanceModel, restored.distance_model)
    np.testing.assert_array_equal(
        restored_model.conductance_surface(region),
        cast(ConductanceModel, model.distance_model).conductance_surface(region),
    )
    np.testing.assert_array_equal(
        restored_model.predict(region).as_matrix(),
        model.distance_model.predict(region).as_matrix(),
    )
    np.testing.assert_array_equal(restored.predict(region).values, model.predict(region).values)
    assert restored_model.solver_config == solver_config


def test_model_reload_uses_native_shape_and_dtype_compatibility(WeightedEmbedding, tmp_path):
    with jax.enable_x64():
        distance_model = WeightedEmbedding(jnp.asarray([0.2, -0.4], dtype=jnp.float32))
        model = LandscapeGeneticsModel(
            distance_model, DirectCalibrationModule(TargetSpec("divergence")), 2
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
            DirectCalibrationModule(TargetSpec("divergence")),
            2,
            feature_names=region.feature_names,
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
    save_model(path, LandscapeGeneticsModel(distance_model, DirectCalibrationModule(target), 2))
    with float64_required():
        load_model(path, distance_model=distance_model)
