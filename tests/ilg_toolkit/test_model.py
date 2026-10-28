"""Scientific behavior of model."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    calibrate,
    DirectCalibrationModule,
    LandscapeGeneticsModel,
    load_model,
    Log1pLoss,
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import LogLinearConductance


def test_genetic_queries_use_saved_species_order_and_original_scale(observation_problem):
    region, _, _ = observation_problem()
    distance_model = LogLinearConductance(2, num_species=2, patch_size=1)
    distance_model = eqx.tree_at(
        lambda model: model.intercepts, distance_model, jnp.log(jnp.array([1.99, 0.99]))
    )
    model = LandscapeGeneticsModel(
        distance_model,
        DirectCalibrationModule(TargetSpec("divergence", transform="log1p")),
        region.feature_names,
        species=("stoat", "hare"),
    )
    region = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=jnp.concatenate((region.grid_positions, region.grid_positions)),
        species=("hare",) * 3 + ("stoat",) * 3,
    )
    results = model.predict(region, species=("hare", "stoat"))
    np.testing.assert_allclose(results["hare"].values, np.expm1([1, 2, 1]), rtol=1e-6)
    np.testing.assert_allclose(results["stoat"].values, np.expm1([0.5, 1, 0.5]), rtol=1e-6)
    np.testing.assert_array_equal(
        model.predict(region, species="hare").values, results["hare"].values
    )
    partial = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids[:3],
        grid_positions=region.grid_positions[:3],
        species=("hare",) * 3,
    )
    np.testing.assert_array_equal(
        model.predict(partial, species="hare").values, results["hare"].values
    )
    with pytest.raises(ValueError):
        model.predict(region)


def test_species_calibration_and_support_preserve_separate_population_effects(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_model
):
    region, observations, training = four_population_calibration
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    names = ("stoat", "hare")
    references = (
        mlpe_calibrated_model,
        calibrate(
            region,
            replace(training, values=training.values * 2.0),
            ScalarEmbedding(jnp.asarray(2.0)),
            loss=MLPELoss(),
        ),
    )
    model = LandscapeGeneticsModel(
        ScalarEmbedding(jnp.array([1.0, 2.0]), num_species=2),
        MLPECalibrationModule(
            observations.target,
            {
                (region.name, index): cast(MLPECalibrationModule, reference.calibration).for_region(
                    region.name
                )
                for index, reference in enumerate(references)
            },
        ),
        region.feature_names,
        species=names,
    )
    combined = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=jnp.concatenate((region.grid_positions, region.grid_positions)),
        species=tuple(name for name in names for _ in region.sampling_unit_ids),
    )
    support = tuple(
        PairwiseObservations.from_pairs(
            [("population-4", "population-0")],
            [2.0 * (index + 1)],
            target=observations.target,
            species=name,
        )
        for index, name in enumerate(names)
    )
    for mode in ("marginal", "known_effects", "support"):
        results = model.predict(
            combined,
            species=tuple(reversed(names)),
            pairs=pairs,
            known_effects=mode == "known_effects",
            support=tuple(reversed(support)) if mode == "support" else None,
        )
        for index, name in enumerate(names):
            expected = references[index].predict(
                region,
                pairs=pairs,
                known_effects=mode == "known_effects",
                support=replace(support[index], species=None) if mode == "support" else None,
            )
            np.testing.assert_allclose(results[name].values, expected.values, atol=1e-10)
            for extra, values in expected.extras.items():
                np.testing.assert_allclose(results[name].extras[extra], values, atol=1e-10)
    with pytest.raises(ValueError):
        model.predict(combined, species="hare", pairs=pairs, support=support[0])
    with pytest.raises(ValueError):
        model.predict(combined, species=names, pairs=pairs, support=(support[0], support[0]))


def test_model_conditions_real_ecological_distances_with_calibration_fixed(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_model, joint_gaussian_reference
):
    region, observations, training = four_population_calibration
    calibration_pairs = training.observed_pairs
    direct = calibrate(region, training, ScalarEmbedding(jnp.asarray(1.0)), loss=Log1pLoss())
    model = mlpe_calibrated_model
    head = cast(MLPECalibrationModule, model.calibration).regions[(region.name, 0)]
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    known = model.predict(region, pairs=pairs, known_effects=True)
    distances = model.distance_model.predict(region).as_matrix()
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}

    def selected_distances(selected):
        return np.array([distances[lookup[a], lookup[b]] for a, b in selected])

    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs,
        selected_distances(calibration_pairs),
        observations.aligned_values(region)[
            [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
        ],
        pairs,
        selected_distances(pairs),
    )
    np.testing.assert_allclose(known.extras["model_values"], expected_mean, atol=1e-10)
    np.testing.assert_allclose(known.extras["model_variance"], expected_variance, atol=1e-10)
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0"), ("population-4", "population-1")],
        [2.0, 2.5],
        target=head.target,
    )
    result = model.predict(region, pairs=pairs, support=support)
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs + support.observed_pairs,
        np.r_[selected_distances(calibration_pairs), selected_distances(support.observed_pairs)],
        np.r_[
            observations.aligned_values(region)[
                [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
            ],
            support.observed_values,
        ],
        pairs,
        selected_distances(pairs),
    )
    np.testing.assert_allclose(result.extras["model_values"], expected_mean, atol=1e-10)
    np.testing.assert_allclose(result.extras["model_variance"], expected_variance, atol=1e-10)
    np.testing.assert_allclose(result.as_matrix()[[4, 5], [2, 3]], expected_mean, atol=1e-10)
    np.testing.assert_allclose(
        result.as_matrix("effect_variance")[[4, 5], [2, 3]],
        expected_variance - head.residual_variance - head.loss.jitter,
        atol=1e-10,
    )
    assert eqx.tree_equal(model.distance_model, direct.distance_model)
    reused = PairwiseObservations.from_pairs(
        [tuple(reversed(calibration_pairs[0]))], [1.0], target=head.target
    )
    with pytest.raises(ValueError):
        model.predict(region, pairs=pairs, support=reused)
    with pytest.raises(ValueError):
        model.predict(
            region,
            pairs=[(support.observed_pairs[0][1], support.observed_pairs[0][0])],
            support=support,
        )
    with pytest.raises(ValueError):
        model.predict(region, support=support)
    with pytest.raises(ValueError):
        model.predict(region, pairs=pairs, known_effects=True, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs=pairs, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs=pairs, known_effects=True)
    with pytest.raises(ValueError):
        model.predict(region, pairs=[("missing-location", "population-0")], known_effects=True)


def test_explicit_transform_returns_original_units_and_separate_ecological_distances(
    ScalarEmbedding, observation_problem
):
    region, _, _ = observation_problem()
    target = TargetSpec("explicitly transformed divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [np.e - 1], target=target
    )
    result = train(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    prediction = result.selected_model.predict(region)
    assert result.history[0].training_loss < 1e-12
    np.testing.assert_allclose(prediction.as_matrix()[0, 1], np.e - 1, rtol=1e-6)
    np.testing.assert_allclose(
        result.selected_model.distance_model.predict(region).as_matrix()[0, 1], 1
    )


def test_recalibrated_model_retains_target_scale_after_save_and_load(
    ScalarEmbedding, four_population_calibration, tmp_path
):
    region, observations, training = four_population_calibration
    target = replace(observations.target, transform="log1p")
    observations = replace(observations, values=jnp.expm1(observations.values), target=target)
    training = replace(training, values=jnp.expm1(training.values), target=target)
    state = train(
        region,
        training,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    assert state.selected_model.target is state.selected_model.calibration.target
    converted = calibrate(
        region,
        training,
        state.selected_model.distance_model,
        loss=MLPELoss(),
    )
    assert eqx.tree_equal(converted.distance_model, state.selected_model.distance_model)
    unseen = replace(region, name="unseen")
    assert np.isfinite(state.selected_model.predict(unseen).values).all()
    with pytest.raises(ValueError):
        converted.predict(unseen)
    path = tmp_path / "converted.ilg"
    save_model(path, converted)
    restored = load_model(path, distance_model=ScalarEmbedding(jnp.asarray(0.0)))
    assert restored.target == observations.target
    np.testing.assert_array_equal(restored.predict(region).values, converted.predict(region).values)


def test_prediction_requires_float64(observation_problem, float64_required):
    region, target, distance_model = observation_problem()
    model = LandscapeGeneticsModel(
        distance_model, DirectCalibrationModule(target), region.feature_names
    )
    with float64_required():
        model.predict(region)
