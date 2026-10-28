"""Tests of landscape genetics models: species queries, original scale and conditioning."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    DirectCalibrationHead,
    fit_calibration_head,
    LandscapeGeneticsModel,
    load_model,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import LogLinearConductance


def test_genetic_queries_use_distance_model_species_and_original_scale(observation_problem):
    region, _, _ = observation_problem()
    distance_model = LogLinearConductance(2, species=("stoat", "hare"), patch_size=1)
    distance_model = eqx.tree_at(
        lambda model: model.intercepts, distance_model, jnp.log(jnp.array([1.99, 0.99]))
    )
    model = LandscapeGeneticsModel(
        distance_model,
        DirectCalibrationHead(TargetSpec("divergence", transform="log1p")),
        region.feature_names,
    )
    assert model.species == ("stoat", "hare")
    region = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=jnp.concatenate((region.grid_positions, region.grid_positions)),
        species=("hare",) * 3 + ("stoat",) * 3,
    )
    results = model.predict(region, {"hare": None, "stoat": [("unit/99", "unit/7")]})
    assert tuple(results) == ("hare", "stoat")
    assert results["hare"].pairs == (
        ("unit/7", "unit/20"),
        ("unit/7", "unit/99"),
        ("unit/20", "unit/99"),
    )
    assert results["stoat"].pairs == (("unit/99", "unit/7"),)
    np.testing.assert_allclose(results["hare"].values, np.expm1([1, 2, 1]), rtol=1e-6)
    np.testing.assert_allclose(results["stoat"].values, np.expm1([1]), rtol=1e-6)
    assert results["stoat"].species == "stoat"
    partial = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids[:3],
        grid_positions=region.grid_positions[:3],
        species=("hare",) * 3,
    )
    np.testing.assert_array_equal(
        model.predict(partial, {"hare": None})["hare"].values, results["hare"].values
    )
    for query, message in (
        (None, "Multispecies models require a query mapping"),
        ([("unit/7", "unit/20")], "Multispecies models require a query mapping"),
        ({"missing": None}, "Unknown species"),
        ({}, "at least one species"),
    ):
        with pytest.raises(ValueError, match=message):
            model.predict(region, query)


def test_single_species_genetic_queries_return_one_prediction(observation_problem):
    region, target, distance_model = observation_problem()
    model = LandscapeGeneticsModel(
        distance_model, DirectCalibrationHead(target), region.feature_names
    )
    every_pair = model.predict(region)
    assert every_pair.pairs == (
        ("unit/7", "unit/20"),
        ("unit/7", "unit/99"),
        ("unit/20", "unit/99"),
    )
    selected = model.predict(region, [("unit/99", "unit/7")])
    assert selected.pairs == (("unit/99", "unit/7"),)
    np.testing.assert_array_equal(selected.values, every_pair.values[1:2])
    with pytest.raises(ValueError, match="species"):
        model.predict(region, {"frog": None})
    named = replace(
        model, distance_model=type(distance_model)(distance_model.weight, species=("frog",))
    )
    by_species = named.predict(region, {"frog": [("unit/99", "unit/7")]})
    assert tuple(by_species) == ("frog",)
    np.testing.assert_array_equal(by_species["frog"].values, selected.values)
    assert named.predict(region, [("unit/99", "unit/7")]).species == "frog"


def test_species_calibration_and_support_preserve_separate_population_effects(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_model
):
    region, observations, training = four_population_calibration
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    names = ("stoat", "hare")
    doubled = ScalarEmbedding(jnp.asarray(2.0))
    references = (
        mlpe_calibrated_model,
        LandscapeGeneticsModel(
            doubled,
            fit_calibration_head(
                region, replace(training, values=training.values * 2.0), doubled, loss=MLPELoss()
            ),
            region.feature_names,
        ),
    )
    model = LandscapeGeneticsModel(
        ScalarEmbedding(jnp.array([1.0, 2.0]), species=names),
        MLPECalibrationHead(
            observations.target,
            {
                (region.name, name): cast(
                    MLPECalibrationHead, reference.calibration_head
                ).for_region(region.name)
                for name, reference in zip(names, references, strict=True)
            },
        ),
        region.feature_names,
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
            {name: pairs for name in reversed(names)},
            known_effects=mode == "known_effects",
            support=tuple(reversed(support)) if mode == "support" else None,
        )
        for index, name in enumerate(names):
            expected = references[index].predict(
                region,
                pairs,
                known_effects=mode == "known_effects",
                support=replace(support[index], species=None) if mode == "support" else None,
            )
            np.testing.assert_allclose(results[name].values, expected.values, atol=1e-10)
            for extra, values in expected.extras.items():
                np.testing.assert_allclose(results[name].extras[extra], values, atol=1e-10)
    with pytest.raises(ValueError):
        model.predict(combined, {"hare": pairs}, support=support[0])
    with pytest.raises(ValueError):
        model.predict(combined, {name: pairs for name in names}, support=(support[0], support[0]))


def test_model_conditions_real_ecological_distances_with_calibration_fixed(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_model, joint_gaussian_reference
):
    region, observations, training = four_population_calibration
    calibration_pairs = training.observed_pairs
    model = mlpe_calibrated_model
    head = cast(MLPECalibrationHead, model.calibration_head).for_region(region.name)
    direct = replace(model, calibration_head=DirectCalibrationHead(observations.target))
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    known = model.predict(region, pairs, known_effects=True)
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
        target=model.target,
    )
    result = model.predict(region, pairs, support=support)
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
    reused = PairwiseObservations.from_pairs(
        [tuple(reversed(calibration_pairs[0]))], [1.0], target=model.target
    )
    with pytest.raises(ValueError):
        model.predict(region, pairs, support=reused)
    with pytest.raises(ValueError):
        model.predict(
            region,
            [(support.observed_pairs[0][1], support.observed_pairs[0][0])],
            support=support,
        )
    with pytest.raises(ValueError):
        model.predict(region, support=support)
    with pytest.raises(ValueError):
        model.predict(region, pairs, known_effects=True, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs, known_effects=True)
    with pytest.raises(ValueError):
        model.predict(region, [("missing-location", "population-0")], known_effects=True)


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


def test_mlpe_calibration_head_retains_target_scale_after_save_and_load(
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
    assert state.selected_model.target is state.selected_model.calibration_head.target
    converted = LandscapeGeneticsModel(
        state.selected_model.distance_model,
        fit_calibration_head(
            region, training, state.selected_model.distance_model, loss=MLPELoss()
        ),
        region.feature_names,
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
    # A fitted (converged) MLPE calibration head round-trips with its optimizer diagnostics.
    assert eqx.tree_equal(restored.calibration_head, converted.calibration_head)
    np.testing.assert_array_equal(restored.predict(region).values, converted.predict(region).values)


def test_mlpe_calibration_heads_match_the_distance_model_species(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_head
):
    region, _, _ = four_population_calibration
    for species in (("frog",), ("frog", "newt")):
        with pytest.raises(ValueError, match="species"):
            LandscapeGeneticsModel(
                ScalarEmbedding(jnp.zeros(len(species)), species=species),
                mlpe_calibrated_head,
                region.feature_names,
            )
    frog = MLPECalibrationHead(
        mlpe_calibrated_head.target,
        {(region.name, "frog"): mlpe_calibrated_head.for_region(region.name)},
    )
    with pytest.raises(ValueError, match="species"):
        LandscapeGeneticsModel(ScalarEmbedding(jnp.asarray(1.0)), frog, region.feature_names)
    with pytest.raises(ValueError, match="species"):
        LandscapeGeneticsModel(
            ScalarEmbedding(jnp.zeros(1), species=("newt",)), frog, region.feature_names
        )
    named = LandscapeGeneticsModel(
        ScalarEmbedding(jnp.ones(2), species=("frog", "newt")), frog, region.feature_names
    )
    assert named.species == ("frog", "newt")


def test_mlpe_prediction_rejects_individuals_before_computing_distances(
    PointConductance, four_population_calibration, mlpe_calibrated_head
):
    region, _, _ = four_population_calibration
    model = LandscapeGeneticsModel(
        PointConductance(jnp.asarray(0.0)), mlpe_calibrated_head, region.feature_names
    )
    # Zero conductance would fail the effective-resistance solve.
    individuals = replace(region, features=region.features * 0.0, individual_based=True)
    with pytest.raises(ValueError, match="individual-based"):
        model.predict(individuals)


def test_support_requires_explicit_query_pairs(four_population_calibration, mlpe_calibrated_model):
    region, observations, _ = four_population_calibration
    support = observations.select_pairs(observations.observed_pairs[-1:])
    with pytest.raises(ValueError, match="Support requires explicit query pairs"):
        mlpe_calibrated_model.predict(region, support=support)


def test_prediction_rejects_stateful_layers(BatchNormConductance, observation_problem):
    region, target, _ = observation_problem()
    model = LandscapeGeneticsModel(
        BatchNormConductance(), DirectCalibrationHead(target), region.feature_names
    )
    with pytest.raises(TypeError, match="stateful layers"):
        model.predict(region)


def test_prediction_requires_float64(observation_problem, float64_required):
    region, target, distance_model = observation_problem()
    model = LandscapeGeneticsModel(
        distance_model, DirectCalibrationHead(target), region.feature_names
    )
    with float64_required():
        model.predict(region)


def test_prediction_checks_the_patch_grid(patch_check_problem):
    problem = patch_check_problem
    model = LandscapeGeneticsModel(
        problem.model, DirectCalibrationHead(problem.target), ("elevation",)
    )
    with problem.trim_required():
        model.predict(problem.misaligned)
    with problem.shared_vertices_warned():
        model.predict(problem.shared)
