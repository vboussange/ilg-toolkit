"""Calibration heads and their fitting for frozen distance models."""

from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    DirectCalibrationHead,
    fit_calibration_head,
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.mlpe import MLPECalibration


def test_fitted_mlpe_calibration_matches_the_dense_reference_profile(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_head, dense_profile
):
    region, _, observations = four_population_calibration
    head = mlpe_calibrated_head
    assert isinstance(head, MLPECalibrationHead)
    assert head.target == observations.target
    assert set(head.calibrations) == {(region.name, None)}
    calibration = head.for_region(region.name)
    assert isinstance(calibration, MLPECalibration)
    assert calibration.population_ids == region.sampling_unit_ids[:4]
    assert calibration.calibration_pairs == observations.observed_pairs
    assert calibration.converged and calibration.optimizer_iterations > 0

    distances = np.asarray(
        ScalarEmbedding(jnp.asarray(1.0)).predict(region, observations.observed_pairs).values
    )
    lookup = {label: index for index, label in enumerate(calibration.population_ids)}
    left = np.array([lookup[a] for a, _ in observations.observed_pairs])
    right = np.array([lookup[b] for _, b in observations.observed_pairs])
    targets = np.asarray(observations.observed_values)

    def dense(variances):
        return dense_profile(distances, targets, left, right, 4, variances)

    variances = np.array([calibration.unit_variance, calibration.residual_variance])
    nll, beta, effect_mean, effect_covariance = dense(variances)
    assert calibration.ml_log_likelihood == pytest.approx(-nll, abs=1e-10)
    np.testing.assert_allclose([calibration.intercept, calibration.slope], beta, atol=1e-10)
    np.testing.assert_allclose(calibration.effect_mean, effect_mean, atol=1e-10)
    np.testing.assert_allclose(calibration.effect_covariance, effect_covariance, atol=1e-10)
    assert calibration.distance_center == pytest.approx(distances.mean(), abs=1e-14)
    assert calibration.distance_scale == pytest.approx(distances.std(ddof=1), abs=1e-14)
    # The fitted variance components maximize the dense profile likelihood.
    for factors in ((1.05, 1.0), (0.95, 1.0), (1.0, 1.05), (1.0, 0.95)):
        assert dense(variances * np.asarray(factors))[0] > nll - 1e-9

    initialized = train(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(loss=MLPELoss(), steps=0),
    ).selected_model.calibration_head
    assert isinstance(initialized, MLPECalibrationHead)
    initial = initialized.for_region(region.name)
    assert calibration.ml_log_likelihood > initial.ml_log_likelihood + 1e-5


def test_calibration_fitting_leaves_the_distance_model_unchanged(
    DropoutConductance, four_population_calibration
):
    region, _, observations = four_population_calibration
    distance_model = DropoutConductance()
    assert not distance_model.dropout.inference
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    assert isinstance(head, MLPECalibrationHead)
    assert eqx.tree_equal(distance_model, DropoutConductance())
    # Inference mode makes the fit deterministic without a random key.
    again = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    assert eqx.tree_equal(head, again)
    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    calibration = head.for_region(region.name)
    distances = distance_model.predict(region, observations.observed_pairs).values
    np.testing.assert_allclose(
        model.predict(region, observations.observed_pairs).extras["model_values"],
        calibration.intercept
        + calibration.slope
        * (distances - calibration.distance_center)
        / calibration.distance_scale,
        rtol=1e-12,
    )


def test_mlpe_calibration_head_looks_up_groups_by_region_and_species(
    training_problem, ScalarEmbedding, four_population_calibration, mlpe_calibrated_head
):
    region, observations, distance_model = training_problem("embedding", num_species=2)
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    assert isinstance(head, MLPECalibrationHead)
    assert set(head.calibrations) == {(region.name, "species-0"), (region.name, "species-1")}
    first = head.for_region(region.name, "species-0")
    second = head.for_region(region.name, "species-1")
    assert first is head.calibrations[(region.name, "species-0")]
    assert second is head.calibrations[(region.name, "species-1")]
    assert second.intercept != pytest.approx(first.intercept)
    with pytest.raises(ValueError, match="species"):
        head.for_region(region.name)
    with pytest.raises(ValueError, match="fit_calibration_head"):
        head.for_region(region.name, "species-2")
    with pytest.raises(ValueError, match="fit_calibration_head"):
        head.for_region("elsewhere", "species-0")

    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    distances = distance_model.predict(region, {"species-1": None})["species-1"].values
    np.testing.assert_allclose(
        model.predict(region, {"species-1": None})["species-1"].extras["model_values"],
        second.intercept
        + second.slope * (distances - second.distance_center) / second.distance_scale,
        rtol=1e-12,
    )

    # A single-species head needs no species name, whether or not the species is named.
    region, _, observations = four_population_calibration
    frog = fit_calibration_head(
        region, observations, ScalarEmbedding(jnp.asarray(1.0), species=("frog",)), loss=MLPELoss()
    )
    assert isinstance(frog, MLPECalibrationHead)
    assert set(frog.calibrations) == {(region.name, "frog")}
    assert frog.for_region(region.name) is frog.for_region(region.name, "frog")
    assert eqx.tree_equal(
        frog.for_region(region.name), mlpe_calibrated_head.for_region(region.name)
    )
    with pytest.raises(ValueError, match="fit_calibration_head"):
        mlpe_calibrated_head.for_region(region.name, "frog")


def test_mlpe_calibration_heads_validate_their_target_and_groups(
    four_population_calibration, mlpe_calibrated_head
):
    region, _, _ = four_population_calibration
    calibration = mlpe_calibrated_head.for_region(region.name)
    target = mlpe_calibrated_head.target
    assert eqx.tree_equal(
        MLPECalibrationHead(target, {(region.name, None): calibration}), mlpe_calibrated_head
    )
    assert MLPECalibrationHead(target, {}).calibrations == {}
    with pytest.raises(ValueError, match="dissimilarity"):
        MLPECalibrationHead(TargetSpec("relatedness", kind="relatedness"), {})
    for groups in (
        {(region.name, None): "not a calibration"},
        {("", None): calibration},
        {(region.name, 0): calibration},
        {region.name: calibration},
    ):
        with pytest.raises(ValueError):
            MLPECalibrationHead(target, groups)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="TargetSpec"):
        DirectCalibrationHead("divergence")  # type: ignore[arg-type]


def test_direct_calibration_head_predicts_in_unsampled_regions(
    observation_problem, four_population_calibration, mlpe_calibrated_model
):
    region, _, distance_model = observation_problem()
    target = TargetSpec("divergence", transform="log1p")
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    head = fit_calibration_head(region, observations, distance_model, loss=Log1pLoss())
    assert head == DirectCalibrationHead(target)
    unsampled = replace(
        region,
        name="unsampled",
        features=cx.field(
            region.feature_array[:, ::-1] * 1.5,
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(region.feature_names)),
        ),
    )
    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    np.testing.assert_allclose(
        model.predict(unsampled).values, np.expm1(distance_model.predict(unsampled).values)
    )
    with pytest.raises(ValueError, match="population-effect"):
        model.predict(unsampled, known_effects=True)
    with pytest.raises(ValueError, match="iteration"):
        fit_calibration_head(
            region, observations, distance_model, loss=Log1pLoss(), max_iterations=5
        )
    # An MLPE calibration head only predicts in the regions it was fitted on.
    fitted_region, _, _ = four_population_calibration
    with pytest.raises(ValueError, match="fit_calibration_head"):
        mlpe_calibrated_model.predict(replace(fitted_region, name="unsampled"))


def test_calibration_reads_species_from_the_distance_model(training_problem):
    region, observations, distance_model = training_problem("embedding", num_species=2)
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    predictions = model.predict(region, {"species-0": None, "species-1": None})
    assert [prediction.species for prediction in predictions.values()] == [
        "species-0",
        "species-1",
    ]
    unnamed = type(distance_model)(distance_model.log_weights)
    with pytest.raises(ValueError, match="distance model with species"):
        fit_calibration_head(region, observations, unnamed, loss=MLPELoss())


def test_single_species_calibration_names_its_species_once(observation_problem):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    head = fit_calibration_head(region, observations, distance_model, loss=Log1pLoss())
    unnamed = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    assert unnamed.species is None
    assert unnamed.predict(region).species is None
    frog_model = type(distance_model)(distance_model.weight, species=("frog",))
    region = replace(region, species=("frog",) * len(region.sampling_unit_ids))
    model = LandscapeGeneticsModel(
        frog_model,
        fit_calibration_head(region, observations, frog_model, loss=Log1pLoss()),
        region.feature_names,
    )
    assert model.species == ("frog",)
    assert model.predict(region).species == "frog"
    np.testing.assert_array_equal(
        model.predict(region).values, unnamed.predict(replace(region, species=None)).values
    )
    for invalid_model, invalid_observations in (
        (frog_model, replace(observations, species="newt")),
        (distance_model, replace(observations, species="frog")),
    ):
        with pytest.raises(ValueError, match="species"):
            fit_calibration_head(region, invalid_observations, invalid_model, loss=Log1pLoss())


def test_multi_region_calibration_requires_shared_named_features(observation_problem):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    other = replace(region, name="other")
    head = fit_calibration_head(
        (region, other), (observations, observations), distance_model, loss=Log1pLoss()
    )
    assert head == DirectCalibrationHead(target)
    swapped = cx.LabeledAxis("feature", np.array(["canopy", "elevation"]))
    with pytest.raises(ValueError, match="feature"):
        fit_calibration_head(
            (
                region,
                replace(other, features=cx.field(region.feature_array, "row", "column", swapped)),
            ),
            (observations, observations),
            distance_model,
            loss=Log1pLoss(),
        )


def test_calibration_fitting_rejects_invalid_inputs(observation_problem):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    with pytest.raises(TypeError, match="AbstractDistanceModel"):
        fit_calibration_head(region, observations, "model", loss=Log1pLoss())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AbstractLoss"):
        fit_calibration_head(region, observations, distance_model, loss="mlpe")  # type: ignore[arg-type]


def test_calibration_fitting_rejects_stateful_layers(BatchNormConductance, observation_problem):
    region, target, _ = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [1.0, 2.0], target=target
    )
    with pytest.raises(TypeError, match="stateful layers"):
        fit_calibration_head(region, observations, BatchNormConductance(), loss=Log1pLoss())


def test_calibration_fitting_requires_float64(observation_problem, float64_required):
    region, target, model = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [1.0, 2.0], target=target
    )
    with float64_required():
        fit_calibration_head(region, observations, model, loss=Log1pLoss())


def test_mlpe_calibration_head_rejects_individual_based_regions(
    four_population_calibration, mlpe_calibrated_model, ScalarEmbedding
):
    region, _, observations = four_population_calibration
    individuals = replace(region, individual_based=True)
    with pytest.raises(ValueError, match="individual-based"):
        mlpe_calibrated_model.predict(individuals)
    with pytest.raises(ValueError, match="individual-based"):
        fit_calibration_head(
            individuals, observations, ScalarEmbedding(jnp.asarray(1.0)), loss=MLPELoss()
        )


def test_calibration_checks_the_patch_grid(patch_check_problem):
    problem = patch_check_problem
    misaligned, shared = problem.misaligned, problem.shared
    with problem.trim_required():
        fit_calibration_head(
            misaligned, problem.observations(misaligned), problem.model, loss=Log1pLoss()
        )
    with problem.shared_vertices_warned():
        fit_calibration_head(shared, problem.observations(shared), problem.model, loss=Log1pLoss())
