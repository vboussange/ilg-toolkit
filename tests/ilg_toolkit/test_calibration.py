"""Tests of calibration heads and their fitting for frozen distance models."""

import itertools
from dataclasses import replace
from types import MappingProxyType

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
    ResistanceSolverConfig,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.mlpe import MLPECalibration, MLPEError


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


@pytest.mark.parametrize("variation", ["residual-free", "effect-free"])
def test_mlpe_calibrations_at_extreme_variance_ratios_match_the_dense_profile(
    mlpe_reference, dense_profile, variation
):
    rng = np.random.default_rng(12)
    ids = tuple(f"p{index}" for index in range(6))
    # Three absent pairs make the design incomplete; pairs are reversed and shuffled.
    indices = [pair[::-1] for pair in itertools.combinations(range(6), 2)]
    indices = [indices[index] for index in rng.permutation(len(indices))[3:]]
    distances = rng.uniform(0.5, 3.0, size=len(indices))
    effects = rng.normal(scale=0.7, size=6)
    left, right = (np.array(ends) for ends in zip(*indices, strict=True))
    if variation == "residual-free":
        targets = 2 + 0.4 * distances + effects[left] + effects[right]
    else:
        targets = 2 + 0.4 * distances + rng.normal(scale=0.3, size=len(indices))
    pairs = [(ids[a], ids[b]) for a, b in indices]
    observations = PairwiseObservations.from_pairs(
        pairs, targets, target=TargetSpec("divergence", units="index"), sampling_unit_ids=ids
    )
    by_pair = {frozenset(pair): value for pair, value in zip(pairs, distances, strict=True)}
    region, distance_model = mlpe_reference.problem(
        observations.observed_pairs,
        [by_pair[frozenset(pair)] for pair in observations.observed_pairs],
    )
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    assert isinstance(head, MLPECalibrationHead)
    calibration = head.for_region(region.name)
    ratio = calibration.unit_variance / calibration.residual_variance
    assert ratio > 1e8 if variation == "residual-free" else ratio < 1e-8

    lookup = {label: index for index, label in enumerate(calibration.population_ids)}
    nll, beta, effect_mean, effect_covariance = dense_profile(
        np.asarray(distance_model.predict(region, observations.observed_pairs).values),
        np.asarray(observations.observed_values),
        [lookup[a] for a, _ in observations.observed_pairs],
        [lookup[b] for _, b in observations.observed_pairs],
        6,
        (calibration.unit_variance, calibration.residual_variance),
    )
    assert -calibration.ml_log_likelihood == pytest.approx(nll, rel=2e-7, abs=1e-8)
    np.testing.assert_allclose([calibration.intercept, calibration.slope], beta, rtol=2e-7)
    np.testing.assert_allclose(calibration.effect_mean, effect_mean, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(calibration.effect_covariance, effect_covariance, atol=1e-10)


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
    region, observations, distance_model = training_problem("embedding", species=("frog", "newt"))
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    assert isinstance(head, MLPECalibrationHead)
    assert set(head.calibrations) == {(region.name, "frog"), (region.name, "newt")}
    first = head.for_region(region.name, "frog")
    second = head.for_region(region.name, "newt")
    assert first is head.calibrations[(region.name, "frog")]
    assert second is head.calibrations[(region.name, "newt")]
    assert second.intercept != pytest.approx(first.intercept)
    with pytest.raises(ValueError, match="species"):
        head.for_region(region.name)
    with pytest.raises(ValueError, match="fit_calibration_head"):
        head.for_region(region.name, "toad")
    with pytest.raises(ValueError, match="fit_calibration_head"):
        head.for_region("elsewhere", "frog")

    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    distances = distance_model.predict(region, {"newt": None})["newt"].values
    np.testing.assert_allclose(
        model.predict(region, {"newt": None})["newt"].extras["model_values"],
        second.intercept
        + second.slope * (distances - second.distance_center) / second.distance_scale,
        rtol=1e-12,
    )

    # A single-species MLPE calibration head needs no species name, named or unnamed.
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
    groups: dict[tuple[str, str | None], MLPECalibration] = {(region.name, None): calibration}
    head = MLPECalibrationHead(target, MappingProxyType(groups))
    groups[("other", None)] = calibration
    assert set(head.calibrations) == {(region.name, None)}
    with pytest.raises(ValueError, match="species names or all None"):
        MLPECalibrationHead(
            target, {(region.name, None): calibration, ("other", "frog"): calibration}
        )
    for invalid in (
        {(region.name, None): "not a calibration"},
        {("", None): calibration},
        {(region.name, 0): calibration},
        {region.name: calibration},
        [((region.name, None), calibration)],
    ):
        with pytest.raises(ValueError):
            MLPECalibrationHead(target, invalid)  # type: ignore[arg-type]
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
    region, observations, distance_model = training_problem("embedding", species=("frog", "newt"))
    head = fit_calibration_head(region, observations, distance_model, loss=MLPELoss())
    model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
    predictions = model.predict(region, {"frog": None, "newt": None})
    assert [prediction.species for prediction in predictions.values()] == [
        "frog",
        "newt",
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


def test_direct_calibration_heads_need_no_ecological_distances(
    observation_problem, PointConductance
):
    region, target, _ = observation_problem()
    # Zero conductance would fail any effective-resistance solve.
    region = replace(
        region,
        features=cx.field(
            np.zeros(region.feature_array.shape),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(region.feature_names)),
        ),
    )
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    distance_model = PointConductance(jnp.asarray(0.0))
    head = fit_calibration_head(region, observations, distance_model, loss=Log1pLoss())
    assert head == DirectCalibrationHead(target)
    with pytest.raises(ValueError, match="iteration budget"):
        fit_calibration_head(
            region, observations, distance_model, loss=Log1pLoss(), max_iterations=5
        )
    with pytest.raises(MLPEError, match="max_iterations"):
        fit_calibration_head(
            region, observations, distance_model, loss=MLPELoss(), max_iterations=0
        )


def test_calibration_head_predictions_require_float64(
    four_population_calibration, mlpe_calibrated_head, float64_required
):
    region, _, _ = four_population_calibration
    pairs = [region.sampling_unit_ids[:2]]
    for head in (mlpe_calibrated_head, DirectCalibrationHead(mlpe_calibrated_head.target)):
        with float64_required():
            head.predict(jnp.ones(1), pairs, region_name=region.name)


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


def test_mlpe_calibration_groups_share_a_solver_state(
    cudss_double, training_problem, PointConductance
):
    species = ("frog", "newt")
    regions, observations = zip(
        *(
            training_problem("conductance", name=name, species=species)[:2]
            for name in ("alpine", "valley")
        ),
        strict=True,
    )
    scales = jnp.array([0.0, 0.5])
    direct, iterative = (
        PointConductance(scales, species=species, solver_config=config)
        for config in (
            ResistanceSolverConfig(solver="cudss"),
            ResistanceSolverConfig(rtol=1e-11, atol=1e-11),
        )
    )
    rebuilds, analyses = cudss_double.rebuild_count(), cudss_double.analysis_count()
    head = fit_calibration_head(regions, observations, direct, loss=MLPELoss())
    assert cudss_double.rebuild_count() == rebuilds
    assert cudss_double.analysis_count() == analyses + 1
    expected = fit_calibration_head(regions, observations, iterative, loss=MLPELoss())
    assert isinstance(head, MLPECalibrationHead) and isinstance(expected, MLPECalibrationHead)
    assert head.calibrations.keys() == expected.calibrations.keys()
    for group, calibration in head.calibrations.items():
        np.testing.assert_allclose(calibration.slope, expected.calibrations[group].slope, rtol=1e-6)
