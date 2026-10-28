"""Public calibration freezes distances and owns fitted target behavior."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    calibrate,
    Log1pLoss,
    MLPECalibrationModule,
    MLPEError,
    MLPELoss,
    PairwiseObservations,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.mlpe import calibrate_mlpe


def test_calibration_freezes_distance_model_and_matches_distance_vector_fit(
    ScalarEmbedding, four_population_calibration, mlpe_calibrated_model
):
    region, _, observations = four_population_calibration
    distance_model = ScalarEmbedding(jnp.asarray(1.0))
    model = mlpe_calibrated_model
    reference = calibrate_mlpe(
        distance_model.predict(region, pairs=observations.observed_pairs).values,
        observations,
        region_name=region.name,
        loss=MLPELoss(),
    )
    head = cast(MLPECalibrationModule, model.calibration).for_region(region.name)
    assert eqx.tree_equal(model.distance_model, distance_model)
    assert eqx.tree_equal(head, reference)
    assert head.converged and head.optimizer_iterations > 0
    initialized = train(
        region,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(loss=MLPELoss(), steps=0),
    ).selected_model
    initial_head = cast(MLPECalibrationModule, initialized.calibration).for_region(region.name)
    assert head.ml_log_likelihood > initial_head.ml_log_likelihood + 1e-5
    assert not np.allclose(
        [head.unit_variance, head.residual_variance],
        [initial_head.unit_variance, initial_head.residual_variance],
    )
    assert model.target == observations.target
    assert set(cast(MLPECalibrationModule, model.calibration).regions) == {(region.name, 0)}
    assert "model_values" in model.predict(region).extras
    with pytest.raises(ValueError):
        MLPECalibrationModule(model.target, {("wrong-region", 0): head})
    with pytest.raises(ValueError):
        MLPECalibrationModule(TargetSpec("another target"), {(region.name, 0): head})
    with pytest.raises(MLPEError):
        replace(head, target=TargetSpec("relatedness", kind="relatedness"))


def test_direct_calibration_preserves_target_transform_and_rejects_iteration_budget(
    observation_problem,
):
    region, _, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [2.0], target=TargetSpec("divergence", transform="log1p")
    )
    model = calibrate(region, observations, distance_model, loss=Log1pLoss())
    np.testing.assert_allclose(
        model.predict(region).values, np.expm1(distance_model.predict(region).values)
    )
    assert eqx.tree_equal(model.distance_model, distance_model)
    with pytest.raises(ValueError, match="iteration"):
        calibrate(region, observations, distance_model, loss=Log1pLoss(), max_iterations=5)


def test_calibration_reads_species_from_the_distance_model(training_problem):
    region, observations, distance_model = training_problem("embedding", num_species=2)
    model = calibrate(region, observations, distance_model, loss=MLPELoss())
    assert model.species == distance_model.species == ("species-0", "species-1")
    assert set(cast(MLPECalibrationModule, model.calibration).regions) == {
        (region.name, 0),
        (region.name, 1),
    }
    assert eqx.tree_equal(model.distance_model, distance_model)
    predictions = model.predict(region, species=("species-0", "species-1"))
    assert [prediction.species for prediction in predictions.values()] == [
        "species-0",
        "species-1",
    ]
    unnamed = type(distance_model)(distance_model.log_weights)
    with pytest.raises(ValueError, match="distance model with species"):
        calibrate(region, observations, unnamed, loss=MLPELoss())


def test_single_species_calibration_names_its_species_once(observation_problem):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    unnamed = calibrate(region, observations, distance_model, loss=Log1pLoss())
    assert unnamed.species is None
    assert unnamed.predict(region).species is None
    frog_model = type(distance_model)(distance_model.weight, species=("frog",))
    region = replace(region, species=("frog",) * len(region.sampling_unit_ids))
    model = calibrate(region, observations, frog_model, loss=Log1pLoss())
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
            calibrate(region, invalid_observations, invalid_model, loss=Log1pLoss())


def test_multi_region_calibration_requires_shared_named_features(observation_problem):
    region, target, distance_model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [2.0], target=target)
    other = replace(region, name="other")
    with pytest.raises(ValueError, match="feature"):
        calibrate((region, other), (observations, observations), distance_model, loss=Log1pLoss())
    named = replace(region, feature_names=("elevation", "canopy"))
    other_named = replace(named, name="other")
    fitted = calibrate(
        (named, other_named), (observations, observations), distance_model, loss=Log1pLoss()
    )
    assert fitted.feature_names == ("elevation", "canopy")
    with pytest.raises(ValueError, match="feature"):
        calibrate(
            (named, replace(other_named, feature_names=("canopy", "elevation"))),
            (observations, observations),
            distance_model,
            loss=Log1pLoss(),
        )


def test_calibration_fitting_requires_float64(observation_problem, float64_required):
    region, target, model = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [1.0, 2.0], target=target
    )
    with float64_required():
        calibrate(region, observations, model, loss=Log1pLoss())
