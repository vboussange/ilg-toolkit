"""Scientific behavior of model."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    MLPECalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    TargetSpec,
    TrainingConfig,
    fit,
)


def test_model_conditions_real_ecological_distances_without_updating_encoder(
    ScalarEmbedding, calibration_problem, joint_gaussian_reference
):
    import itertools

    import jax.numpy as jnp

    from ilg_toolkit import (
        PairwiseObservations,
        TrainingConfig,
        calibrate,
        fit,
    )

    region, observations, _, _ = calibration_problem("conditioned-landscape")
    calibration_pairs = tuple(itertools.combinations(region.sampling_unit_ids[:4], 2))
    training = ObservationPartition(region.name, calibration_pairs, role="training")
    direct = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(weight=jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    ).model
    model = calibrate(direct, region, observations)
    head = cast(MLPECalibrationModule, model.calibration).regions[region.name]
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    known = model.predict_known_effects(region, pairs)
    assert known.prediction.sampling_unit_ids == region.sampling_unit_ids
    assert known.prediction.pairs == pairs
    distances = model.ecological_distances(region)
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
    np.testing.assert_allclose(known.model_values, expected_mean, atol=1e-10)
    np.testing.assert_allclose(known.model_variance, expected_variance, atol=1e-10)
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0"), ("population-4", "population-1")],
        [2.0, 2.5],
        target=head.target,
    )
    result = model.predict_with_support(
        region,
        pairs,
        support,
        support_partition=ObservationPartition(region.name, support.observed_pairs, role="support"),
    )
    assert result.prediction.sampling_unit_ids == region.sampling_unit_ids
    assert result.prediction.pairs == pairs
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
    np.testing.assert_allclose(result.model_values, expected_mean, atol=1e-10)
    np.testing.assert_allclose(result.model_variance, expected_variance, atol=1e-10)
    assert eqx.tree_equal(model.distance_model, direct.distance_model)
    assert result.provenance.support_pairs == tuple(
        tuple(sorted(pair)) for pair in support.observed_pairs
    )
    with pytest.raises(ValueError):
        direct.predict_known_effects(region, pairs)
    with pytest.raises(ValueError):
        model.predict_known_effects(region, [("missing-location", "population-0")])


def test_explicit_transform_returns_original_units_and_separate_ecological_distances(
    ScalarEmbedding, observation_problem
):
    region, _, _ = observation_problem()
    target = TargetSpec("explicitly transformed divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [np.e - 1], target=target
    )
    result = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    prediction = result.model.predict(region)
    assert result.history[0].training_loss < 1e-12
    assert prediction.scale == "original"
    assert prediction.target.transform == "log1p"
    assert prediction.target.units == "index"
    np.testing.assert_allclose(prediction.as_matrix()[0, 1], np.e - 1, rtol=1e-6)
    np.testing.assert_allclose(result.model.ecological_distances(region)[0, 1], 1)


def test_composed_calibration_owns_inference_and_preserves_training_objective(
    ScalarEmbedding, calibration_problem, tmp_path
):
    import pytest

    from ilg_toolkit import (
        calibrate,
        load_model,
        save_model,
    )

    region, observations, training, _ = calibration_problem()
    state = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
    )
    assert state.model.target is state.model.calibration.target
    converted = calibrate(state.model, region, observations)
    assert converted.training_objective == state.config.objective == "direct_log1p"
    assert eqx.tree_equal(converted.distance_model, state.model.distance_model)
    unseen = replace(region, name="unseen")
    assert np.isfinite(state.model.predict(unseen).values).all()
    with pytest.raises(ValueError):
        converted.predict(unseen)
    path = tmp_path / "converted.ilg"
    save_model(path, converted)
    restored = load_model(path, distance_model=ScalarEmbedding(jnp.asarray(0.0)))
    assert restored.training_objective == "direct_log1p"
    assert restored.target == observations.target
    np.testing.assert_array_equal(restored.predict(region).values, converted.predict(region).values)
