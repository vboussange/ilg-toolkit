"""Scientific behavior of data."""

from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import PairwiseObservations, TargetSpec, TrainingConfig, fit


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_observation_dtype_and_absent_pairs_survive_matrix_pair_and_region_alignment(
    observation_problem, dtype
):
    region, target, _ = observation_problem()
    measurements = jnp.asarray([0.123456789012345], dtype=dtype)
    paired = PairwiseObservations.from_pairs([("unit/20", "unit/7")], measurements, target=target)
    matrix = PairwiseObservations.from_matrix(
        ("unit/20", "unit/7"),
        jnp.asarray([[0, measurements[0]], [measurements[0], 0]], dtype=dtype),
        target=target,
    )
    for observations in (paired, matrix):
        assert observations.values.dtype == observations.observed_values.dtype == dtype
        aligned = observations.aligned_values(region)
        assert aligned.dtype == dtype
        assert float(aligned[0, 1]) == float(measurements[0])
        assert bool(jnp.isnan(aligned[0, 2]))
        np.testing.assert_array_equal(jnp.diag(aligned), 0)


def test_region_requires_named_features_and_normalizes_numpy_data_without_narrowing(
    observation_problem,
):
    from dataclasses import replace

    region, _, _ = observation_problem()
    with pytest.raises(TypeError):
        replace(region, features=cast(cx.Field, region.feature_array))
    features = np.array([[[0.123456789012345, 2.0], [1.0, 3.0], [2.0, 4.0]]])
    field = cx.field(features, "row", "column", "feature")
    normalized = replace(region, features=field, grid_positions=[[0, 0], [0, 1], [0, 2]])
    assert normalized.feature_array.dtype == jnp.float64
    np.testing.assert_array_equal(normalized.feature_array, features)
    np.testing.assert_array_equal(normalized.pixel_nodes, [0, 1, 2])
    with pytest.raises(ValueError):
        replace(region, grid_positions=np.array([[0, 0], [0, 1.5], [0, 2]]))
    with pytest.raises(ValueError):
        replace(region, grid_positions=[[0, 0], [0, 1], [0, 3]])


def test_labelled_raster_axes_normalize_without_changing_feature_meaning(observation_problem):
    """A named-axis permutation describes the same prepared raster and locations."""
    from dataclasses import replace

    region, target, model = observation_problem()
    names = ("elevation", "canopy")
    plain = replace(region, feature_names=names)
    field = cx.field(
        plain.feature_array,
        cx.LabeledAxis("row", np.array([100])),
        cx.LabeledAxis("column", np.array([10, 20, 30])),
        cx.LabeledAxis("feature", np.array(names)),
    ).order_as("feature", "column", "row")
    labelled = replace(plain, features=field)
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [0.4, 0.6], target=target
    )
    expected = fit(
        plain,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=1),
    ).model
    actual = fit(
        labelled,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=1),
    ).model
    assert labelled.features.dims == ("row", "column", "feature")
    row_axis = labelled.features.axes["row"]
    column_axis = labelled.features.axes["column"]
    np.testing.assert_array_equal(cast(cx.LabeledAxis, row_axis).ticks, [100])
    np.testing.assert_array_equal(cast(cx.LabeledAxis, column_axis).ticks, [10, 20, 30])
    assert labelled.feature_names == names
    np.testing.assert_array_equal(actual.predict(labelled).values, expected.predict(plain).values)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_array_like_observations_and_targets_follow_configured_precision(
    observation_problem, enable_x64
):
    precise = 0.123456789012345
    with jax.enable_x64(enable_x64):
        region, target, model = observation_problem()
        dtype = jnp.float64 if enable_x64 else jnp.float32
        expected = precise if enable_x64 else float(np.float32(precise))
        predictions = []
        for values in ([precise], np.array([precise]), jnp.asarray([precise])):
            observations = PairwiseObservations.from_pairs(
                [("unit/7", "unit/20")], values, target=target
            )
            assert observations.values.dtype == dtype
            assert float(observations.observed_values[0]) == expected
            for transformed in (target.forward(values), target.inverse(values)):
                assert transformed.dtype == dtype
                assert float(transformed[0]) == expected
            fitted = fit(
                region,
                observations,
                distance_model=model,
                config=TrainingConfig(objective="direct_log1p", epochs=0),
            )
            prediction = fitted.model.predict(region)
            assert prediction.values.dtype == prediction.as_matrix().dtype == dtype
            predictions.append(prediction.values)
        np.testing.assert_array_equal(predictions[0], predictions[1])
        np.testing.assert_array_equal(predictions[0], predictions[2])
        assert jax.config.read("jax_enable_x64") is enable_x64

        integer_observations = PairwiseObservations.from_pairs(
            [("unit/7", "unit/20")], [1], target=target, sampling_unit_ids=region.sampling_unit_ids
        )
        assert integer_observations.values.dtype == dtype
        assert bool(jnp.isnan(integer_observations.values[0, 2]))
        assert float(integer_observations.observed_values[0]) == 1.0


def test_coordinate_declarations_reject_unknown_axes_and_conflicting_feature_labels(
    observation_problem,
):
    from dataclasses import replace

    region, _, _ = observation_problem()
    with pytest.raises(ValueError):
        replace(region, features=cx.field(region.feature_array, "row", "column", "channel"))
    field = cx.field(
        region.feature_array,
        "row",
        "column",
        cx.LabeledAxis("feature", np.array(["elevation", "canopy"])),
    )
    with pytest.raises(ValueError):
        replace(region, features=field, feature_names=("canopy", "elevation"))
    with pytest.raises(ValueError):
        replace(
            region,
            features=cx.field(
                region.feature_array, "row", "column", cx.LabeledAxis("feature", np.array([1, 2]))
            ),
        )


def test_equivalent_matrix_and_pair_inputs_train_equivalent_models(observation_problem):
    region, target, model = observation_problem()
    matrix = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, [[0, 1, 4], [1, 0, 1], [4, 1, 0]], target=target
    )
    pairs = PairwiseObservations.from_pairs(
        [("unit/99", "unit/7"), ("unit/20", "unit/99"), ("unit/7", "unit/20")],
        [4, 1, 1],
        target=target,
    )
    left = fit(
        region,
        matrix,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=2),
    )
    right = fit(
        region,
        pairs,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=2),
    )
    np.testing.assert_allclose(
        left.model.predict(region).values, right.model.predict(region).values, rtol=1e-6
    )
    assert left.history[-1].training_loss == right.history[-1].training_loss


def test_incomplete_pairs_remain_absent_through_fitting(ScalarEmbedding, observation_problem):
    region, target, _ = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")],
        [1],
        target=target,
        sampling_unit_ids=region.sampling_unit_ids,
    )
    result = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    assert observations.observed_pairs == (("unit/7", "unit/20"),)
    assert np.isnan(observations.values[0, 2])
    assert result.history[0].training_loss == pytest.approx(0, abs=1e-14)
    # Unobserved pairs can still be predicted, but never become zero training targets.
    np.testing.assert_allclose(result.model.predict(region).as_matrix()[0, 2], 4)


def test_supplied_partitions_select_observations_and_reject_invalid_membership(observation_problem):
    import pytest

    from ilg_toolkit import ObservationPartition

    region, target, model = observation_problem()
    observations = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, [[0, 1, 4], [1, 0, 1], [4, 1, 0]], target=target
    )
    training = ObservationPartition(region.name, (("unit/7", "unit/20"),), role="training")
    validation = ObservationPartition(region.name, (("unit/7", "unit/99"),), role="validation")
    result = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    )
    assert result.history[0].validation_loss is not None
    with pytest.raises(ValueError):
        fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            validation=(region, observations),
        )
    wrong_region = ObservationPartition("somewhere-else", training.pairs)
    with pytest.raises(ValueError):
        fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            partition=wrong_region,
        )
    incomplete = PairwiseObservations.from_pairs(training.pairs, [1], target=target)
    absent = ObservationPartition(region.name, (("unit/7", "unit/99"),))
    with pytest.raises(ValueError):
        fit(
            region,
            incomplete,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            partition=absent,
        )


def test_feature_order_and_sampling_unit_kinds_are_explicit(observation_problem):
    from dataclasses import replace

    import pytest

    region, target, model = observation_problem()
    region = replace(
        region,
        feature_names=("elevation", "canopy"),
        sampling_unit_kinds=("individual", "population", "individual"),
    )
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [1], target=target)
    result = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    assert region.sampling_unit_kinds == ("individual", "population", "individual")
    assert result.model.feature_names == ("elevation", "canopy")
    swapped = replace(
        region,
        features=cx.field(region.feature_array, "row", "column", "feature"),
        feature_names=("canopy", "elevation"),
    )
    with pytest.raises(ValueError):
        result.model.predict(swapped)
    relatedness = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [-0.2], target=TargetSpec("relatedness", kind="relatedness")
    )
    with pytest.raises(ValueError):
        fit(
            region,
            relatedness,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )


def test_invalid_observations_and_partitions_fail_at_the_public_boundary(observation_problem):
    import pytest

    from ilg_toolkit import ObservationPartition

    region, target, model = observation_problem()
    for pairs, values, labels in [
        ([("a", "b")], [np.nan], None),
        ([("a", "b")], [np.inf], None),
        ([("a", "a")], [1], None),
        ([("a", "b"), ("b", "a")], [1, 1], None),
        ([("a", "b")], [1], ("a", "c")),
    ]:
        with pytest.raises(ValueError):
            PairwiseObservations.from_pairs(pairs, values, target=target, sampling_unit_ids=labels)
    unknown = PairwiseObservations.from_pairs([("unit/7", "missing")], [1], target=target)
    with pytest.raises(ValueError):
        fit(
            region,
            unknown,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [1], target=target)
    for (partition,) in [
        (ObservationPartition(region.name, (("unit/7", "missing"),)),),
        (ObservationPartition(region.name, observations.observed_pairs, role="validation"),),
    ]:
        with pytest.raises(ValueError):
            fit(
                region,
                observations,
                distance_model=model,
                config=TrainingConfig(objective="direct_log1p", epochs=0),
                partition=partition,
            )
    with pytest.raises(ValueError):
        ObservationPartition(region.name, ())
    with pytest.raises(ValueError):
        ObservationPartition(region.name, (("a", "b"), ("b", "a")))


def test_sqrt_transform_has_an_explicit_nonnegative_codomain_and_preserves_measurements():
    import pytest

    target = TargetSpec("divergence", units="index", transform="sqrt")
    np.testing.assert_allclose(target.forward([0, 4, 9]), [0, 2, 3])
    np.testing.assert_allclose(target.inverse([0, 2, 3]), [0, 4, 9])
    with pytest.raises(ValueError):
        target.inverse([-1])
    with pytest.raises(ValueError):
        target.forward([-1])
    observations = PairwiseObservations.from_pairs([("a", "b")], [0.123456789012345], target=target)
    assert observations.observed_values[0] == 0.123456789012345
