"""Scientific behavior of data."""

from typing import cast

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import Log1pLoss, PairwiseObservations, TargetSpec, train, TrainingConfig


def test_species_observations_align_repeated_ids_without_cross_species_pairs(observation_problem):
    from dataclasses import replace

    region, target, _ = observation_problem()
    region = replace(
        region,
        sampling_unit_ids=("a", "b", "a", "b"),
        grid_positions=[[0, 0], [0, 1], [0, 1], [0, 2]],
        sampling_unit_kinds=None,
        species=("frog", "frog", "newt", "newt"),
    )
    for species, value, indices in (("frog", 2.0, (0, 1)), ("newt", 5.0, (2, 3))):
        observations = PairwiseObservations.from_pairs(
            [("b", "a")], [value], target=target, species=species
        )
        matrix_observations = PairwiseObservations.from_matrix(
            ("a", "b"), [[0, value], [value, 0]], target=target, species=species
        )
        for observed in (observations, matrix_observations):
            pairs, values = observed.aligned_pairs(region)
            np.testing.assert_array_equal(pairs, np.asarray(indices)[:, None])
            np.testing.assert_array_equal(values, [value])
            aligned = observed.aligned_values(region)
            assert np.isnan(aligned[0, 2])
    with pytest.raises(ValueError):
        replace(region, species=("frog",) * 4)
    with pytest.raises(ValueError):
        replace(region, species=("frog",))
    with pytest.raises(ValueError):
        replace(observations, species=None).aligned_pairs(region)
    with pytest.raises(ValueError):
        replace(observations, species="absent").aligned_pairs(region)


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
    expected = train(
        plain,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1),
    ).selected_model
    actual = train(
        labelled,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1),
    ).selected_model
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
            fitted = train(
                region,
                observations,
                distance_model=model,
                config=TrainingConfig(loss=Log1pLoss(), steps=0),
            )
            prediction = fitted.selected_model.predict(region)
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
    left = train(
        region,
        matrix,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2),
    )
    right = train(
        region,
        pairs,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2),
    )
    np.testing.assert_allclose(
        left.selected_model.predict(region).values,
        right.selected_model.predict(region).values,
        rtol=1e-6,
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
    result = train(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    assert observations.observed_pairs == (("unit/7", "unit/20"),)
    assert np.isnan(observations.values[0, 2])
    assert result.history[0].training_loss == pytest.approx(0, abs=1e-14)
    # Unobserved pairs can still be predicted, but never become zero training targets.
    np.testing.assert_allclose(result.selected_model.predict(region).as_matrix()[0, 2], 4)


def test_selected_observations_are_disjoint_for_training_and_validation(observation_problem):
    region, target, model = observation_problem()
    observations = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, [[0, 1, 4], [1, 0, 1], [4, 1, 0]], target=target
    )
    training = observations.select_pairs([("unit/7", "unit/20")])
    validation = observations.select_pairs([("unit/7", "unit/99")])
    result = train(
        region,
        training,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
        validation=(region, validation),
    )
    assert result.history[0].validation_loss is not None
    with pytest.raises(ValueError):
        train(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
            validation=(region, observations),
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
    result = train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    assert region.sampling_unit_kinds == ("individual", "population", "individual")
    assert result.selected_model.feature_names == ("elevation", "canopy")
    swapped = replace(
        region,
        features=cx.field(region.feature_array, "row", "column", "feature"),
        feature_names=("canopy", "elevation"),
    )
    with pytest.raises(ValueError):
        result.selected_model.predict(swapped)
    relatedness = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [-0.2], target=TargetSpec("relatedness", kind="relatedness")
    )
    with pytest.raises(ValueError):
        train(
            region,
            relatedness,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )


def test_invalid_observations_fail_at_the_public_boundary(observation_problem):
    import pytest

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
        train(
            region,
            unknown,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )


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


def test_odds_transform_linearizes_bounded_dissimilarities_and_round_trips():
    target = TargetSpec("Fst", units="proportion", transform="odds")
    values = np.array([0.0, 0.05, 0.2, 0.5, 0.9])
    model_scale = target.forward(values)
    np.testing.assert_allclose(model_scale, values / (1 - values))
    np.testing.assert_allclose(target.inverse(model_scale), values)
    observations = PairwiseObservations.from_pairs([("a", "b")], [0.123456789012345], target=target)
    assert observations.observed_values[0] == 0.123456789012345


@pytest.mark.parametrize("invalid", [-0.01, 1.0, 1.5, np.nan, np.inf])
def test_odds_transform_rejects_values_outside_the_unit_interval(invalid):
    target = TargetSpec("Fst", units="proportion", transform="odds")
    with pytest.raises(ValueError):
        target.forward([0.1, invalid])


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_select_pairs_preserves_metadata_precision_and_deterministic_alignment(dtype):
    target = TargetSpec("divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("c", "a"), ("b", "c"), ("a", "b")],
        jnp.asarray([0.123456789012345, 2, 3], dtype=dtype),
        target=target,
        species="newt",
        sampling_unit_ids=("c", "b", "a"),
    )
    selected = observations.select_pairs([("a", "c"), ("b", "a")])
    assert selected.target == target and selected.species == "newt"
    assert selected.sampling_unit_ids == ("c", "b", "a")
    assert selected.observed_pairs == (("c", "a"), ("b", "a"))
    assert selected.values.dtype == dtype
    np.testing.assert_array_equal(
        selected.observed_values, observations.observed_values[jnp.asarray([1, 2])]
    )
    for pairs in (
        [],
        ["ab"],
        [("a", "a")],
        [("a", "missing")],
        [("a", "b"), ("b", "a")],
        [("", "a")],
        [(1, "a")],
        [("a",)],
    ):
        with pytest.raises(ValueError):
            observations.select_pairs(pairs)
    with pytest.raises(ValueError, match="unobserved"):
        selected.select_pairs([("c", "b")])
