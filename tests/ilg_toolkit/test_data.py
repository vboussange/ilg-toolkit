"""Tests of region batches, pairwise genetic observations and target transforms."""

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
        species=("frog", "frog", "newt", "newt"),
    )
    for species, value, indices in (("frog", 2.0, (0, 1)), ("newt", 5.0, (2, 3))):
        observations = PairwiseObservations.from_pairs(
            [("b", "a")], [value], target=target, species=species
        )
        matrix_observations = PairwiseObservations(
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
    matrix = PairwiseObservations(
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
    field = cx.field(features, "row", "column", region.features.axes["feature"])
    normalized = replace(region, features=field, grid_positions=[[0, 0], [0, 1], [0, 2]])
    assert normalized.feature_array.dtype == jnp.float64
    np.testing.assert_array_equal(normalized.feature_array, features)
    np.testing.assert_array_equal(normalized.pixel_nodes, [0, 1, 2])
    with pytest.raises(ValueError):
        replace(region, grid_positions=np.array([[0, 0], [0, 1.5], [0, 2]]))
    with pytest.raises(ValueError):
        replace(region, grid_positions=[[0, 0], [0, 1], [0, 3]])


def test_labeled_raster_axes_normalize_without_changing_feature_meaning(observation_problem):
    """A named-axis permutation describes the same landscape features and grid positions."""
    from dataclasses import replace

    plain, target, model = observation_problem()
    names = plain.feature_names
    field = cx.field(
        plain.feature_array,
        cx.LabeledAxis("row", np.array([100])),
        cx.LabeledAxis("column", np.array([10, 20, 30])),
        cx.LabeledAxis("feature", np.array(names)),
    ).order_as("feature", "column", "row")
    labeled = replace(plain, features=field)
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
        labeled,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1),
    ).selected_model
    assert labeled.features.dims == ("row", "column", "feature")
    row_axis = labeled.features.axes["row"]
    column_axis = labeled.features.axes["column"]
    np.testing.assert_array_equal(cast(cx.LabeledAxis, row_axis).ticks, [100])
    np.testing.assert_array_equal(cast(cx.LabeledAxis, column_axis).ticks, [10, 20, 30])
    assert labeled.feature_names == names
    np.testing.assert_array_equal(actual.predict(labeled).values, expected.predict(plain).values)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_array_like_observations_and_targets_follow_configured_precision(
    observation_problem, enable_x64
):
    precise = 0.123456789012345
    with jax.enable_x64(enable_x64):
        region, target, _ = observation_problem()
        dtype = jnp.float64 if enable_x64 else jnp.float32
        expected = precise if enable_x64 else float(np.float32(precise))
        for values in ([precise], np.array([precise]), jnp.asarray([precise])):
            observations = PairwiseObservations.from_pairs(
                [("unit/7", "unit/20")], values, target=target
            )
            assert observations.values.dtype == dtype
            assert float(observations.observed_values[0]) == expected
            for transformed in (target.forward(values), target.inverse(values)):
                assert transformed.dtype == dtype
                assert float(transformed[0]) == expected
        integer_observations = PairwiseObservations.from_pairs(
            [("unit/7", "unit/20")], [1], target=target, sampling_unit_ids=region.sampling_unit_ids
        )
        assert integer_observations.values.dtype == dtype
        assert bool(jnp.isnan(integer_observations.values[0, 2]))
        assert float(integer_observations.observed_values[0]) == 1.0


def test_array_like_observations_give_identical_float64_predictions(observation_problem):
    region, target, model = observation_problem()
    precise = 0.123456789012345
    predictions = []
    for values in ([precise], np.array([precise]), jnp.asarray([precise])):
        observations = PairwiseObservations.from_pairs(
            [("unit/7", "unit/20")], values, target=target
        )
        trained = train(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )
        prediction = trained.selected_model.predict(region)
        assert prediction.values.dtype == prediction.as_matrix().dtype == jnp.float64
        predictions.append(prediction.values)
    np.testing.assert_array_equal(predictions[0], predictions[1])
    np.testing.assert_array_equal(predictions[0], predictions[2])


def test_coordinate_declarations_reject_unknown_axes(observation_problem):
    from dataclasses import replace

    region, _, _ = observation_problem()
    labels = cx.LabeledAxis("channel", np.array(region.feature_names))
    with pytest.raises(ValueError):
        replace(region, features=cx.field(region.feature_array, "row", "column", labels))


def test_equivalent_matrix_and_pair_inputs_train_equivalent_models(observation_problem):
    region, target, model = observation_problem()
    matrix = PairwiseObservations(
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


def test_incomplete_pairs_remain_absent_through_training(ScalarEmbedding, observation_problem):
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
    # Unobserved pairs are predicted and never become zero training targets.
    np.testing.assert_allclose(result.selected_model.predict(region).as_matrix()[0, 2], 4)


def test_selected_observations_are_disjoint_for_training_and_validation(observation_problem):
    region, target, model = observation_problem()
    observations = PairwiseObservations(
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


def test_feature_order_is_part_of_the_query_contract(observation_problem):
    from dataclasses import replace

    import pytest

    region, target, model = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [1], target=target)
    result = train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=0),
    )
    assert result.selected_model.feature_names == ("elevation", "canopy")
    swapped = replace(
        region,
        features=cx.field(
            region.feature_array,
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["canopy", "elevation"])),
        ),
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
            observations.select_pairs(pairs)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="unobserved"):
        selected.select_pairs([("c", "b")])


def _raster(y_centers, x_centers, names=("grassland", "water")):
    """Returns a projected raster converted from xarray, with row, column and feature axes."""
    import xarray as xr

    values = np.arange(len(y_centers) * len(x_centers) * len(names), dtype=np.float64)
    data = xr.DataArray(
        values.reshape(len(y_centers), len(x_centers), len(names)),
        dims=("y", "x", "band"),
        coords={"y": y_centers, "x": x_centers, "band": list(names)},
    )
    return cx.from_xarray(data.rename(y="row", x="column", band="feature"))


def test_feature_names_and_raster_coordinates_come_from_xarray_axes():
    from dataclasses import FrozenInstanceError

    from ilg_toolkit import RegionBatch

    features = _raster([25.0, 15.0, 5.0], [5.0, 15.0, 25.0, 35.0])
    region = RegionBatch("alps", features, ("a", "b"), [[0, 0], [2, 3]])
    assert region.feature_names == ("grassland", "water")
    np.testing.assert_array_equal(
        cast(cx.LabeledAxis, region.features.axes["row"]).ticks, [25, 15, 5]
    )
    np.testing.assert_array_equal(
        cast(cx.LabeledAxis, region.features.axes["column"]).ticks, [5, 15, 25, 35]
    )
    with pytest.raises((AttributeError, FrozenInstanceError)):
        region.feature_names = ("water", "grassland")  # type: ignore[misc]


def test_features_without_a_labeled_feature_axis_raise(observation_problem):
    from ilg_toolkit import RegionBatch

    region, _, _ = observation_problem()
    unlabeled = cx.field(region.feature_array, "row", "column", "feature")
    with pytest.raises(ValueError, match="labeled feature axis"):
        RegionBatch("anywhere", unlabeled, region.sampling_unit_ids, region.grid_positions)
    for labels in (["elevation", ""], ["elevation", "elevation"], [1, 2]):
        field = cx.field(
            region.feature_array, "row", "column", cx.LabeledAxis("feature", np.array(labels))
        )
        with pytest.raises(ValueError, match="Feature names"):
            RegionBatch("anywhere", field, region.sampling_unit_ids, region.grid_positions)


@pytest.mark.parametrize("descending", [True, False])
def test_points_snap_to_the_raster_cells_containing_them(descending):
    from ilg_toolkit import RegionBatch

    # Cells are 10 units wide: columns span x in [0, 40], rows span y in [0, 30].
    y_centers = [25.0, 15.0, 5.0] if descending else [5.0, 15.0, 25.0]
    features = _raster(y_centers, [5.0, 15.0, 25.0, 35.0])
    points = {
        "corner": (0.0, 30.0),  # Outer edges belong to the raster.
        "far-corner": (40.0, 0.0),
        "off-center": (14.9, 20.1),
        # A shared edge belongs to the next cell in row or column order, as in GDAL.
        "shared-edge": (10.0, 20.0),
        "center": (35.0, 5.0),
    }
    region = RegionBatch.from_points(
        "alps", features, tuple(points), np.array(list(points.values()))
    )
    expected = (
        [[0, 0], [2, 3], [0, 1], [1, 1], [2, 3]]
        if descending
        else [[2, 0], [0, 3], [2, 1], [2, 1], [0, 3]]
    )
    np.testing.assert_array_equal(region.grid_positions, expected)
    assert region.sampling_unit_ids == tuple(points)
    assert region.feature_names == ("grassland", "water")


def test_points_outside_the_raster_extent_raise_naming_the_sampling_units(observation_problem):
    from ilg_toolkit import RegionBatch

    features = _raster([25.0, 15.0, 5.0], [5.0, 15.0, 25.0, 35.0])
    points = np.array([[5.0, 5.0], [40.01, 5.0], [5.0, -0.01], [-1.0, 31.0]])
    with pytest.raises(ValueError, match="east, south, northwest"):
        RegionBatch.from_points("alps", features, ("inside", "east", "south", "northwest"), points)
    with pytest.raises(ValueError):
        RegionBatch.from_points("alps", features, ("a", "b"), np.array([[5.0, 5.0]]))
    with pytest.raises(ValueError):
        RegionBatch.from_points("alps", features, ("a", "b"), np.array([[5.0, 5.0], [np.nan, 5]]))
    region, _, _ = observation_problem()
    with pytest.raises(ValueError, match="row coordinates"):
        RegionBatch.from_points(
            "anywhere", region.features, ("a", "b"), np.array([[0.0, 0.0], [1.0, 0.0]])
        )


def test_individual_based_regions_are_declared_once_for_all_sampling_units(observation_problem):
    from dataclasses import replace

    region, _, _ = observation_problem()
    assert region.individual_based is False
    assert replace(region, individual_based=True).individual_based is True
    with pytest.raises(TypeError):
        replace(region, individual_based="individual")
