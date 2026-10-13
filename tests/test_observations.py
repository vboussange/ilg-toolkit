"""Public prepared-input and fit/predict contracts from the approved spec seam."""

from typing import cast, final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import PairwiseObservations, RegionBatch, TargetSpec, TrainingConfig, fit
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class LinearEmbedding(EmbeddingDistanceModel):
    """Small real distance encoder using the documented model extension boundary."""

    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features[..., :1] * self.weight


def problem() -> tuple[RegionBatch, TargetSpec, LinearEmbedding]:
    region = RegionBatch(
        "anywhere",
        cx.field(np.array([[[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]]), "row", "column", "feature"),
        ("unit/7", "unit/20", "unit/99"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    target = TargetSpec("pair dissimilarity", units="index")
    return region, target, LinearEmbedding(jnp.asarray(0.7))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_observation_dtype_and_absent_pairs_survive_matrix_pair_and_region_alignment(dtype):
    region, target, _ = problem()
    measurements = jnp.asarray([0.123456789012345], dtype=dtype)
    paired = PairwiseObservations.from_pairs([("unit/20", "unit/7")], measurements, target=target)
    matrix = PairwiseObservations.from_matrix(
        ("unit/20", "unit/7"),
        jnp.asarray([[0, measurements[0]], [measurements[0], 0]], dtype=dtype),
        target=target,
    )
    for observations in (paired, matrix):
        assert isinstance(observations.values, jax.Array)
        assert observations.values.dtype == observations.observed_values.dtype == dtype
        aligned = observations.aligned_values(region)
        assert isinstance(aligned, jax.Array)
        assert aligned.dtype == dtype
        assert float(aligned[0, 1]) == float(measurements[0])
        assert bool(jnp.isnan(aligned[0, 2]))
        np.testing.assert_array_equal(jnp.diag(aligned), 0)


def test_region_requires_named_features_and_normalizes_numpy_data_without_narrowing():
    from dataclasses import replace

    region, _, _ = problem()
    with pytest.raises(TypeError, match="coordax.Field"):
        replace(region, features=cast(cx.Field, region.feature_array))
    features = np.array([[[0.123456789012345, 2.0], [1.0, 3.0], [2.0, 4.0]]])
    field = cx.field(features, "row", "column", "feature")
    assert isinstance(field.data, np.ndarray)
    normalized = replace(region, features=field, grid_positions=[[0, 0], [0, 1], [0, 2]])
    assert isinstance(normalized.feature_array, jax.Array)
    assert normalized.feature_array.dtype == jnp.float64
    np.testing.assert_array_equal(normalized.feature_array, features)
    np.testing.assert_array_equal(normalized.pixel_nodes, [0, 1, 2])
    with pytest.raises(ValueError, match="integer"):
        replace(region, grid_positions=np.array([[0, 0], [0, 1.5], [0, 2]]))
    with pytest.raises(ValueError, match="outside"):
        replace(region, grid_positions=[[0, 0], [0, 1], [0, 3]])


def test_fresh_fit_requires_declared_encoder_and_objective_while_resume_reuses_them():
    region, target, encoder = problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    omitted = {}
    with pytest.raises(TypeError, match="objective"):
        TrainingConfig(**omitted)
    config = TrainingConfig(objective="direct_log1p", epochs=0)
    with pytest.raises(ValueError, match="explicit.*model|explicit.*encoder"):
        fit(region, observations, config=config)
    with pytest.raises(ValueError, match="explicit.*TrainingConfig"):
        fit(region, observations, distance_model=encoder)
    initial = fit(region, observations, distance_model=encoder, config=config)

    continued = fit(region, observations, state=initial)
    assert continued.history == initial.history

    assert continued.config == config
    np.testing.assert_array_equal(
        continued.model.predict(region).values, initial.model.predict(region).values
    )
    with pytest.raises(ValueError, match="omit distance_model"):
        fit(region, observations, state=initial, distance_model=encoder)


def test_encoder_distance_boundary_accepts_array_like_rasters_and_nodes():
    encoder = LinearEmbedding(jnp.asarray(1.0))
    distances = encoder.predict_distances([[[0.0], [1.0], [2.0]]], [0, 1, 2])
    np.testing.assert_array_equal(distances, [[0, 1, 4], [1, 0, 1], [4, 1, 0]])


def test_labelled_raster_axes_normalize_without_changing_feature_meaning():
    """A named-axis permutation describes the same prepared raster and locations."""
    from dataclasses import replace

    region, target, model = problem()
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
    assert isinstance(labelled.features, cx.Field)
    assert labelled.features.dims == ("row", "column", "feature")
    row_axis = labelled.features.axes["row"]
    column_axis = labelled.features.axes["column"]
    assert isinstance(row_axis, cx.LabeledAxis)
    assert isinstance(column_axis, cx.LabeledAxis)
    np.testing.assert_array_equal(row_axis.ticks, [100])
    np.testing.assert_array_equal(column_axis.ticks, [10, 20, 30])
    assert labelled.feature_names == names
    np.testing.assert_array_equal(actual.predict(labelled).values, expected.predict(plain).values)


def test_public_landscape_and_target_predictions_return_jax_values():
    region, _, model = problem()
    target = TargetSpec("divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    fitted = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    ).model
    distances = fitted.ecological_distances(region)
    prediction = fitted.predict(region)
    selected = fitted.predict(region, [("unit/20", "unit/7")])
    assert isinstance(distances, jax.Array)
    assert isinstance(prediction.values, jax.Array)
    assert isinstance(selected.values, jax.Array)
    assert isinstance(target.forward(jnp.array([0.5])), jax.Array)
    assert isinstance(target.inverse(jnp.array([0.5])), jax.Array)
    np.testing.assert_allclose(distances[0, 1], 0.49, rtol=1e-6)
    np.testing.assert_allclose(selected.values, [np.expm1(0.49)], rtol=1e-6)
    np.testing.assert_array_equal(np.diag(prediction.as_matrix()), 0)


@pytest.mark.parametrize("enable_x64", [False, True])
def test_array_like_observations_and_targets_follow_configured_precision(enable_x64):
    precise = 0.123456789012345
    with jax.enable_x64(enable_x64):
        region, target, model = problem()
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
                assert isinstance(transformed, jax.Array)
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


def test_coordinate_declarations_reject_unknown_axes_and_conflicting_feature_labels():
    from dataclasses import replace

    region, _, _ = problem()
    with pytest.raises(ValueError, match="row, column, feature"):
        replace(region, features=cx.field(region.feature_array, "row", "column", "channel"))
    field = cx.field(
        region.feature_array,
        "row",
        "column",
        cx.LabeledAxis("feature", np.array(["elevation", "canopy"])),
    )
    with pytest.raises(ValueError, match="meanings and order"):
        replace(region, features=field, feature_names=("canopy", "elevation"))
    with pytest.raises(ValueError, match="feature.*string"):
        replace(
            region,
            features=cx.field(
                region.feature_array, "row", "column", cx.LabeledAxis("feature", np.array([1, 2]))
            ),
        )


def test_equivalent_matrix_and_pair_inputs_train_equivalent_models():
    region, target, model = problem()
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


def test_incomplete_pairs_remain_absent_through_fitting():
    region, target, _ = problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")],
        [1],
        target=target,
        sampling_unit_ids=region.sampling_unit_ids,
    )
    result = fit(
        region,
        observations,
        distance_model=LinearEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    assert observations.observed_pairs == (("unit/7", "unit/20"),)
    assert np.isnan(observations.values[0, 2])
    assert result.history[0].training_loss == pytest.approx(0, abs=1e-14)
    # Unobserved pairs can still be predicted, but never become zero training targets.
    np.testing.assert_allclose(result.model.predict(region).as_matrix()[0, 2], 4)


def test_explicit_transform_returns_original_units_and_separate_ecological_distances():
    region, _, _ = problem()
    target = TargetSpec("explicitly transformed divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [np.e - 1], target=target
    )
    result = fit(
        region,
        observations,
        distance_model=LinearEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", epochs=0),
    )
    prediction = result.model.predict(region)
    assert result.history[0].training_loss < 1e-12
    assert prediction.scale == "original"
    assert prediction.target.transform == "log1p"
    assert prediction.target.units == "index"
    np.testing.assert_allclose(prediction.as_matrix()[0, 1], np.e - 1, rtol=1e-6)
    np.testing.assert_allclose(result.model.ecological_distances(region)[0, 1], 1)


def test_supplied_partitions_select_observations_and_reject_invalid_membership():
    import pytest

    from ilg_toolkit import ObservationPartition

    region, target, model = problem()
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
    with pytest.raises(ValueError, match="overlap"):
        fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            validation=(region, observations),
        )
    wrong_region = ObservationPartition("somewhere-else", training.pairs)
    with pytest.raises(ValueError, match="region"):
        fit(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            partition=wrong_region,
        )
    incomplete = PairwiseObservations.from_pairs(training.pairs, [1], target=target)
    absent = ObservationPartition(region.name, (("unit/7", "unit/99"),))
    with pytest.raises(ValueError, match="unobserved"):
        fit(
            region,
            incomplete,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            partition=absent,
        )


def test_feature_order_and_sampling_unit_kinds_are_explicit():
    from dataclasses import replace

    import pytest

    region, target, model = problem()
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
    with pytest.raises(ValueError, match="feature.*order|feature.*contract"):
        result.model.predict(swapped)
    relatedness = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [-0.2], target=TargetSpec("relatedness", kind="relatedness")
    )
    with pytest.raises(ValueError, match="nonnegative dissimilarities"):
        fit(
            region,
            relatedness,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )


def test_invalid_observations_and_partitions_fail_at_the_public_boundary():
    import pytest

    from ilg_toolkit import ObservationPartition

    region, target, model = problem()
    for pairs, values, labels, message in [
        ([("a", "b")], [np.nan], None, "finite"),
        ([("a", "b")], [np.inf], None, "finite"),
        ([("a", "a")], [1], None, "distinct"),
        ([("a", "b"), ("b", "a")], [1, 1], None, "unique"),
        ([("a", "b")], [1], ("a", "c"), "endpoint"),
    ]:
        with pytest.raises(ValueError, match=message):
            PairwiseObservations.from_pairs(pairs, values, target=target, sampling_unit_ids=labels)
    unknown = PairwiseObservations.from_pairs([("unit/7", "missing")], [1], target=target)
    with pytest.raises(ValueError, match="outside region"):
        fit(
            region,
            unknown,
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", epochs=0),
        )
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [1], target=target)
    for partition, message in [
        (ObservationPartition(region.name, (("unit/7", "missing"),)), "endpoints"),
        (
            ObservationPartition(region.name, observations.observed_pairs, role="validation"),
            "training",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            fit(
                region,
                observations,
                distance_model=model,
                config=TrainingConfig(objective="direct_log1p", epochs=0),
                partition=partition,
            )
    with pytest.raises(ValueError, match="nonempty"):
        ObservationPartition(region.name, ())
    with pytest.raises(ValueError, match="unique"):
        ObservationPartition(region.name, (("a", "b"), ("b", "a")))


def test_sqrt_transform_has_an_explicit_nonnegative_codomain_and_preserves_measurements():
    import pytest

    target = TargetSpec("divergence", units="index", transform="sqrt")
    np.testing.assert_allclose(target.forward([0, 4, 9]), [0, 2, 3])
    np.testing.assert_allclose(target.inverse([0, 2, 3]), [0, 4, 9])
    with pytest.raises(ValueError, match="nonnegative transformed"):
        target.inverse([-1])
    with pytest.raises(ValueError, match="nonnegative values"):
        target.forward([-1])
    observations = PairwiseObservations.from_pairs([("a", "b")], [0.123456789012345], target=target)
    assert observations.observed_values[0] == 0.123456789012345


@pytest.mark.parametrize("field", ["name", "units", "kind", "transform"])
@pytest.mark.parametrize("value", [1, True, ["genetic"], None, ""])
def test_target_metadata_requires_nonempty_strings_at_construction(field, value):
    declared = dict(
        name="genetic divergence", units="index", kind="dissimilarity", transform="identity"
    )
    declared[field] = value
    with pytest.raises(ValueError, match="nonempty strings"):
        TargetSpec(**declared)
