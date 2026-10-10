"""Public prepared-input and fit/predict contracts from the approved spec seam."""

from typing import final

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

    weight: jnp.ndarray
    patch_size: int = eqx.field(static=True, default=1)

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features[..., :1] * self.weight


def problem():
    region = RegionBatch(
        "anywhere",
        np.array([[[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]]]),
        ("unit/7", "unit/20", "unit/99"),
        np.array([[0, 0], [0, 1], [0, 2]]),
    )
    target = TargetSpec("pair dissimilarity", units="index")
    return region, target, LinearEmbedding(jnp.asarray(0.7))


def test_labelled_raster_axes_normalize_without_changing_feature_meaning():
    """A named-axis permutation describes the same prepared raster and locations."""
    from dataclasses import replace

    region, target, model = problem()
    names = ("elevation", "canopy")
    plain = replace(region, features=region.feature_array, feature_names=names)
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
    expected = fit(plain, observations, model=model, config=TrainingConfig(epochs=1)).model
    actual = fit(labelled, observations, model=model, config=TrainingConfig(epochs=1)).model
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
    fitted = fit(region, observations, model=model, config=TrainingConfig(epochs=0)).model
    scores = fitted.landscape_scores(region)
    prediction = fitted.predict(region)
    selected = fitted.predict_pairs(region, [("unit/20", "unit/7")])
    assert isinstance(scores, jax.Array)
    assert isinstance(prediction.values, jax.Array)
    assert isinstance(selected.values, jax.Array)
    assert isinstance(target.forward(jnp.array([0.5])), jax.Array)
    assert isinstance(target.inverse(jnp.array([0.5])), jax.Array)
    np.testing.assert_allclose(scores[0, 1], 0.49, rtol=1e-6)
    np.testing.assert_allclose(selected.values, [np.expm1(0.49)], rtol=1e-6)
    np.testing.assert_array_equal(np.diag(prediction.values), 0)


def test_precise_observations_require_explicit_device_precision_without_toggling_it():
    precise = 0.123456789012345
    with jax.enable_x64(False):
        region, target, model = problem()
        assert isinstance(region.feature_array, jax.Array)
        assert isinstance(region.grid_positions, jax.Array)
        observations = PairwiseObservations.from_pairs(
            [("unit/7", "unit/20")], [precise], target=target
        )
        assert observations.observed_values[0] == precise
        assert jax.config.read("jax_enable_x64") is False
        with pytest.raises(RuntimeError, match="float64.*JAX_ENABLE_X64"):
            target.forward(observations.observed_values)
        with pytest.raises(RuntimeError, match="float64.*JAX_ENABLE_X64"):
            target.inverse(np.array([precise]))
        with pytest.raises(RuntimeError, match="float64.*JAX_ENABLE_X64"):
            fit(region, observations, model=model, config=TrainingConfig(epochs=0))
        assert jax.config.read("jax_enable_x64") is False
    with jax.enable_x64():
        transformed = target.forward(observations.observed_values)
        assert isinstance(transformed, jax.Array)
        assert transformed.dtype == jnp.float64
        assert float(transformed[0]) == precise


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
    left = fit(region, matrix, model=model, config=TrainingConfig(epochs=2))
    right = fit(region, pairs, model=model, config=TrainingConfig(epochs=2))
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
        model=LinearEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
    )
    assert observations.observed_pairs == (("unit/7", "unit/20"),)
    assert np.isnan(observations.values[0, 2])
    assert result.history[0].training_loss == pytest.approx(0, abs=1e-14)
    # Unobserved pairs can still be predicted, but never become zero training targets.
    np.testing.assert_allclose(result.model.predict(region).values[0, 2], 4)


def test_explicit_transform_returns_original_units_and_separate_landscape_scores():
    region, _, _ = problem()
    target = TargetSpec("explicitly transformed divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [np.e - 1], target=target
    )
    result = fit(
        region,
        observations,
        model=LinearEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
    )
    prediction = result.model.predict(region)
    assert result.history[0].training_loss < 1e-12
    assert prediction.scale == "original"
    assert prediction.target.transform == "log1p"
    assert prediction.target.units == "index"
    np.testing.assert_allclose(prediction.values[0, 1], np.e - 1, rtol=1e-6)
    np.testing.assert_allclose(result.model.landscape_scores(region)[0, 1], 1)


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
        model=model,
        config=TrainingConfig(epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    )
    assert result.history[0].validation_loss is not None
    with pytest.raises(ValueError, match="overlap"):
        fit(
            region,
            observations,
            model=model,
            config=TrainingConfig(epochs=0),
            validation=(region, observations),
        )
    wrong_region = ObservationPartition("somewhere-else", training.pairs)
    with pytest.raises(ValueError, match="region"):
        fit(region, observations, model=model, partition=wrong_region)
    incomplete = PairwiseObservations.from_pairs(training.pairs, [1], target=target)
    absent = ObservationPartition(region.name, (("unit/7", "unit/99"),))
    with pytest.raises(ValueError, match="unobserved"):
        fit(region, incomplete, model=model, partition=absent)


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
    result = fit(region, observations, model=model, config=TrainingConfig(epochs=0))
    assert region.sampling_unit_kinds == ("individual", "population", "individual")
    assert result.model.feature_names == ("elevation", "canopy")
    swapped = replace(region, features=region.feature_array, feature_names=("canopy", "elevation"))
    with pytest.raises(ValueError, match="feature.*order|feature.*contract"):
        result.model.predict(swapped)
    relatedness = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [-0.2], target=TargetSpec("relatedness", kind="relatedness")
    )
    with pytest.raises(ValueError, match="nonnegative dissimilarities"):
        fit(region, relatedness, model=model)


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
        fit(region, unknown, model=model)
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [1], target=target)
    for partition, message in [
        (ObservationPartition(region.name, (("unit/7", "missing"),)), "endpoints"),
        (
            ObservationPartition(region.name, observations.observed_pairs, role="validation"),
            "training",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            fit(region, observations, model=model, partition=partition)
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
