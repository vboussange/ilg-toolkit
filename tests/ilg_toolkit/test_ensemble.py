"""Original-scale aggregation of ordered fitted models."""

from dataclasses import replace

import coordax as cx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import Ensemble


def test_ordered_members_are_snapshotted_and_repeated_entries_are_preserved(known_ensemble):
    region, original, _ = known_ensemble()
    caller_members = [original.members[1], original.members[0], original.members[1]]
    ensemble = Ensemble(caller_members)
    caller_members.clear()
    prediction = ensemble.predict(region, pairs=[("d", "b"), ("a", "b")])
    assert len(ensemble.members) == 3
    assert prediction.member_ids == (0, 1, 2)
    assert prediction.prediction.pairs == (("d", "b"), ("a", "b"))
    np.testing.assert_allclose(prediction.member_values, [[16, 4], [4, 1], [16, 4]])
    np.testing.assert_allclose(prediction.prediction.values, [12, 3])


def test_multispecies_predictions_and_distances_preserve_member_and_species_order(known_ensemble):
    region, ensemble, _ = known_ensemble(num_species=2)
    species = ("newt", "frog")
    predictions = ensemble.predict(region, species=species)
    ecological = ensemble.ecological_distances(region, species=species)
    assert tuple(predictions) == species
    for name in species:
        reference = np.stack(
            [model.predict(region, species=name).values for model in ensemble.members]
        )
        np.testing.assert_array_equal(predictions[name].member_values, reference)
        np.testing.assert_allclose(predictions[name].prediction.values, reference.mean(axis=0))
        for index, member in enumerate(ensemble.members):
            np.testing.assert_array_equal(
                ecological[name][index],
                member.distance_model.predict(region, species=name).as_matrix(),
            )
        np.testing.assert_array_equal(
            ensemble.predict(region, species=name).prediction.values,
            predictions[name].prediction.values,
        )


def test_members_apply_target_inverse_before_averaging(known_ensemble, FeatureEmbedding):
    from ilg_toolkit import DirectCalibrationModule, TargetSpec

    region, ensemble, _ = known_ensemble()
    target = TargetSpec("divergence", units="index", transform="log1p")
    models = [
        replace(
            ensemble.members[0],
            distance_model=FeatureEmbedding(jnp.sqrt(jnp.log(value))),
            calibration=DirectCalibrationModule(target),
        )
        for value in (2.0, 10.0)
    ]
    result = Ensemble(models).predict(region, pairs=[("a", "b")])
    np.testing.assert_allclose(result.member_values, [[1], [9]], atol=1e-12)
    np.testing.assert_allclose(result.prediction.values, [5], atol=1e-12)
    np.testing.assert_allclose(result.member_spread, [4], atol=1e-12)


def test_original_scale_aggregation_preserves_inverse_transform_and_large_values():
    from ilg_toolkit import aggregate_predictions, Prediction

    with jax.enable_x64():
        for levels, expected_mean, expected_spread in [
            ((np.expm1(np.log(2)), np.expm1(np.log(10))), 5, 4),
            ((1e308, 1e308), 1e308, 0),
            ((-1e308, 1e308), 0, 1e308),
        ]:
            predictions = tuple(
                Prediction(jnp.asarray([level]), (("b", "a"),), "alpine", ("a", "b", "extra"))
                for level in levels
            )
            result = aggregate_predictions(predictions)
            assert result.member_ids == (0, 1)
            assert result.prediction.pairs == (("b", "a"),)
            assert result.prediction.values.dtype == jnp.float64
            np.testing.assert_allclose(result.prediction.values, [expected_mean], rtol=1e-14)
            np.testing.assert_allclose(result.member_spread, [expected_spread], rtol=1e-14)


def test_incompatible_members_and_misaligned_predictions_report_position(known_ensemble):
    from ilg_toolkit import aggregate_predictions, DirectCalibrationModule, TargetSpec

    region, ensemble, _ = known_ensemble()
    first = ensemble.members[0]
    for changed in (
        replace(first, calibration=DirectCalibrationModule(TargetSpec("another target"))),
        replace(first, feature_names=("changed",)),
        replace(
            first, distance_model=type(first.distance_model)(jnp.asarray(1.0), species=("frog",))
        ),
    ):
        with pytest.raises(ValueError, match="member 1"):
            Ensemble([first, changed])
    with pytest.raises(ValueError):
        Ensemble([])
    with pytest.raises(ValueError, match="member 1"):
        Ensemble([first, None])  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError):
        Ensemble({"model": first})  # pyright: ignore[reportArgumentType]
    prediction = first.predict(region)
    with pytest.raises(ValueError, match="member 1"):
        aggregate_predictions([prediction, replace(prediction, region_name="other")])


def test_prediction_failure_identifies_position_without_dropping_member(
    known_ensemble, FeatureEmbedding
):
    region, ensemble, _ = known_ensemble()
    broken = replace(
        ensemble.members[0], distance_model=FeatureEmbedding(jnp.asarray(float("inf")))
    )
    requested = Ensemble([ensemble.members[0], broken, ensemble.members[1]])
    with pytest.raises(RuntimeError, match="member 1 prediction failed"):
        requested.predict(region)
    assert len(requested.members) == 3


def test_known_graph_outputs_keep_surface_mean_separate_from_distance_mean(PointConductance):
    import coordax as cx
    from ilg_toolkit import DirectCalibrationModule, LandscapeGeneticsModel, RegionBatch, TargetSpec

    with jax.enable_x64():
        region = RegionBatch(
            "graph",
            cx.field(
                np.ones((1, 4, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["cover"]))
            ),
            ("a", "b", "c", "d"),
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
        )
        calibration = DirectCalibrationModule(TargetSpec("divergence"))
        models = [
            LandscapeGeneticsModel(
                PointConductance(jnp.log(jnp.asarray(level))), calibration, ("cover",)
            )
            for level in (4.0, 1.0)
        ]
        ensemble = Ensemble(models)
        prediction = ensemble.predict(region)
        summary = ensemble.conductance_surfaces(region)
        averaged_surface_model = LandscapeGeneticsModel(
            PointConductance(jnp.log(jnp.asarray(2.5))), calibration, ("cover",)
        )
        surface_prediction = averaged_surface_model.predict(region)
    assert summary.member_ids == (0, 1)
    np.testing.assert_allclose(summary.member_values[:, 0, 0], [4, 1], atol=1e-12)
    np.testing.assert_allclose(prediction.prediction.as_matrix()[0, 1], 0.625, atol=1e-7)
    np.testing.assert_allclose(summary.values, 2.5, atol=1e-12)
    np.testing.assert_allclose(summary.member_spread, 1.5, atol=1e-12)
    np.testing.assert_allclose(surface_prediction.as_matrix()[0, 1], 0.4, atol=1e-7)


@pytest.mark.parametrize("method", ["predict", "ecological_distances", "conductance_surfaces"])
def test_feature_meanings_are_validated_for_every_ensemble_operation(known_ensemble, method):
    region, ensemble, _ = known_ensemble()
    meaning = cx.LabeledAxis("feature", np.array(["different meaning"]))
    changed = replace(region, features=cx.field(region.feature_array, "row", "column", meaning))
    with pytest.raises(RuntimeError, match="member 0.*feature"):
        getattr(ensemble, method)(changed)


def test_surface_failures_identify_member_without_restricting_genetic_prediction():
    import coordax as cx
    import equinox as eqx
    from ilg_toolkit import DirectCalibrationModule, LandscapeGeneticsModel, RegionBatch, TargetSpec
    from ilg_toolkit.models import LogLinearConductance

    region = RegionBatch(
        "grid",
        cx.field(
            np.ones((4, 4, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["cover"]))
        ),
        ("west", "east"),
        [[0, 0], [3, 3]],
    )
    first = LandscapeGeneticsModel(
        LogLinearConductance(1, patch_size=1),
        DirectCalibrationModule(TargetSpec("divergence")),
        ("cover",),
    )
    # A coarser patch size changes only the surface shape; genetic predictions still align.
    coarse = replace(first, distance_model=LogLinearConductance(1, patch_size=2))
    ensemble = Ensemble([first, coarse, first])
    result = ensemble.predict(region, pairs=[("east", "west")])
    assert result.member_ids == (0, 1, 2)
    np.testing.assert_array_equal(result.member_values[0], result.member_values[2])
    assert np.isfinite(result.prediction.values).all()
    nonfinite = replace(
        first,
        distance_model=eqx.tree_at(
            lambda model: model.intercepts, first.distance_model, jnp.asarray([float("nan")])
        ),
    )
    for second, failure in ((coarse, "shape"), (nonfinite, "nonfinite")):
        ensemble = Ensemble([first, second, first])
        with pytest.raises((ValueError, RuntimeError), match=f"(?s)member 1.*{failure}"):
            ensemble.conductance_surfaces(region)
        assert ensemble.members == (first, second, first)


def test_prediction_requires_float64(known_ensemble, float64_required):
    region, ensemble, _ = known_ensemble()
    with float64_required():
        ensemble.predict(region)
