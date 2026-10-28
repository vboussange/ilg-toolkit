"""Tests of ensembles: original-scale averaging of ordered landscape genetics models."""

import warnings
from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import Ensemble, Prediction


def test_ordered_members_are_copied_and_repeated_entries_are_preserved(known_ensemble):
    region, original, _ = known_ensemble()
    caller_members = [original.members[1], original.members[0], original.members[1]]
    ensemble = Ensemble(caller_members)
    caller_members.clear()
    result = ensemble.predict(region, [("d", "b"), ("a", "b")])
    assert len(ensemble.members) == 3
    assert isinstance(result.mean, Prediction)
    assert result.mean.pairs == (("d", "b"), ("a", "b"))
    assert result.mean.sampling_unit_ids == ("a", "b", "c", "d")
    assert len(result.members) == 3
    for member, model in zip(result.members, ensemble.members, strict=True):
        assert member.pairs == result.mean.pairs
        np.testing.assert_array_equal(
            member.values, model.predict(region, [("d", "b"), ("a", "b")]).values
        )
    np.testing.assert_allclose(
        [member.values for member in result.members], [[16, 4], [4, 1], [16, 4]]
    )
    np.testing.assert_allclose(result.mean.values, [12, 3])
    every_pair = ensemble.predict(region)
    assert every_pair.mean.pairs == tuple(
        (a, b) for index, a in enumerate("abcd") for b in "abcd"[index + 1 :]
    )


def test_species_queries_average_each_species_and_keep_member_predictions(known_ensemble):
    region, ensemble, _ = known_ensemble(species=("frog", "newt"))
    query = {"newt": None, "frog": [("d", "a")]}
    results = ensemble.predict(region, query)
    assert tuple(results) == ("newt", "frog")
    assert results["frog"].mean.pairs == (("d", "a"),)
    assert results["newt"].mean.pairs == (
        ("a", "b"),
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
        ("c", "d"),
    )
    for name in query:
        assert results[name].mean.species == name
        members = [model.predict(region, query)[name] for model in ensemble.members]
        for actual, expected in zip(results[name].members, members, strict=True):
            assert actual.species == name
            np.testing.assert_array_equal(actual.values, expected.values)
        np.testing.assert_allclose(
            results[name].mean.values, np.mean([member.values for member in members], axis=0)
        )
    np.testing.assert_array_equal(
        ensemble.predict(region, {"frog": [("d", "a")]})["frog"].mean.values,
        results["frog"].mean.values,
    )
    for invalid, message in (
        (None, "Multispecies models require a query mapping"),
        ([("a", "b")], "Multispecies models require a query mapping"),
        ({"toad": None}, "Unknown species"),
        ({}, "at least one species"),
        ({"frog": [("a", "z")]}, "absent from region 'alpine': z"),
        ({"frog": []}, "at least one sampling-unit pair"),
        ({"frog": [("a", "b"), ("b", "a")]}, "reversed duplicates"),
    ):
        with pytest.raises(ValueError, match=message) as error:
            ensemble.predict(region, invalid)
        assert not hasattr(error.value, "__notes__")


def test_members_apply_target_inverse_before_averaging(known_ensemble, FeatureEmbedding):
    from ilg_toolkit import DirectCalibrationHead, TargetSpec

    region, ensemble, _ = known_ensemble()
    target = TargetSpec("divergence", units="index", transform="log1p")
    models = [
        replace(
            ensemble.members[0],
            distance_model=FeatureEmbedding(jnp.sqrt(jnp.log(value))),
            calibration_head=DirectCalibrationHead(target),
        )
        for value in (2.0, 10.0)
    ]
    result = Ensemble(models).predict(region, [("a", "b")])
    np.testing.assert_allclose([member.values for member in result.members], [[1], [9]])
    np.testing.assert_allclose(result.mean.values, [5])
    assert result.mean.values.dtype == jnp.float64


def test_averaging_large_member_predictions_does_not_overflow(known_ensemble, FeatureEmbedding):
    region, ensemble, _ = known_ensemble()
    # Squared embedding distance between features 0 and 1 is 1e308 for each member.
    member = replace(ensemble.members[0], distance_model=FeatureEmbedding(jnp.asarray(1e154)))
    result = Ensemble([member, member]).predict(region, [("a", "b")])
    np.testing.assert_allclose(result.mean.values, [1e308], rtol=1e-14)


def test_incompatible_members_report_position(known_ensemble):
    from ilg_toolkit import DirectCalibrationHead, TargetSpec

    region, ensemble, _ = known_ensemble()
    first = ensemble.members[0]
    for changed in (
        replace(first, calibration_head=DirectCalibrationHead(TargetSpec("another target"))),
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


def test_prediction_failure_identifies_position_without_dropping_member(
    known_ensemble, FeatureEmbedding
):
    region, ensemble, _ = known_ensemble()
    broken = replace(
        ensemble.members[0], distance_model=FeatureEmbedding(jnp.asarray(float("inf")))
    )
    requested = Ensemble([ensemble.members[0], broken, ensemble.members[1]])
    with pytest.raises(eqx.EquinoxRuntimeError, match="ensemble member 1") as error:
        requested.predict(region)
    assert error.value.__notes__ == ["Raised by ensemble member 1"]
    assert len(requested.members) == 3


def test_members_are_averaged_on_the_original_scale_not_through_their_surfaces(PointConductance):
    from ilg_toolkit import DirectCalibrationHead, LandscapeGeneticsModel, RegionBatch, TargetSpec

    region = RegionBatch(
        "graph",
        cx.field(
            np.ones((1, 4, 1)), "row", "column", cx.LabeledAxis("feature", np.array(["cover"]))
        ),
        ("a", "b", "c", "d"),
        np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
    )
    calibration = DirectCalibrationHead(TargetSpec("divergence"))
    distance_models = [PointConductance(jnp.log(jnp.asarray(level))) for level in (4.0, 1.0)]
    models = [LandscapeGeneticsModel(model, calibration, ("cover",)) for model in distance_models]
    result = Ensemble(models).predict(region)
    # Member surfaces come from each member's own distance model.
    surfaces = [model.conductance_surface(region).data for model in distance_models]
    averaged_surface_model = LandscapeGeneticsModel(
        PointConductance(jnp.log(jnp.mean(jnp.stack(surfaces)[:, 0, 0]))), calibration, ("cover",)
    )
    np.testing.assert_allclose(result.mean.as_matrix()[0, 1], 0.625, atol=1e-7)
    np.testing.assert_allclose(
        averaged_surface_model.predict(region).as_matrix()[0, 1], 0.4, atol=1e-7
    )


def test_feature_meanings_are_validated_once_for_the_ensemble(known_ensemble):
    region, ensemble, _ = known_ensemble()
    meaning = cx.LabeledAxis("feature", np.array(["different meaning"]))
    changed = replace(region, features=cx.field(region.feature_array, "row", "column", meaning))
    with pytest.raises(ValueError, match="feature") as error:
        ensemble.predict(changed)
    assert not hasattr(error.value, "__notes__")


def test_shared_vertices_warn_once_per_ensemble_prediction(known_ensemble):
    region, ensemble, _ = known_ensemble()
    shared = replace(region, grid_positions=np.array([[0, 0], [0, 0], [0, 2], [0, 3]]))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ensemble.predict(shared, [("a", "c")])
    messages = [str(warning.message) for warning in record]
    assert len(messages) == 1
    assert messages[0].endswith("region 'alpine': (a, b)")


def test_members_with_different_patch_sizes_align_genetic_predictions():
    from ilg_toolkit import DirectCalibrationHead, LandscapeGeneticsModel, RegionBatch, TargetSpec
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
        DirectCalibrationHead(TargetSpec("divergence")),
        ("cover",),
    )
    # A coarser patch size changes only the surface shape; genetic predictions still align.
    coarse = replace(first, distance_model=LogLinearConductance(1, patch_size=2))
    result = Ensemble([first, coarse, first]).predict(region, [("east", "west")])
    assert len(result.members) == 3
    np.testing.assert_array_equal(result.members[0].values, result.members[2].values)
    assert np.isfinite(result.mean.values).all()


def test_prediction_requires_float64(known_ensemble, float64_required):
    region, ensemble, _ = known_ensemble()
    with float64_required():
        ensemble.predict(region)
