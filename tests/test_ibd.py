"""Public geographic reference behavior, independent of landscape encoders."""

import jax
import numpy as np
import pytest

from ilg_toolkit import IBDModel, TargetSpec


def test_fit_recovers_shared_affine_geographic_relationship():
    distances = {
        "alpine": np.array([0.0, 0.5, 2.0, 4.0]),
        "coastal": np.array([1.0, 3.0, 7.0]),
    }
    targets = {name: 0.7 * values + 0.2 for name, values in distances.items()}
    target = TargetSpec("synthetic genetic divergence", units="index")

    model = IBDModel.fit(distances, targets, distance_units="km", target=target)

    assert model.slope == pytest.approx(0.7, abs=1e-6)
    assert model.intercept == pytest.approx(0.2, abs=1e-6)
    assert model.distance_units == "km"
    assert model.target == target
    prediction = model.predict(np.array([0.0, 2.0, 5.0]), distance_units="km")
    assert isinstance(prediction, jax.Array)
    np.testing.assert_allclose(prediction, [0.2, 1.6, 3.7], atol=1e-6)


def test_regions_have_equal_weight_despite_unequal_pair_counts():
    # At each distance the regional targets plus one differ by a factor of four.
    # Their geometric mean is 2 * distance + 4, so log1p-MSE has the independently
    # derived optimum prediction 2 * distance + 3 and loss log(2)**2.
    short = np.array([0.0, 1.0, 3.0, 6.0])
    distances = {"large": np.tile(short, 50), "small": short}
    targets = {"large": distances["large"] + 1.0, "small": 4.0 * short + 7.0}
    model = IBDModel.fit(
        distances,
        targets,
        distance_units="m",
        target=TargetSpec("synthetic divergence", units="index"),
    )

    assert model.slope == pytest.approx(2.0, abs=1e-6)
    assert model.intercept == pytest.approx(3.0, abs=1e-6)
    regional_losses = [
        np.linalg.norm(
            np.log1p(model.predict(values, distance_units="m")) - np.log1p(targets[name])
        )
        ** 2
        / len(values)
        for name, values in distances.items()
    ]
    assert np.mean(regional_losses) == pytest.approx(np.log(2.0) ** 2, abs=1e-12)


def test_matrix_prediction_preserves_structural_zeros_and_uses_euclidean_distance():
    # Explicit raster-to-metre conversion: row and column pixel sizes differ.
    grid_positions = np.array([[0, 0], [3, 4]])
    projected_positions = grid_positions * np.array([2.0, 1.0])
    distance = np.linalg.norm(projected_positions[1] - projected_positions[0])
    assert distance == pytest.approx(np.sqrt(52.0))
    distances = np.array([[0.0, distance], [distance, 0.0]])
    original = distances.copy()
    model = IBDModel(0.5, 0.3, "m", TargetSpec("divergence", units="index"))

    prediction = model.predict(distances, distance_units="m")

    assert isinstance(prediction, jax.Array)
    np.testing.assert_allclose(prediction, [[0, 3.905551275463989], [3.905551275463989, 0]])
    np.testing.assert_array_equal(distances, original)
    # A zero distance between distinct endpoints retains the affine intercept.
    np.testing.assert_allclose(model.predict([0.0], distance_units="m"), [0.3])


@pytest.mark.parametrize(
    ("distances", "targets", "message"),
    [
        ({}, {}, "nonempty.*regions"),
        ({"a": [1.0]}, {"b": [1.0]}, "matching.*regions"),
        ({"": [1.0]}, {"": [1.0]}, "region.*nonempty"),
        ({"a": []}, {"a": []}, "a.*nonempty.*vectors"),
        ({"a": [1.0, 2.0]}, {"a": [1.0]}, "a.*aligned"),
        ({"a": [[1.0, 2.0]]}, {"a": [[1.0, 2.0]]}, "a.*vectors"),
        ({"a": [np.nan]}, {"a": [1.0]}, "a.*finite.*nonnegative"),
        ({"a": [1.0]}, {"a": [np.inf]}, "a.*finite.*nonnegative"),
        ({"a": [-1.0]}, {"a": [1.0]}, "a.*finite.*nonnegative"),
        ({"a": [1.0]}, {"a": [-1.0]}, "a.*finite.*nonnegative"),
        ({"a": [1j]}, {"a": [1.0]}, "a.*real numeric"),
        ({"a": ["1.0"]}, {"a": [1.0]}, "a.*real numeric"),
    ],
)
def test_fit_rejects_missing_misaligned_or_invalid_regions(distances, targets, message):
    with pytest.raises(ValueError, match=message):
        IBDModel.fit(
            distances,
            targets,
            distance_units="m",
            target=TargetSpec("divergence", units="index"),
        )


@pytest.mark.parametrize(
    "distances",
    [[], -1.0, [-1.0], [np.nan], [np.inf], [[0.0, 1.0]], np.zeros((2, 2, 2))],
)
def test_prediction_rejects_invalid_distances_or_shapes(distances):
    model = IBDModel(1.0, 0.2, "m", TargetSpec("divergence", units="index"))
    with pytest.raises(ValueError, match="finite|nonnegative|nonempty|vector|square"):
        model.predict(distances, distance_units="m")


@pytest.mark.parametrize("coefficient", [-1.0, np.inf, np.nan, True])
def test_constructed_models_require_finite_nonnegative_coefficients(coefficient):
    with pytest.raises(ValueError, match="coefficients.*finite.*nonnegative"):
        IBDModel(coefficient, 0.0, "m", TargetSpec("divergence", units="index"))
    with pytest.raises(ValueError, match="coefficients.*finite.*nonnegative"):
        IBDModel(0.0, coefficient, "m", TargetSpec("divergence", units="index"))


def test_distance_units_and_original_target_meaning_are_explicit():
    target = TargetSpec("divergence", units="index")
    for units in ("", " "):
        with pytest.raises(ValueError, match="distance_units"):
            IBDModel.fit({"a": [1]}, {"a": [1]}, distance_units=units, target=target)
    for incompatible in (
        TargetSpec("relatedness", units="index", kind="relatedness"),
        TargetSpec("divergence", units="index", transform="sqrt"),
    ):
        with pytest.raises(ValueError, match="original-scale dissimilarity"):
            IBDModel.fit({"a": [1]}, {"a": [1]}, distance_units="m", target=incompatible)
    model = IBDModel(0.5, 0.2, "m", target)
    with pytest.raises(ValueError, match="distance_units.*match"):
        model.predict([1.0], distance_units="km")


def test_nonnegative_bounds_allow_the_optimum_on_the_constraint_boundary():
    model = IBDModel.fit(
        {"a": [1.0, 2.0, 3.0]},
        {"a": [9.0, 3.0, 0.0]},
        distance_units="m",
        target=TargetSpec("divergence", units="index"),
    )
    assert model.slope == pytest.approx(0.0, abs=1e-10)
    assert model.intercept == pytest.approx(40.0 ** (1 / 3) - 1.0, abs=1e-7)


def test_iteration_budget_failure_is_reported_and_not_returned_as_a_fitted_model():
    with pytest.raises(RuntimeError, match="IBD.*did not converge.*[Ii][Tt][Ee][Rr]"):
        IBDModel.fit(
            {"a": [0.0, 0.5, 2.0, 4.0]},
            {"a": [0.2, 0.55, 1.6, 3.0]},
            distance_units="m",
            target=TargetSpec("divergence", units="index"),
            max_iterations=1,
        )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_iteration_budget_must_be_a_positive_integer(budget):
    with pytest.raises(ValueError, match="max_iterations.*positive integer"):
        IBDModel.fit(
            {"a": [1]},
            {"a": [1]},
            distance_units="m",
            target=TargetSpec("divergence", units="index"),
            max_iterations=budget,
        )


def test_explicit_distance_unit_conversion_changes_slope_and_preserves_predictions():
    kilometres = np.array([0.0, 0.5, 2.0, 4.0])
    targets = {"a": np.array([0.2, 0.55, 1.6, 3.0])}
    target = TargetSpec("divergence", units="index")
    km_model = IBDModel.fit({"a": kilometres}, targets, distance_units="km", target=target)
    m_model = IBDModel.fit({"a": kilometres * 1000}, targets, distance_units="m", target=target)

    assert m_model.slope == pytest.approx(km_model.slope / 1000, rel=1e-6)
    assert m_model.intercept == pytest.approx(km_model.intercept, abs=1e-7)
    np.testing.assert_allclose(
        km_model.predict(kilometres, distance_units="km"),
        m_model.predict(kilometres * 1000, distance_units="m"),
        atol=1e-7,
    )


def test_zero_targets_admit_zero_coefficients():
    model = IBDModel.fit(
        {"a": [0.0, 2.0, 5.0]},
        {"a": [0.0, 0.0, 0.0]},
        distance_units="m",
        target=TargetSpec("divergence", units="index"),
    )
    assert model.slope == model.intercept == 0.0
    np.testing.assert_array_equal(model.predict([0.0, 2.0], distance_units="m"), [0.0, 0.0])


def test_prediction_does_not_silently_truncate_fitted_float64_or_change_jax_precision():
    model = IBDModel(0.7, 0.2, "m", TargetSpec("divergence", units="index"))
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="float64.*x64"):
            model.predict([1.00000000001], distance_units="m")
        assert not jax.config.read("jax_enable_x64")


def test_unrepresentable_predictions_raise_a_numerical_error():
    model = IBDModel(2.0, 0.0, "m", TargetSpec("divergence", units="index"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        model.predict([np.finfo(np.float64).max], distance_units="m")
