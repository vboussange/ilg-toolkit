"""Experimental diagnostic seam; independent Gaussian and SciPy references."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import helmert
from scipy.stats import matrix_normal, wishart

from ilg_toolkit import PairwiseObservations, PreparedRegion, TargetSpec
from ilg_toolkit.experimental import (
    GaussianMarkerDistances,
    diagnose_wishart,
    heldout_log_likelihood,
    wishart_log_likelihood,
)
from ilg_toolkit.models import ConductanceModel


def gaussian_example(populations=4, markers=11):
    rng = np.random.default_rng(21)
    population_covariance = 0.6 ** np.abs(
        np.arange(populations)[:, None] - np.arange(populations)[None, :]
    )
    raw = np.linalg.cholesky(population_covariance) @ rng.normal(size=(populations, markers))
    distance = np.mean((raw[:, None, :] - raw[None, :, :]) ** 2, axis=-1)
    labels = tuple(f"population-{i}" for i in range(populations))
    return labels, raw, distance, population_covariance


def test_declared_marker_average_and_scatter_match_normalized_reference_density():
    labels, raw, distances, covariance = gaussian_example()
    basis = helmert(len(labels))
    scatter = (basis @ raw) @ (basis @ raw).T
    model = basis @ covariance @ basis.T
    common = dict(
        region_name="synthetic",
        sampling_unit_ids=labels,
        marker_count=raw.shape[1],
        interpretation="squared_gaussian_marker_distance",
    )
    average = GaussianMarkerDistances(distances=distances, marker_scaling="average", **common)
    summed = GaussianMarkerDistances(
        distances=distances * raw.shape[1], marker_scaling="scatter", **common
    )
    np.testing.assert_allclose(average.centered_scatter, scatter, atol=1e-12)
    expected = wishart.logpdf(scatter, df=raw.shape[1], scale=model)
    assert np.isclose(wishart_log_likelihood(summed, model), expected)
    matrix_dimension = len(labels) - 1
    independent_entries = matrix_dimension * (matrix_dimension + 1) // 2
    assert np.isclose(
        wishart_log_likelihood(average, model),
        expected + independent_entries * np.log(raw.shape[1]),
    )


def test_population_holdout_matches_independent_gaussian_regression_decomposition():
    labels, raw, distances, population_covariance = gaussian_example(populations=5)
    markers = raw.shape[1]
    observations = GaussianMarkerDistances(
        "synthetic", labels, distances, markers, "average", "squared_gaussian_marker_distance"
    )
    # Use a non-first anchor and a different training order to exercise identities.
    anchor = 2
    training = (labels[2], labels[4], labels[0])
    order = [4, 0, 1, 3]
    contrast = np.eye(5)[order] - np.eye(5)[anchor]
    marker_contrasts = raw[order] - raw[anchor]
    scatter = marker_contrasts @ marker_contrasts.T
    sigma = contrast @ population_covariance @ contrast.T
    train_dimension = len(training) - 1
    train_scatter = scatter[:train_dimension, :train_dimension]
    cross_scatter = scatter[train_dimension:, :train_dimension]
    regression = sigma[train_dimension:, :train_dimension] @ np.linalg.inv(
        sigma[:train_dimension, :train_dimension]
    )
    conditional_covariance = (
        sigma[train_dimension:, train_dimension:]
        - regression @ sigma[:train_dimension, train_dimension:]
    )
    residual_scatter = scatter[
        train_dimension:, train_dimension:
    ] - cross_scatter @ np.linalg.solve(train_scatter, cross_scatter.T)
    expected = matrix_normal.logpdf(
        cross_scatter,
        mean=regression @ train_scatter,
        rowcov=conditional_covariance,
        colcov=train_scatter,
    ) + wishart.logpdf(residual_scatter, df=markers - train_dimension, scale=conditional_covariance)
    full_dimension = len(labels) - 1
    additional_entries = (
        full_dimension * (full_dimension + 1) - train_dimension * (train_dimension + 1)
    ) // 2
    expected += additional_entries * np.log(markers)
    basis = helmert(len(labels))
    actual = heldout_log_likelihood(
        observations,
        basis @ population_covariance @ basis.T,
        training_unit_ids=training,
        anchor_id=labels[anchor],
        parameter_source="fixed",
    )
    assert np.isclose(actual.log_likelihood, expected, atol=1e-10)
    assert actual.heldout_unit_ids == (labels[1], labels[3])
    assert actual.training_unit_ids == training
    assert actual.parameter_source == "fixed"
    assert actual.coordinate_measure == "anchored_marker_average_covariance_entries"

    # Only held-out marker values change; the training marginal stays fixed.
    changed_raw = raw.copy()
    changed_raw[[1, 3]] *= 1.5
    changed_distances = np.mean((changed_raw[:, None] - changed_raw[None, :]) ** 2, axis=-1)
    changed = GaussianMarkerDistances(
        "synthetic",
        labels,
        changed_distances,
        markers,
        "average",
        "squared_gaussian_marker_distance",
    )
    rescored = heldout_log_likelihood(
        changed,
        basis @ population_covariance @ basis.T,
        training_unit_ids=training,
        anchor_id=labels[anchor],
        parameter_source="training_only",
    )
    assert rescored.training_log_likelihood == actual.training_log_likelihood
    assert not np.isclose(rescored.log_likelihood, actual.log_likelihood)


class UniformConductance(ConductanceModel):
    scale: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def conductance(self, features, *, patch_batch_size=None):
        return self.scale * features[..., 0]


def test_actual_graph_diagnostic_connects_covariance_and_retains_no_go_gate():
    labels = ("west", "east", "north", "south")
    region = PreparedRegion(
        "valley", np.ones((2, 2, 1)), labels, np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    )
    resistance = np.array(
        [[0, 0.75, 0.75, 1], [0.75, 0, 1, 0.75], [0.75, 1, 0, 0.75], [1, 0.75, 0.75, 0]]
    )
    basis = helmert(4)
    covariance = -0.5 * 1.3 * basis @ resistance @ basis.T + 0.2 * np.eye(3)
    raw = basis.T @ np.linalg.cholesky(covariance) @ np.random.default_rng(5).normal(size=(3, 13))
    distances = np.mean((raw[:, None] - raw[None, :]) ** 2, axis=-1)
    observations = GaussianMarkerDistances(
        "valley", labels, distances, 13, "average", "squared_gaussian_marker_distance"
    )
    with jax.enable_x64():
        report = diagnose_wishart(
            region,
            observations,
            encoder=UniformConductance(jnp.array(1.0)),
            scale=1.3,
            nugget=0.2,
            training_unit_ids=labels[:2],
            anchor_id=labels[0],
            parameter_source="fixed",
        )
    np.testing.assert_allclose(report.resistance_scores, resistance, atol=1e-6)
    np.testing.assert_allclose(report.model_covariance, covariance, atol=1e-6)
    assert np.isclose(report.log_likelihood, wishart_log_likelihood(observations, covariance))
    assert np.isfinite(report.heldout_score.log_likelihood)
    assert report.scale_nugget_rank == 2
    assert report.training_gate == "no_go"
    assert any("normalization" in reason for reason in report.gate_reasons)


@pytest.mark.parametrize("interpretation", ["FST", "relatedness", "dissimilarity"])
def test_unsupported_genetic_interpretations_fail_explicitly(interpretation):
    labels, raw, distances, _ = gaussian_example()
    with pytest.raises(ValueError, match="unsupported"):
        GaussianMarkerDistances(
            "synthetic", labels, distances, raw.shape[1], "average", interpretation
        )


@pytest.mark.parametrize("marker_scaling", [None, "", "unknown"])
def test_unspecified_marker_scaling_is_rejected(marker_scaling):
    labels, raw, distances, _ = gaussian_example()
    with pytest.raises(ValueError, match="marker_scaling"):
        GaussianMarkerDistances(
            "synthetic",
            labels,
            distances,
            raw.shape[1],
            marker_scaling,
            "squared_gaussian_marker_distance",
        )


def test_missing_indefinite_and_singular_inputs_are_rejected_without_repair():
    labels, raw, distances, _ = gaussian_example()
    missing = distances.copy()
    missing[0, 1] = missing[1, 0] = np.nan
    indefinite = np.ones((4, 4)) - np.eye(4)
    indefinite[0, 1] = indefinite[1, 0] = 100
    points = np.arange(4.0)
    singular = (points[:, None] - points[None, :]) ** 2
    for invalid, message in (
        (missing, "complete"),
        (indefinite, "positive definite"),
        (singular, "positive definite"),
    ):
        with pytest.raises(ValueError, match=message):
            GaussianMarkerDistances(
                "synthetic",
                labels,
                invalid,
                raw.shape[1],
                "average",
                "squared_gaussian_marker_distance",
            )
    for marker_count in (None, True, 2, 11.5):
        with pytest.raises(ValueError, match="marker_count"):
            GaussianMarkerDistances(
                "synthetic",
                labels,
                distances,
                marker_count,
                "average",
                "squared_gaussian_marker_distance",
            )


def test_holdout_rejects_unknown_anchors_empty_partitions_and_target_fitted_parameters():
    labels, raw, distances, covariance = gaussian_example()
    data = GaussianMarkerDistances(
        "synthetic", labels, distances, raw.shape[1], "average", "squared_gaussian_marker_distance"
    )
    basis = helmert(len(labels))
    model = basis @ covariance @ basis.T
    for training, anchor in (
        (labels, labels[0]),
        (labels[:1], labels[0]),
        (labels[:2], labels[-1]),
        (("unknown", labels[0]), labels[0]),
    ):
        with pytest.raises(ValueError, match="Holdout"):
            heldout_log_likelihood(
                data, model, training_unit_ids=training, anchor_id=anchor, parameter_source="fixed"
            )
    with pytest.raises(ValueError, match="training_only"):
        heldout_log_likelihood(
            data,
            model,
            training_unit_ids=labels[:2],
            anchor_id=labels[0],
            parameter_source="full_data",
        )


def test_generic_genetic_targets_are_not_silently_promoted_to_marker_observations():
    labels, _, distances, _ = gaussian_example()
    generic = PairwiseObservations.from_matrix(labels, distances, target=TargetSpec("FST"))
    with pytest.raises(TypeError, match="GaussianMarkerDistances"):
        wishart_log_likelihood(generic, np.eye(len(labels) - 1))


def test_scientific_nugget_constraints_and_fixed_encoder_confounding_are_visible():
    labels, raw, distances, _ = gaussian_example(populations=3)
    observations = GaussianMarkerDistances(
        "coincident", labels, distances, raw.shape[1], "average", "squared_gaussian_marker_distance"
    )
    region = PreparedRegion("coincident", np.ones((2, 2, 1)), labels, np.zeros((3, 2), dtype=int))
    options = dict(
        encoder=UniformConductance(jnp.array(1.0)),
        training_unit_ids=labels[:2],
        anchor_id=labels[0],
        parameter_source="fixed",
    )
    with jax.enable_x64():
        report = diagnose_wishart(region, observations, scale=1, nugget=0.2, **options)
        assert report.scale_nugget_rank == 1
        assert np.isinf(report.scale_nugget_condition_number)
        np.testing.assert_allclose(report.model_covariance, 0.2 * np.eye(2), atol=1e-10)
        with pytest.raises(ValueError, match="Model covariance.*positive definite"):
            diagnose_wishart(region, observations, scale=1, nugget=0, **options)
    with pytest.raises(ValueError, match="scale.*positive"):
        diagnose_wishart(region, observations, scale=0, nugget=0.2, **options)
    with pytest.raises(ValueError, match="nugget.*nonnegative"):
        diagnose_wishart(region, observations, scale=1, nugget=-0.2, **options)
