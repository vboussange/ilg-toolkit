"""Public fitting/prediction seam accepted in the specification's Test Strategy."""

import jax
import numpy as np

from ilg_toolkit import FitConfig, PairwiseObservations, PreparedRegion, TargetSpec, fit
from ilg_toolkit.models import UNetEmbeddingDistance


def synthetic_problem():
    features = np.arange(32, dtype=np.float32).reshape(4, 4, 2) / 32
    region = PreparedRegion(
        name="arbitrary-catchment",
        features=features,
        sampling_unit_ids=("north", "east", "south", "west"),
        grid_positions=np.array([[0, 0], [0, 3], [3, 0], [3, 3]]),
    )
    # Independent simple dissimilarity with known units; no paper assets.
    targets = np.array(
        [[0, 0.2, 0.4, 0.6], [0.2, 0, 0.2, 0.4], [0.4, 0.2, 0, 0.2], [0.6, 0.4, 0.2, 0]]
    )
    observations = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, targets, target=TargetSpec("synthetic divergence", units="index")
    )
    model = UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=jax.random.key(3)
    )
    return region, observations, model


def test_fixed_budget_fit_improves_and_predicts_without_query_targets():
    region, observations, model = synthetic_problem()
    result = fit(
        region, observations, model=model, config=FitConfig(epochs=20, learning_rate=0.01, seed=3)
    )
    prediction = result.predictor.predict(region)
    assert result.history[-1].training_loss < result.history[0].training_loss * 0.75
    assert result.selected_epoch == 20
    assert result.selection == "final"
    assert prediction.target.units == "index"
    assert prediction.target.transform == "identity"
    assert prediction.values.shape == (4, 4)
    assert np.all(np.isfinite(prediction.values))
    np.testing.assert_allclose(prediction.values, prediction.values.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(prediction.values), 0, atol=1e-6)
    np.testing.assert_allclose(result.predictor.landscape_scores(region), prediction.values)


def test_validation_selects_predictor_without_refitting_on_validation_targets():
    region, observations, model = synthetic_problem()
    # This deliberately prefers the initial predictor while training targets differ.
    initial = np.asarray(model.predict_distances(region.features, region.pixel_nodes))
    validation = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, initial, target=observations.target
    )
    from dataclasses import replace

    validation_region = replace(region, name="independent-validation-region")
    result = fit(
        region,
        observations,
        model=model,
        config=FitConfig(epochs=3, learning_rate=0.01),
        validation=(validation_region, validation),
    )
    assert result.selection == "validation"
    assert result.selected_epoch == 0
    assert result.history[-1].validation_loss > result.history[0].validation_loss
    np.testing.assert_allclose(result.predictor.predict(region).values, initial, atol=1e-5)


def test_observation_labels_are_aligned_and_signed_relatedness_is_refused():
    import pytest

    region, observations, model = synthetic_problem()
    reverse = tuple(reversed(region.sampling_unit_ids))
    reordered = PairwiseObservations.from_matrix(
        reverse, observations.values[::-1, ::-1], target=observations.target
    )
    first = fit(region, observations, model=model, config=FitConfig(epochs=0))
    second = fit(region, reordered, model=model, config=FitConfig(epochs=0))
    assert first.history[0].training_loss == second.history[0].training_loss
    signed = PairwiseObservations.from_matrix(
        region.sampling_unit_ids,
        -observations.values,
        target=TargetSpec("relatedness", kind="relatedness"),
    )
    with pytest.raises(ValueError, match="nonnegative dissimilarities"):
        fit(region, signed, model=model, config=FitConfig(epochs=0))
