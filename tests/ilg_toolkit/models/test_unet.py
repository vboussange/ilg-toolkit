"""Scientific behavior of models.unet, covered once at tiny width."""

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    ArtifactError,
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    load_model,
    save_model,
    train,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def _unet(
    seed: int,
    *,
    num_species: int = 1,
    patch_size: int = 1,
    embedding_dim: int = 2,
    dropout: float = 0.1,
    patch_microbatch_size: int | None = None,
) -> UNetEmbeddingDistance:
    return UNetEmbeddingDistance(
        2,
        num_species=num_species,
        patch_size=patch_size,
        base_channels=2,
        embedding_dim=embedding_dim,
        dropout=dropout,
        patch_microbatch_size=patch_microbatch_size,
        key=jax.random.key(seed),
    )


def _gradient_vector(gradient):
    leaves = jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array))
    return np.concatenate([np.asarray(leaf, dtype=np.float64).ravel() for leaf in leaves])


def test_patch_microbatches_preserve_predictions_and_gradients():
    # Six patches in batches of four exercise a final batch smaller than four.
    features = np.random.default_rng(8).normal(size=(4, 6, 2)).astype(np.float32)
    region = RegionBatch(
        "valley",
        cx.field(features, "row", "column", "feature"),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 5], [3, 5]]),
    )
    reference = _unet(19, patch_size=2)
    microbatched = _unet(19, patch_size=2, patch_microbatch_size=4)

    @eqx.filter_jit
    @eqx.filter_value_and_grad(has_aux=True)
    def objective(model):
        predictions = model.predict(region).values
        return jnp.sum(predictions**2), predictions

    (expected_value, expected), expected_gradient = objective(reference)
    (actual_value, actual), actual_gradient = objective(microbatched)
    assert expected.shape == (3,)
    assert np.all(np.isfinite(expected) & (expected > 0))
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5)
    expected_gradient = _gradient_vector(expected_gradient)
    actual_gradient = _gradient_vector(actual_gradient)
    assert np.all(np.isfinite(expected_gradient)) and np.any(expected_gradient != 0)
    # One scale for all leaves: parameters with vanishing gradients carry only roundoff.
    scale = np.max(np.abs(expected_gradient))
    np.testing.assert_allclose(actual_gradient / scale, expected_gradient / scale, atol=1e-5)


def test_species_fit_improves_predicts_without_query_targets_and_reloads(tmp_path):
    features = np.arange(18, dtype=np.float32).reshape(3, 3, 2) / 18
    targets = np.array(
        [[0, 0.2, 0.4, 0.6], [0.2, 0, 0.2, 0.4], [0.4, 0.2, 0, 0.2], [0.6, 0.4, 0.2, 0]]
    )
    ids = ("north", "east", "south", "west")
    region = RegionBatch(
        "arbitrary-catchment",
        cx.field(features, "row", "column", "feature"),
        ids * 2,
        np.tile([[0, 0], [0, 2], [2, 0], [2, 2]], (2, 1)),
        feature_names=("gradient", "texture"),
        species=("frog",) * 4 + ("newt",) * 4,
    )
    target = TargetSpec("synthetic divergence", units="index")
    observations = tuple(
        PairwiseObservations.from_matrix(ids, targets * scale, target=target, species=name)
        for scale, name in ((1, "frog"), (2, "newt"))
    )
    order = ("frog", "newt")
    result = train(
        region,
        observations,
        distance_model=_unet(3, num_species=2, dropout=0),
        species=order,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, learning_rate=0.01, seed=3),
    )
    assert result.history[-1].training_loss < result.history[0].training_loss * 0.75
    fitted = result.selected_model
    for name in order:
        prediction = fitted.predict(region, species=name)
        matrix = prediction.as_matrix()
        assert matrix.shape == (4, 4)
        assert np.all(np.isfinite(prediction.values))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        assert np.isnan(np.diag(matrix)).all()
        np.testing.assert_allclose(
            fitted.distance_model.predict(region, species=name, species_order=order).as_matrix(),
            matrix,
        )

    path = tmp_path / "unet.eqx"
    save_model(path, fitted)
    restored = load_model(path, distance_model=_unet(7, num_species=2, dropout=0))
    assert isinstance(restored.distance_model, UNetEmbeddingDistance)
    for name in order:
        np.testing.assert_array_equal(
            restored.predict(region, species=name).values,
            fitted.predict(region, species=name).values,
        )
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=_unet(7, num_species=2, dropout=0, embedding_dim=3))
