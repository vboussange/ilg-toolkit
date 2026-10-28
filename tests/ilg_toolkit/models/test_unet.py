"""Tests of the U-Net embedding-distance model at tiny width."""

import coordax as cx
import equinox as eqx
import jax
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    ArtifactError,
    load_model,
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def _unet(
    seed: int,
    *,
    species: tuple[str, ...] | None = None,
    patch_size: int = 1,
    embedding_dim: int = 2,
    dropout: float = 0.1,
    patch_microbatch_size: int | None = None,
) -> UNetEmbeddingDistance:
    return UNetEmbeddingDistance(
        2,
        species=species,
        patch_size=patch_size,
        base_channels=2,
        embedding_dim=embedding_dim,
        dropout=dropout,
        patch_microbatch_size=patch_microbatch_size,
        key=jax.random.key(seed),
    )


def test_layers_keep_float32_weights_with_float64_enabled():
    model = _unet(0)
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_inexact_array))
    assert jax.enable_x64.value
    assert {leaf.dtype for leaf in leaves} == {np.dtype("float32")}


def test_patch_microbatches_preserve_predictions_and_gradients(assert_patch_microbatches_match):
    # Six patches in batches of four exercise a final batch smaller than four.
    features = np.random.default_rng(8).normal(size=(4, 6, 2)).astype(np.float32)
    region = RegionBatch(
        "valley",
        cx.field(
            features, "row", "column", cx.LabeledAxis("feature", np.array(["elevation", "canopy"]))
        ),
        ("a", "b", "c"),
        np.array([[0, 0], [0, 5], [3, 5]]),
    )
    assert_patch_microbatches_match(
        _unet(19, patch_size=2), _unet(19, patch_size=2, patch_microbatch_size=4), region, rtol=1e-5
    )


def test_species_training_improves_predicts_without_query_targets_and_reloads(tmp_path):
    features = np.arange(18, dtype=np.float32).reshape(3, 3, 2) / 18
    targets = np.array(
        [[0, 0.2, 0.4, 0.6], [0.2, 0, 0.2, 0.4], [0.4, 0.2, 0, 0.2], [0.6, 0.4, 0.2, 0]]
    )
    ids = ("north", "east", "south", "west")
    region = RegionBatch(
        "arbitrary-catchment",
        cx.field(
            features, "row", "column", cx.LabeledAxis("feature", np.array(("gradient", "texture")))
        ),
        ids * 2,
        np.tile([[0, 0], [0, 2], [2, 0], [2, 2]], (2, 1)),
        species=("frog",) * 4 + ("newt",) * 4,
    )
    target = TargetSpec("synthetic divergence", units="index")
    observations = tuple(
        PairwiseObservations(ids, targets * scale, target=target, species=name)
        for scale, name in ((1, "frog"), (2, "newt"))
    )
    species = ("frog", "newt")
    result = train(
        region,
        observations,
        distance_model=_unet(3, species=species, dropout=0),
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.01), seed=3),
    )
    assert result.history[-1].training_loss < result.history[0].training_loss * 0.75
    trained = result.selected_model
    for name in species:
        prediction = trained.predict(region, {name: None})[name]
        matrix = prediction.as_matrix()
        assert matrix.shape == (4, 4)
        assert np.all(np.isfinite(prediction.values))
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-6)
        assert np.isnan(np.diag(matrix)).all()
        np.testing.assert_allclose(
            trained.distance_model.predict(region, {name: None})[name].as_matrix(),
            matrix,
        )

    path = tmp_path / "unet.eqx"
    save_model(path, trained)
    restored = load_model(path, distance_model=_unet(7, species=species, dropout=0))
    assert isinstance(restored.distance_model, UNetEmbeddingDistance)
    for name in species:
        np.testing.assert_array_equal(
            restored.predict(region, {name: None})[name].values,
            trained.predict(region, {name: None})[name].values,
        )
    with pytest.raises(ArtifactError):
        load_model(path, distance_model=_unet(7, species=species, dropout=0, embedding_dim=3))


@pytest.mark.parametrize("base_channels", [3, 6, 12])
def test_widths_that_are_not_multiples_of_eight_predict(base_channels):
    model = UNetEmbeddingDistance(
        1, patch_size=1, base_channels=base_channels, embedding_dim=2, key=jax.random.key(0)
    )
    region = RegionBatch(
        "plot",
        cx.field(
            np.arange(16.0).reshape(4, 4, 1),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation"])),
        ),
        ("a", "b"),
        np.array([[0, 0], [3, 3]]),
    )
    assert np.isfinite(model.predict(region).values).all()
