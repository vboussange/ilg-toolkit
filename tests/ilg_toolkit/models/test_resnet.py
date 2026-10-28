"""Scientific behavior of models.resnet, covered once at tiny width."""

from dataclasses import replace

import coordax as cx
import equinox as eqx
import jax
import numpy as np
import pytest
from ilg_toolkit import (
    load_model,
    Log1pLoss,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import ResNet9Conductance


_TINY = (8, 8, 8)
# ResNet9Conductance(3) parameter count, recorded before channel widths became configurable.
_DEFAULT_PARAMETER_COUNT = 741_761


def _region(shape=(8, 12, 3)):
    features = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    return RegionBatch(
        "small-valley",
        cx.field(features, "row", "column", "feature"),
        ("south", "north", "east"),
        np.array([[0, 0], [0, shape[1] - 1], [shape[0] - 1, shape[1] - 1]]),
        feature_names=tuple(f"feature-{index}" for index in range(shape[2])),
    )


def test_default_widths_keep_the_paper_architecture_parameter_count_and_float32_weights():
    # Shapes only: the default-width network is neither initialized nor compiled.
    model = eqx.filter_eval_shape(ResNet9Conductance, 3, key=jax.random.key(0))
    leaves = [leaf for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.ShapeDtypeStruct)]
    assert model.channels == (64, 128, 256)
    assert sum(leaf.size for leaf in leaves) == _DEFAULT_PARAMETER_COUNT
    # Float64 is enabled, and the patch encoder layers still hold float32 weights.
    assert jax.enable_x64.value
    assert {leaf.dtype for leaf in leaves} == {np.dtype("float32")}


def test_patch_microbatches_preserve_surfaces_predictions_and_gradients(
    assert_patch_microbatches_match,
):
    # Six patches in batches of four exercise a final batch smaller than four.
    key = jax.random.key(19)
    surface = assert_patch_microbatches_match(
        ResNet9Conductance(3, patch_size=4, channels=_TINY, key=key),
        ResNet9Conductance(3, patch_size=4, channels=_TINY, patch_microbatch_size=4, key=key),
        _region(),
        rtol=1e-6,
    )
    assert surface is not None and surface.shape == (2, 3)


def test_species_fit_matches_dense_resistance_and_reloads(dense_resistance, tmp_path):
    region = _region((8, 8, 2))
    species = ("frog", "newt")
    region = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=np.tile(region.grid_positions, (2, 1)),
        sampling_unit_kinds=None,
        species=("frog",) * 3 + ("newt",) * 3,
    )
    nodes = np.array([0, 1, 3])
    target = TargetSpec("synthetic dissimilarity", units="index")
    observations = tuple(
        PairwiseObservations(
            ("south", "north", "east"),
            dense_resistance(np.full((2, 2), 2.0 * scale), nodes),
            target=target,
            species=name,
        )
        for scale, name in ((1, "frog"), (0.5, "newt"))
    )

    def model(seed, solver_config=None):
        return ResNet9Conductance(
            2,
            num_species=2,
            patch_size=4,
            channels=_TINY,
            min_conductance=0.02,
            patch_microbatch_size=3,
            solver_config=solver_config,
            key=jax.random.key(seed),
        )

    result = train(
        region,
        observations,
        distance_model=model(6, ResistanceSolverConfig(rtol=1e-9, atol=1e-10)),
        species=species,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, learning_rate=0.01),
    )
    assert np.isfinite(result.history[-1].training_loss)
    assert result.history[-1].training_loss < result.history[0].training_loss
    fitted = result.selected_model
    distance_model = fitted.distance_model
    assert isinstance(distance_model, ResNet9Conductance)
    surfaces = distance_model.conductance_surface(region, species=species, species_order=species)
    assert not np.allclose(surfaces["frog"], surfaces["newt"])
    for name in species:
        surface = np.asarray(surfaces[name])
        assert surface.shape == (2, 2)
        assert np.all(np.isfinite(surface) & (surface > 0))
        expected = dense_resistance(surface, nodes)
        np.fill_diagonal(expected, np.nan)
        prediction = fitted.predict(region, species=name).as_matrix()
        np.testing.assert_allclose(prediction, expected, rtol=2e-6)
        np.testing.assert_allclose(
            distance_model.predict(region, species=name, species_order=species).as_matrix(),
            prediction,
        )

    path = tmp_path / "resnet.eqx"
    save_model(path, fitted)
    restored = load_model(path, distance_model=model(9))
    restored_model = restored.distance_model
    assert isinstance(restored_model, ResNet9Conductance)
    restored_surfaces = restored_model.conductance_surface(
        region, species=species, species_order=species
    )
    for name in species:
        np.testing.assert_array_equal(restored_surfaces[name], surfaces[name])
        np.testing.assert_array_equal(
            restored.predict(region, species=name).values,
            fitted.predict(region, species=name).values,
        )
    assert restored_model.solver_config == distance_model.solver_config
    assert restored_model.solver_config != ResistanceSolverConfig()
    assert restored_model.channels == _TINY
    assert restored_model.patch_microbatch_size == 3
    assert restored_model.min_conductance == 0.02


@pytest.mark.parametrize("channels", [(8, 8), (8, 12, 16), (0, 8, 8), (8, 8, -8), (8.0, 8, 8)])
def test_widths_must_be_three_positive_multiples_of_eight(channels):
    with pytest.raises(ValueError):
        ResNet9Conductance(2, channels=channels, key=jax.random.key(0))
