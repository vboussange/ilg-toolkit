"""Native species workflows and independent scientific objective references."""

from dataclasses import replace
from typing import Any, cast, final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import AbstractDistanceModel


@final
class SpeciesDistance(AbstractDistanceModel):
    weights: jax.Array
    species: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, species=("fox",), weights=None):
        self.species = tuple(species)
        self.weights = jnp.ones(len(species)) if weights is None else jnp.asarray(weights)

    def _pair_distances(
        self,
        features,
        selection,
        species_index,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        points = jnp.asarray(features).reshape(-1)[selection.nodes]
        return (
            self.weights[species_index] ** 2
            * (points[selection.left] - points[selection.right]) ** 2
        )


def region(name="valley", species=("fox",), values=(0.0, 1.0, 3.0)):
    return RegionBatch(
        name,
        cx.field(jnp.asarray(values).reshape(1, -1, 1), "row", "column", "feature"),
        {name: SamplingUnits(("a", "b", "c"), [[0, 0], [0, 1], [0, 2]]) for name in species},
        feature_names=("elevation",),
    )


def observations(targets=(4.0, 36.0, 16.0)):
    return PairwiseObservations.from_pairs(
        [("a", "b"), ("a", "c"), ("b", "c")],
        targets,
        target=TargetSpec("divergence", units="index"),
    )


def test_one_species_native_custom_model_fits_and_returns_keyed_pair_predictions():
    prepared = region()
    state = fit(
        [prepared],
        {"valley": {"fox": observations()}},
        distance_model=SpeciesDistance(),
        config=TrainingConfig(objective="direct_log1p", steps=2, learning_rate=0.1),
    )
    assert state.step == state.selected_step == 2
    assert state.history[-1].training_loss < state.history[0].training_loss
    requested = {"fox": (("c", "a"), ("b", "c"))}
    raw = state.model.predict_raw_distances(prepared, pairs=requested)
    predicted = state.model.predict(prepared, pairs=requested)
    assert tuple(raw) == tuple(predicted) == ("fox",)
    assert raw["fox"].pairs == requested["fox"]
    assert raw["fox"].species_name == predicted["fox"].species_name == "fox"
    assert raw["fox"].region_name == "valley"
    np.testing.assert_allclose(raw["fox"].values, predicted["fox"].values)
    matrix = raw["fox"].as_matrix()
    assert np.isnan(matrix[0, 1])
    np.testing.assert_array_equal(np.diag(matrix), 0)
    full = state.model.ecological_distances(prepared)
    np.testing.assert_allclose(full["fox"][2, 0], raw["fox"].values[0])


@final
class PooledScalarDistance(AbstractDistanceModel):
    weight: jax.Array
    species: tuple[str, ...] = eqx.field(static=True, default=("fox", "lynx"))

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def _pair_distances(
        self,
        features,
        selection,
        species_index,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        points = jnp.asarray(features).reshape(-1)[selection.nodes]
        return self.weight**2 * (points[selection.left] - points[selection.right]) ** 2


def test_irregular_species_coverage_matches_independent_balanced_objective_and_update():
    first = region(species=("fox", "lynx"))
    second = RegionBatch(
        "ridge",
        first.features,
        {"fox": SamplingUnits(("x", "y", "z"), [[0, 0], [0, 1], [0, 2]])},
        feature_names=first.feature_names,
    )
    sparse = PairwiseObservations.from_pairs([("x", "z")], [0.01], target=observations().target)
    observed = {
        "valley": {
            "fox": observations((0.01, 0.01, 0.01)),
            "lynx": observations((50.0, 50.0, 50.0)),
        },
        "ridge": {"fox": sparse},
    }

    def independent(weight):
        distances = weight**2 * np.array([1.0, 9.0, 4.0])
        fox_valley = np.mean((np.log1p(distances) - np.log1p(0.01)) ** 2)
        fox_ridge = (np.log1p(weight**2 * 9.0) - np.log1p(0.01)) ** 2
        lynx = np.mean((np.log1p(distances) - np.log1p(50.0)) ** 2)
        return 0.5 * ((fox_valley + fox_ridge) / 2 + lynx)

    delta = 1e-5
    derivative = (independent(1 + delta) - independent(1 - delta)) / (2 * delta)
    expected = 1 - 0.01 * derivative / (abs(derivative) + 1e-8)
    state = fit(
        [second, first],
        observed,
        distance_model=PooledScalarDistance(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01),
    )
    np.testing.assert_allclose(state.history[0].training_loss, independent(1.0), rtol=1e-12)
    assert isinstance(state.latest_distance_model, PooledScalarDistance)
    np.testing.assert_allclose(state.latest_distance_model.weight, expected, rtol=1e-10)
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, PooledScalarDistance) and isinstance(adam.nu, PooledScalarDistance)
    np.testing.assert_allclose(adam.mu.weight, 0.1 * derivative, rtol=1e-6)
    np.testing.assert_allclose(adam.nu.weight, 0.001 * derivative**2, rtol=1e-6)
    assert set(state.history[0].training_by_species) == {"fox", "lynx"}
    assert set(state.history[0].training_by_group["ridge"]) == {"fox"}


def test_native_unet_learns_species_specific_predictions_with_irregular_coverage():
    from ilg_toolkit.models import UNetEmbeddingDistance

    raster = cx.field(
        jax.random.normal(jax.random.key(41), (4, 4, 2), dtype=jnp.float32),
        "row",
        "column",
        "feature",
    )
    prepared = RegionBatch(
        "forest",
        raster,
        {
            "fox": SamplingUnits(("a", "b", "c"), [[0, 0], [0, 3], [3, 3]]),
            "lynx": SamplingUnits(("a", "b", "c"), [[0, 3], [3, 0], [3, 3]]),
        },
        feature_names=("height", "canopy"),
    )
    other = RegionBatch(
        "ridge",
        raster,
        {"fox": SamplingUnits(("x", "y", "z"), [[0, 0], [0, 3], [3, 0]])},
        feature_names=prepared.feature_names,
    )
    target = observations().target
    observed = {
        "forest": {"fox": observations(), "lynx": observations((0.1, 0.3, 0.7))},
        "ridge": {
            "fox": PairwiseObservations.from_pairs(
                [("x", "y"), ("x", "z")], [1.0, 0.4], target=target
            )
        },
    }
    model = UNetEmbeddingDistance(
        2,
        species=("fox", "lynx"),
        patch_size=1,
        base_channels=2,
        embedding_dim=2,
        dropout=0.1,
        key=jax.random.key(7),
    )
    initial = model.predict_raw_distances(prepared, species=("fox", "lynx"))
    fitted = fit(
        [prepared, other],
        observed,
        distance_model=model,
        config=TrainingConfig(objective="direct_log1p", steps=2),
    )
    final = fitted.model.predict_raw_distances(prepared, species=("fox", "lynx"))
    assert not np.allclose(final["fox"].values, initial["fox"].values)
    assert not np.allclose(final["fox"].values, final["lynx"].values)
    assert set(fitted.model.predict(other, species="fox")) == {"fox"}


@pytest.mark.parametrize("selected", [None, (), ("fox", "fox"), "wolf"])
def test_multispecies_selection_rejects_ambiguous_or_invalid_names(selected):
    model = SpeciesDistance(("fox", "lynx"))
    with pytest.raises(ValueError, match="species"):
        model.predict_raw_distances(region(species=model.species), species=selected)


def test_selection_preserves_declared_order_and_requires_regional_units():
    model = SpeciesDistance(("fox", "lynx"), weights=(1.0, 2.0))
    prepared = region(species=model.species)
    result = model.predict_raw_distances(prepared, species=("lynx", "fox"))
    assert tuple(result) == ("lynx", "fox")
    np.testing.assert_allclose(result["lynx"].values, 4 * result["fox"].values)
    with pytest.raises(ValueError, match="sampling units"):
        model.predict_raw_distances(region(), species="lynx")
    with pytest.raises(ValueError, match="explicit species"):
        model.predict_raw_distances(region())


@pytest.mark.parametrize(
    "pairs",
    [
        {},
        {"lynx": (("a", "b"),)},
        {"fox": ()},
        {"fox": (("a", "a"),)},
        {"fox": (("a", "z"),)},
        {"fox": (("a", "b"), ("b", "a"))},
        {"fox": ("ab",)},
    ],
)
def test_native_pair_requests_reject_invalid_species_or_endpoint_identities(pairs):
    with pytest.raises(ValueError):
        SpeciesDistance().predict_raw_distances(region(), pairs=pairs)


@pytest.mark.parametrize("species", [(), ("fox", "fox"), ("",)])
def test_native_custom_registry_requires_unique_nonempty_names(species):
    with pytest.raises(ValueError, match="registry"):
        SpeciesDistance(species)


def test_native_group_contract_rejects_flat_forms_missing_coverage_and_empty_regions():
    prepared = region(species=("fox", "lynx"))
    model = SpeciesDistance(("fox", "lynx"))
    config = TrainingConfig(objective="direct_log1p", steps=0)
    with pytest.raises(ValueError, match="collection"):
        fit(
            cast(Any, prepared),
            {"valley": {"fox": observations()}},
            distance_model=model,
            config=config,
        )
    with pytest.raises(ValueError, match="Observed species"):
        fit([prepared], cast(Any, {"valley": observations()}), distance_model=model, config=config)
    with pytest.raises(ValueError, match="training coverage"):
        fit([prepared], {"valley": {"fox": observations()}}, distance_model=model, config=config)
    with pytest.raises(ValueError, match="observed species group"):
        fit([prepared], {"valley": {}}, distance_model=model, config=config)
    with pytest.raises(ValueError, match="registered sampling units"):
        fit([prepared], {"valley": {"wolf": observations()}}, distance_model=model, config=config)


def test_declared_assignments_reject_duplicate_species_units_and_allow_geographic_overlap():
    first = region()
    duplicate = replace(first, name="ridge")
    observed = {"valley": {"fox": observations()}, "ridge": {"fox": observations()}}
    config = TrainingConfig(objective="direct_log1p", steps=0)
    with pytest.raises(ValueError, match="one consistent region assignment"):
        fit([first, duplicate], observed, distance_model=SpeciesDistance(), config=config)
    distinct = replace(
        duplicate, sampling_units={"fox": replace(first.sampling_units["fox"], ids=("x", "y", "z"))}
    )
    observed["ridge"]["fox"] = PairwiseObservations.from_matrix(
        ("x", "y", "z"), observations().values, target=observations().target
    )
    fitted = fit([first, distinct], observed, distance_model=SpeciesDistance(), config=config)
    assert fitted.region_names == ("ridge", "valley")


def test_nested_partitions_report_missing_validation_coverage_without_target_leakage():
    prepared = region(species=("fox", "lynx"))
    observed = {"valley": {"fox": observations(), "lynx": observations()}}
    train = {"valley": {"fox": ObservationPartition("valley", "fox", (("a", "b"),)), "lynx": None}}
    validate = {
        "valley": {"fox": ObservationPartition("valley", "fox", (("b", "c"),), role="validation")}
    }
    state = fit(
        [prepared],
        observed,
        partition=train,
        validation=([prepared], {"valley": {"fox": observations()}}),
        validation_partition=validate,
        distance_model=SpeciesDistance(("fox", "lynx")),
        config=TrainingConfig(objective="direct_log1p", steps=1),
    )
    record = state.history[-1]
    assert record.validation_missing_groups == (("valley", "lynx"),)
    assert record.validation_missing_species == ("lynx",)
    assert set(record.validation_by_species) == {"fox"}
    assert state.model.training_pairs["valley"]["fox"] == (("a", "b"),)
    assert state.model.validation_pairs["valley"]["fox"] == (("b", "c"),)
    assert record.validation_loss is not None
    np.testing.assert_allclose(record.validation_loss, record.validation_by_group["valley"]["fox"])
    with pytest.raises(TypeError):
        cast(Any, state.model.predict)(prepared, species="fox", observations=observed)
    for invalid in (
        replace(train["valley"]["fox"], species_name="lynx"),
        replace(train["valley"]["fox"], region_name="ridge"),
        replace(train["valley"]["fox"], role="validation"),
    ):
        with pytest.raises(ValueError):
            fit(
                [prepared],
                observed,
                partition={"valley": {"fox": invalid, "lynx": None}},
                distance_model=SpeciesDistance(("fox", "lynx")),
                config=TrainingConfig(objective="direct_log1p", steps=0),
            )


def test_continuation_rejects_changed_numerical_inputs_and_reduced_declared_budget():
    prepared = region()
    observed = {"valley": {"fox": observations()}}
    config = TrainingConfig(objective="direct_log1p", steps=3)
    snapshots = []
    completed = fit(
        [prepared],
        observed,
        distance_model=SpeciesDistance(),
        config=config,
        on_step=snapshots.append,
    )
    assert [snapshot.step for snapshot in snapshots] == [1, 2, 3]
    with pytest.raises(ValueError, match="step budget"):
        fit([prepared], observed, state=snapshots[0], config=replace(config, steps=2))
    for changed_region, changed_observed in (
        (region(values=(0.0, 1.0, 4.0)), observed),
        (prepared, {"valley": {"fox": observations((3.0, 36.0, 16.0))}}),
    ):
        with pytest.raises(ValueError, match="contracts changed"):
            fit(
                [changed_region], changed_observed, state=completed, config=replace(config, steps=4)
            )
    calls = []
    same = fit([prepared], observed, state=completed, on_step=calls.append)
    assert same.step == completed.step and not calls


@pytest.mark.parametrize(
    "registry", [{"fox", "lynx"}, {"fox": 0, "lynx": 1}, iter(("fox", "lynx")), "fox"]
)
def test_builtin_registry_rejects_unordered_and_nonsequence_inputs(registry):
    from ilg_toolkit.models import UNetEmbeddingDistance

    with pytest.raises(ValueError, match="ordered sequence"):
        UNetEmbeddingDistance(1, species=registry, key=jax.random.key(0))


@pytest.mark.parametrize("x64", [False, True])
def test_sampling_positions_reject_overflow_before_integer_narrowing(x64):
    with jax.enable_x64(x64), pytest.raises(ValueError, match="grid_positions"):
        SamplingUnits(("a", "b"), np.array([[0, 2**32], [0, 1]], dtype=np.int64))


@pytest.mark.parametrize("jit", [False, True])
def test_numeric_failure_reports_region_species_and_step(jit):
    with pytest.raises(RuntimeError, match="valley.*fox.*step 0"):
        fit(
            [region()],
            {"valley": {"fox": observations()}},
            distance_model=SpeciesDistance(weights=(np.inf,)),
            config=TrainingConfig(objective="direct_log1p", steps=0, jit=jit),
        )
