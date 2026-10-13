"""Joint fitting controls verified through public fit and returned state."""

from collections.abc import Callable
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_multiregion import StochasticEmbedding, regions_and_observations

from ilg_toolkit import TrainingConfig, fit


def test_sparse_evaluation_preserves_every_update_and_resumes_between_evaluations():
    regions, observed = regions_and_observations()
    observations = {r.name: {"species": o} for r, o in zip(regions, observed, strict=True)}
    config = TrainingConfig(
        objective="direct_log1p", steps=5, evaluation_interval=3, region_batch_size=1, seed=7
    )
    callbacks = []
    result = fit(
        regions,
        observations,
        distance_model=StochasticEmbedding(jnp.asarray(1.0)),
        config=config,
        on_step=callbacks.append,
    )
    assert [record.step for record in result.history] == [0, 3, 5]
    assert [state.step for state in callbacks] == [1, 2, 3, 4, 5]
    assert [state.selected_step for state in callbacks] == [0, 0, 3, 3, 5]
    assert [record.step for record in callbacks[1].history] == [0]
    resumed = fit(regions, observations, state=callbacks[1], config=config)
    assert eqx.tree_equal(resumed, result)
    frequent = fit(
        list(reversed(regions)),
        dict(reversed(tuple(observations.items()))),
        distance_model=StochasticEmbedding(jnp.asarray(1.0)),
        config=replace(config, evaluation_interval=1),
    )
    assert eqx.tree_equal(result.distance_model, frequent.distance_model)
    assert eqx.tree_equal(result.optimizer_state, frequent.optimizer_state)
    np.testing.assert_array_equal(result.rng_key, frequent.rng_key)


@pytest.mark.parametrize(
    "name", ["region_batch_size", "species_batch_size", "species_chunk_size", "evaluation_interval"]
)
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_training_controls_fail_before_fitting(name, value):
    with pytest.raises(ValueError, match=name):
        TrainingConfig(objective="direct_log1p", **{name: value})


def test_region_first_sampling_has_unbiased_species_coverage_weights():
    from test_species_workflow import PooledScalarDistance, observations, region

    from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits

    valley = region(species=("fox", "lynx"))
    ridge = RegionBatch(
        "ridge",
        valley.features,
        {"fox": SamplingUnits(("x", "y", "z"), [[0, 0], [0, 1], [0, 2]])},
        feature_names=valley.feature_names,
    )
    observed = {
        "valley": {
            "fox": observations((0.01, 0.01, 0.01)),
            "lynx": observations((50.0, 50.0, 50.0)),
        },
        "ridge": {
            "fox": PairwiseObservations.from_pairs(
                [("x", "z")], [0.01], target=observations().target
            )
        },
    }

    # Independent analytic group derivatives at scalar weight=1. Full species
    # coverage coefficients are ridge-fox=1/4, valley-fox=1/4, valley-lynx=1/2.
    def derivative(distances, target):
        d = np.asarray(distances)
        return np.mean(4 * d / (1 + d) * (np.log1p(d) - np.log1p(target)))

    group = np.array(
        [
            derivative([9.0], 0.01),
            derivative([1.0, 9.0, 4.0], 0.01),
            derivative([1.0, 9.0, 4.0], 50.0),
        ]
    )
    # One of two regions, then one eligible species: p=(1/2,1/4,1/4).
    probabilities = np.array([0.5, 0.25, 0.25])
    full_weights = np.array([0.25, 0.25, 0.5])
    estimators = group * full_weights / probabilities
    counts = np.zeros(3)
    observed_estimators = np.zeros(3)
    for seed in range(64):
        state = fit(
            [valley, ridge],
            observed,
            distance_model=PooledScalarDistance(jnp.asarray(1.0)),
            config=TrainingConfig(
                objective="direct_log1p",
                steps=1,
                region_batch_size=1,
                species_batch_size=1,
                seed=seed,
            ),
        )
        assert isinstance(state.optimizer_state, tuple)
        adam = state.optimizer_state[0]
        assert isinstance(adam, optax.ScaleByAdamState)
        assert isinstance(adam.mu, PooledScalarDistance)
        actual = float(adam.mu.weight) / 0.1
        index = np.argmin(abs(estimators - actual))
        np.testing.assert_allclose(actual, estimators[index], rtol=1e-10)
        counts[index] += 1
        observed_estimators[index] = actual
    assert np.all(counts > 0)
    np.testing.assert_allclose(counts / counts.sum(), probabilities, atol=0.17, rtol=0)
    np.testing.assert_allclose(
        probabilities @ observed_estimators, full_weights @ group, rtol=1e-10
    )


def test_unet_reuses_region_encoding_and_chunk_updates_match_with_dropout():
    import coordax as cx
    import jax

    from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits, TargetSpec
    from ilg_toolkit.models import UNetEmbeddingDistance

    calls = []

    def record(inference):
        calls.append(bool(inference))

    class ObservedBackbone(eqx.Module):
        encoder: Callable[..., jax.Array]

        def __call__(self, features, *, inference=True, key=None, patch_batch_size=None):
            # Runtime observation at the public fit seam, including compiled
            # forward/backward work; no replacement numerical computation.
            jax.debug.callback(record, inference)
            return self.encoder(
                features, inference=inference, key=key, patch_batch_size=patch_batch_size
            )

    species = ("fox", "lynx", "owl")
    prepared = RegionBatch(
        "forest",
        cx.field(
            jax.random.normal(jax.random.key(8), (4, 4, 2), dtype=jnp.float32),
            "row",
            "column",
            "feature",
        ),
        {
            name: SamplingUnits(("a", "b", "c", "d"), [[0, 0], [0, 3], [3, 0], [3, 3]])
            for name in species
        },
    )
    target = TargetSpec("divergence")
    pairs = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
    observed = {
        "forest": {
            name: PairwiseObservations.from_pairs(
                pairs, np.arange(1.0, 7.0) / (i + 1), target=target
            )
            for i, name in enumerate(species)
        }
    }
    model = UNetEmbeddingDistance(
        2,
        species=species,
        patch_size=1,
        base_channels=2,
        embedding_dim=2,
        dropout=0.2,
        key=jax.random.key(9),
    )
    model = eqx.tree_at(lambda m: m.backbone, model, ObservedBackbone(model.backbone))
    states = []
    for chunk in (1, 2, 3):
        calls.clear()
        states.append(
            fit(
                [prepared],
                observed,
                distance_model=model,
                config=TrainingConfig(
                    objective="direct_log1p", steps=1, species_chunk_size=chunk, seed=13
                ),
            )
        )
        assert calls.count(True) == 2  # step-zero and final full evaluation
        assert 1 <= calls.count(False) <= 2  # forward plus allowed checkpoint recomputation
    for state in states[1:]:
        for actual, expected in zip(
            jax.tree.leaves((state.distance_model, state.optimizer_state)),
            jax.tree.leaves((states[0].distance_model, states[0].optimizer_state)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-7)
        np.testing.assert_allclose(
            state.history[-1].training_loss, states[0].history[-1].training_loss, rtol=3e-6
        )


@pytest.mark.parametrize("family", ["unet", "resnet"])
@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_neural_chunking_preserves_all_gradients_and_complete_ragged_calibrations(
    family, objective
):
    import coordax as cx
    import jax

    from ilg_toolkit import PairwiseObservations, RegionBatch, SamplingUnits, TargetSpec
    from ilg_toolkit.models import ResNet9Conductance, UNetEmbeddingDistance

    species = ("fox", "lynx", "owl")
    size = 8 if family == "resnet" else 4
    prepared = RegionBatch(
        "forest",
        cx.field(
            jax.random.normal(jax.random.key(101), (size, size, 2), dtype=jnp.float32),
            "row",
            "column",
            "feature",
        ),
        {
            name: SamplingUnits(
                ("a", "b", "c", "d"), [[0, 0], [0, size - 1], [size - 1, 0], [size - 1, size - 1]]
            )
            for name in species
        },
    )
    pairs = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
    target = TargetSpec("divergence")
    observed = {
        "forest": {
            name: PairwiseObservations.from_pairs(
                pairs[:count],
                (np.arange(count) * 0.13 + np.array([0.2, 0.7, 0.1, 0.9, 0.4, 0.3])[:count])
                * (i + 1),
                target=target,
            )
            for i, (name, count) in enumerate(zip(species, (6, 6, 4), strict=True))
        }
    }
    model = (
        ResNet9Conductance(
            2, species=species, patch_size=4, patch_batch_size=2, key=jax.random.key(2)
        )
        if family == "resnet"
        else UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            dropout=0.1,
            key=jax.random.key(2),
        )
    )
    states = [
        fit(
            [prepared],
            observed,
            distance_model=model,
            config=TrainingConfig(
                objective=objective,
                steps=1,
                species_chunk_size=chunk,
                mlpe_initial_variances=(0.2, 0.4),
                seed=13,
            ),
        )
        for chunk in (1, 2)
    ]
    for actual, expected in zip(
        jax.tree.leaves(
            (states[1].distance_model, states[1].raw_variances, states[1].optimizer_state)
        ),
        jax.tree.leaves(
            (states[0].distance_model, states[0].raw_variances, states[0].optimizer_state)
        ),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)
    # Parameters near a zero derivative amplify float32 reduction differences
    # through Adam's normalization; moments directly check gradient magnitudes.
    for actual, expected in zip(
        jax.tree.leaves(states[1].optimizer_state),
        jax.tree.leaves(states[0].optimizer_state),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-8)
    np.testing.assert_allclose(
        [record.training_loss for record in states[1].history],
        [record.training_loss for record in states[0].history],
        rtol=2e-5,
    )
    assert set(states[1].history[-1].training_by_group["forest"]) == set(species)
    if objective == "mlpe":
        from ilg_toolkit.calibration import UnpooledMLPECalibrationModule

        calibrations = [state.model.calibration for state in states]
        assert all(
            isinstance(calibration, UnpooledMLPECalibrationModule) for calibration in calibrations
        )
        for name in species:
            first, second = (
                calibration.for_group("forest", name)
                for calibration in calibrations
                if isinstance(calibration, UnpooledMLPECalibrationModule)
            )
            np.testing.assert_allclose(
                [first.intercept, first.slope, first.unit_variance, first.residual_variance],
                [second.intercept, second.slope, second.unit_variance, second.residual_variance],
                rtol=2e-5,
                atol=1e-7,
            )
            np.testing.assert_allclose(first.effect_mean, second.effect_mean, rtol=2e-5, atol=1e-7)
            np.testing.assert_allclose(
                first.effect_covariance, second.effect_covariance, rtol=2e-5, atol=1e-7
            )


def test_continuation_checks_declared_unobserved_sampling_units():
    from test_species_workflow import PooledScalarDistance, observations, region

    from ilg_toolkit import RegionBatch, SamplingUnits

    valley = region(species=("fox", "lynx"))
    ridge = RegionBatch(
        "ridge",
        valley.features,
        {
            "fox": SamplingUnits(("x", "y", "z"), [[0, 0], [0, 1], [0, 2]]),
            "lynx": SamplingUnits(("u", "v", "w"), [[0, 0], [0, 1], [0, 2]]),
        },
        feature_names=valley.feature_names,
    )
    observed = {
        "valley": {"fox": observations(), "lynx": observations()},
        "ridge": {"fox": replace(observations(), sampling_unit_ids=("x", "y", "z"))},
    }
    initial = fit(
        [valley, ridge],
        observed,
        distance_model=PooledScalarDistance(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    altered = replace(
        ridge,
        sampling_units={
            **ridge.sampling_units,
            "lynx": SamplingUnits(("u", "v", "w"), [[0, 1], [0, 0], [0, 2]]),
        },
    )
    with pytest.raises(ValueError, match="contracts changed"):
        fit([valley, altered], observed, state=initial, config=replace(initial.config, steps=1))


def test_sparse_mlpe_selection_keeps_complete_matching_calibrations_and_rng():
    from test_mlpe_training import WeightedEmbedding
    from test_species_mlpe import irregular_problem, oracle_group

    from ilg_toolkit import ObservationPartition, PairwiseObservations
    from ilg_toolkit.calibration import UnpooledMLPECalibrationModule

    regions, observed, groups = irregular_problem()
    valley = regions[0]
    original = observed["valley"]["fox"]
    held = original.observed_pairs[-1:]
    train_observed = PairwiseObservations.from_pairs(
        original.observed_pairs[:-1], original.observed_values[:-1], target=original.target
    )
    partitions = {
        "alpine": {"fox": None, "lynx": None},
        "valley": {"fox": ObservationPartition("valley", "fox", original.observed_pairs[:-1])},
    }
    validation_partition = {
        "valley": {"fox": ObservationPartition("valley", "fox", held, role="validation")}
    }
    config = TrainingConfig(
        objective="mlpe",
        steps=4,
        evaluation_interval=3,
        region_batch_size=1,
        species_batch_size=1,
        seed=21,
        learning_rate=0.03,
        mlpe_initial_variances=(0.05, 0.1),
    )
    callbacks = []
    states = []
    for target in (50.0, 500.0):
        validation = (
            [valley],
            {
                "valley": {
                    "fox": PairwiseObservations.from_pairs(held, [target], target=original.target)
                }
            },
        )
        states.append(
            fit(
                regions,
                observed,
                partition=partitions,
                validation=validation,
                validation_partition=validation_partition,
                distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
                config=config,
                on_step=callbacks.append if target == 50.0 else None,
            )
        )
    result = states[0]
    assert [record.step for record in result.history] == [0, 3, 4]
    assert result.selected_step in (0, 3, 4)
    assert eqx.tree_equal(states[0].distance_model, states[1].distance_model)
    assert eqx.tree_equal(states[0].optimizer_state, states[1].optimizer_state)
    np.testing.assert_array_equal(states[0].rng_key, states[1].rng_key)
    assert isinstance(result.model.distance_model, WeightedEmbedding)
    assert isinstance(result.model.calibration, UnpooledMLPECalibrationModule)
    assert result.selected_raw_variances is not None
    for index, (prepared, species, observations, _) in enumerate(groups):
        if index == 2:
            observations = train_observed
        _, variances, (_, beta, mean, covariance) = oracle_group(
            result.model.distance_model.log_weights,
            result.selected_raw_variances[index],
            prepared,
            species,
            observations,
        )
        head = result.model.calibration.for_group(prepared.name, species)
        np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            [head.unit_variance, head.residual_variance], variances, rtol=1e-12
        )
        np.testing.assert_allclose(head.effect_mean, mean, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(head.effect_covariance, covariance, rtol=1e-10, atol=1e-12)
    validation = (
        [valley],
        {"valley": {"fox": PairwiseObservations.from_pairs(held, [50.0], target=original.target)}},
    )
    resumed = fit(
        regions,
        observed,
        partition=partitions,
        validation=validation,
        validation_partition=validation_partition,
        state=callbacks[1],
        config=config,
    )
    assert eqx.tree_equal(resumed, result)
    assert callbacks[1].step == 2 and callbacks[1].selected_step == 0
    assert [record.step for record in callbacks[1].history] == [0]
    assert result.history[-1].validation_missing_species == ("lynx",)
    assert result.history[-1].validation_missing_groups == (("alpine", "fox"), ("alpine", "lynx"))


@pytest.mark.parametrize("species_batch_size", [1, 2])
def test_dynamic_species_sampling_reuses_compiled_neural_tail(caplog, species_batch_size):
    import jax
    from test_species_workflow import observations, region

    from ilg_toolkit.models import UNetEmbeddingDistance

    species = ("fox", "lynx", "owl", "deer")
    prepared = region(species=species)
    observed = {prepared.name: {name: observations() for name in species}}
    model = UNetEmbeddingDistance(
        1, species=species, patch_size=1, base_channels=2, embedding_dim=2, key=jax.random.key(912)
    )
    config = TrainingConfig(
        objective="direct_log1p",
        steps=1,
        species_batch_size=species_batch_size,
        evaluation_interval=3,
        seed=51,
    )
    with jax.log_compiles(True):
        fit([prepared], observed, distance_model=model, config=config)
    compiled = [
        record.getMessage() for record in caplog.records if "Compiling jit(" in record.getMessage()
    ]
    assert compiled
    caplog.clear()
    snapshots = []
    with jax.log_compiles(True):
        state = fit(
            [prepared],
            observed,
            distance_model=model,
            config=replace(config, steps=12, seed=72),
            on_step=snapshots.append,
        )
    assert not [record for record in caplog.records if "Compiling jit(" in record.getMessage()]
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, UNetEmbeddingDistance)
    assert np.count_nonzero(np.linalg.norm(adam.mu.projection_weight, axis=(1, 2))) > 1

    reordered = fit(
        [prepared],
        {prepared.name: dict(reversed(tuple(observed[prepared.name].items())))},
        distance_model=model,
        config=replace(config, steps=12, seed=72),
    )
    assert eqx.tree_equal(reordered, state)
    resumed = fit([prepared], observed, state=snapshots[4], config=state.config)
    assert snapshots[4].step == 5 and snapshots[4].selected_step == 3
    assert eqx.tree_equal(resumed, state)


@pytest.mark.parametrize("family", ["unet", "loglinear"])
def test_pooled_mlpe_validation_uses_matching_training_moments_for_one_pair(family):
    import coordax as cx
    import jax
    from test_mlpe_training import dense_profile

    from ilg_toolkit import (
        ObservationPartition,
        PairwiseObservations,
        RegionBatch,
        SamplingUnits,
        TargetSpec,
    )
    from ilg_toolkit.models import LogLinearConductance, UNetEmbeddingDistance

    species = ("fox", "lynx")
    prepared = RegionBatch(
        "forest",
        cx.field(
            jax.random.normal(jax.random.key(29), (4, 4, 2), dtype=jnp.float32),
            "row",
            "column",
            "feature",
        ),
        {
            name: SamplingUnits(("a", "b", "c", "d"), [[0, 0], [0, 3], [3, 0], [3, 3]])
            for name in species
        },
    )
    pairs = (("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d"))
    target = TargetSpec("divergence")
    targets = np.array([0.2, 0.7, 0.1, 0.9, 0.4, 0.3])
    observed = {
        "forest": {
            name: PairwiseObservations.from_pairs(pairs, targets * (index + 1), target=target)
            for index, name in enumerate(species)
        }
    }
    if family == "unet":
        model = UNetEmbeddingDistance(
            2,
            species=species,
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            dropout=0.1,
            key=jax.random.key(71),
        )
    else:
        model = LogLinearConductance(2, species=species, patch_size=1)
        model = eqx.tree_at(lambda m: m.coefficients, model, jnp.asarray([[0.3, -0.1], [0.1, 0.2]]))
    config = TrainingConfig(
        objective="mlpe",
        steps=2,
        evaluation_interval=2,
        species_batch_size=1,
        species_chunk_size=2,
        seed=4,
        mlpe_initial_variances=(0.1, 0.2),
    )
    partition = {
        "forest": {"fox": None, "lynx": ObservationPartition("forest", "lynx", pairs[:-1])}
    }
    validation_partition = {
        "forest": {"lynx": ObservationPartition("forest", "lynx", pairs[-1:], role="validation")}
    }
    states = [
        fit(
            [prepared],
            observed,
            distance_model=model,
            config=config,
            partition=partition,
            validation=(
                [prepared],
                {
                    "forest": {
                        "lynx": PairwiseObservations.from_pairs(pairs[-1:], [value], target=target)
                    }
                },
            ),
            validation_partition=validation_partition,
        )
        for value in (50.0, 500.0)
    ]
    first, second = states
    assert eqx.tree_equal(first.distance_model, second.distance_model)
    assert eqx.tree_equal(first.optimizer_state, second.optimizer_state)
    assert first.raw_variances is not None
    train = np.asarray(
        first.distance_model.predict_raw_distances(
            prepared, species="lynx", pairs={"lynx": pairs[:-1]}
        )["lynx"].values,
        dtype=np.float64,
    )
    query = float(
        first.distance_model.predict_raw_distances(
            prepared, species="lynx", pairs={"lynx": pairs[-1:]}
        )["lynx"].values[0]
    )
    left, right = np.triu_indices(4, 1)
    variances = np.logaddexp(0.0, np.asarray(first.raw_variances[1])) + config.mlpe_variance_floor
    _, beta, _, _ = dense_profile(train, 2 * targets[:-1], left[:-1], right[:-1], 4, variances)
    mean = beta[0] + beta[1] * (query - train.mean()) / train.std(ddof=1)
    variance = 2 * variances[0] + variances[1]
    expected = 0.5 * (np.log(2 * np.pi * variance) + (50.0 - mean) ** 2 / variance)
    assert first.history[-1].validation_loss is not None
    np.testing.assert_allclose(first.history[-1].validation_loss, expected, rtol=2e-6)
    assert [record.step for record in first.history] == [0, 2]


def test_pooled_direct_training_keeps_float32_without_requesting_float64():
    import warnings

    import jax
    from test_species_workflow import observations, region

    from ilg_toolkit.models import UNetEmbeddingDistance

    with jax.enable_x64(False), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        prepared = region()
        model = UNetEmbeddingDistance(
            1,
            species=("fox",),
            patch_size=1,
            base_channels=2,
            embedding_dim=2,
            key=jax.random.key(142),
        )
        result = fit(
            [prepared],
            {prepared.name: {"fox": observations()}},
            distance_model=model,
            config=TrainingConfig(objective="direct_log1p", steps=1),
        )
        assert result.step == 1
        assert np.isfinite(result.history[-1].training_loss)
