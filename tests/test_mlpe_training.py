"""MLPE fitting checked through the accepted public synthetic workflow seam."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from ilg_toolkit import (
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    fit,
)
from ilg_toolkit.models import ConductanceModel, EmbeddingDistanceModel
from ilg_toolkit.training import TrainingState


@final
class WeightedEmbedding(EmbeddingDistanceModel):
    """Actual squared embedding distances with a learnable feature relationship."""

    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    log_weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, log_weights: jax.Array, *, species=("species",)):
        self.species = tuple(species)
        self.log_weights = log_weights

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return features * jnp.exp(self.log_weights)


def adam_moments(state: TrainingState) -> tuple[WeightedEmbedding, jax.Array]:
    """Expose Adam's public first moments after checking their parameter structure."""
    assert isinstance(state.optimizer_state, tuple)
    adam = state.optimizer_state[0]
    assert isinstance(adam, optax.ScaleByAdamState)
    assert isinstance(adam.mu, tuple)
    encoder, variances = adam.mu
    assert isinstance(encoder, WeightedEmbedding)
    assert isinstance(variances, jax.Array)
    return encoder, variances


def embedding_problem(name="alpine", offset=0.0, seed=14):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(2, 3, 2)).astype(np.float32)
    ids = tuple(f"unit-{i}" if name == "alpine" else f"{name}-unit-{i}" for i in range(6))
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        {"species": SamplingUnits(ids, np.array(list(np.ndindex(2, 3))))},
        feature_names=("elevation", "canopy"),
    )
    embedding = features.reshape(6, 2) * np.array([0.3, 1.5])
    matrix = ((embedding[:, None] - embedding[None, :]) ** 2).sum(axis=-1)
    left, right = np.triu_indices(6, 1)
    distances = matrix[left, right]
    effects = rng.normal(scale=0.04, size=6)
    values = (
        1.4
        + offset
        - 0.4 * (distances - distances.mean()) / distances.std(ddof=1)
        + effects[left]
        + effects[right]
        + rng.normal(scale=0.02, size=len(left))
    )
    observations = PairwiseObservations.from_pairs(
        [(ids[a], ids[b]) for a, b in zip(left, right, strict=True)],
        values,
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=ids,
    )
    return (region, observations)


def test_public_mlpe_embedding_fit_updates_encoder_and_variances_and_returns_predictions():
    with jax.enable_x64():
        region, observations = embedding_problem()
        model = WeightedEmbedding(jnp.log(jnp.array([1.0, 1.0])))
        result = fit(
            [region],
            {region.name: {"species": observations}},
            distance_model=model,
            config=TrainingConfig(
                objective="mlpe", steps=20, learning_rate=0.03, mlpe_initial_variances=(0.05, 0.1)
            ),
        )
        assert result.raw_variances is not None
        prediction = result.model.predict(region)["species"]
        distances = result.model.ecological_distances(region)["species"]
    assert result.history[-1].training_loss < result.history[0].training_loss - 0.1
    assert isinstance(result.model.distance_model, WeightedEmbedding)
    assert not np.allclose(result.model.distance_model.log_weights, model.log_weights)
    assert isinstance(result.model.calibration, UnpooledMLPECalibrationModule)
    head = result.model.calibration.groups[region.name]["species"]
    assert not np.allclose([head.unit_variance, head.residual_variance], [0.05, 0.1])
    assert head.slope < 0
    assert result.model.training_objective == "mlpe"
    assert prediction.target.units == "index"
    assert np.isfinite(prediction.values).all()
    assert not np.allclose(prediction.as_matrix(), distances)
    left, right = np.triu_indices(6, 1)
    np.testing.assert_allclose(head.distance_center, np.mean(distances[left, right]), atol=1e-12)
    np.testing.assert_allclose(
        head.distance_scale, np.std(distances[left, right], ddof=1), atol=1e-12
    )
    assert result.step == result.selected_step == 20
    assert set(head.calibration_roles) == {"training"}


def dense_profile(distances, targets, left, right, n, variances, jitter=0):
    z = np.zeros((len(distances), n))
    z[np.arange(len(distances)), left] += 1
    z[np.arange(len(distances)), right] += 1
    u, e = variances
    covariance = u * z @ z.T + (e + jitter) * np.eye(len(distances))
    design = np.column_stack(
        (np.ones(len(distances)), (distances - distances.mean()) / distances.std(ddof=1))
    )
    inverse_design = np.linalg.solve(covariance, design)
    beta = np.linalg.solve(design.T @ inverse_design, inverse_design.T @ targets)
    residual = targets - design @ beta
    nll = 0.5 * (
        len(distances) * np.log(2 * np.pi)
        + np.linalg.slogdet(covariance)[1]
        + residual @ np.linalg.solve(covariance, residual)
    )
    mean = u * z.T @ np.linalg.solve(covariance, residual)
    posterior = u * np.eye(n) - u**2 * z.T @ np.linalg.solve(covariance, z)
    return (nll, beta, mean, posterior)


@pytest.mark.parametrize("jit", [False, True])
def test_joint_adam_update_matches_dense_finite_difference_objective(jit):
    with jax.enable_x64():
        region, observations = embedding_problem()
        left, right = np.triu_indices(6, 1)
        targets = observations.values[left, right]
        initial = np.array(
            [0.0, 0.0, np.log(np.expm1(0.05 - 1e-10)), np.log(np.expm1(0.1 - 1e-10))]
        )

        def objective(parameters):
            embedding = region.feature_array.reshape(6, 2) * np.exp(parameters[:2])
            distances = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)[left, right]
            variances = np.logaddexp(0, parameters[2:]) + 1e-10
            return dense_profile(distances, targets, left, right, 6, variances)[0] / len(distances)

        step = 1e-05
        gradient = []
        for index in range(4):
            delta = np.eye(4)[index] * step
            gradient.append((objective(initial + delta) - objective(initial - delta)) / (2 * step))
        gradient = np.array(gradient)
        assert np.all(np.abs(gradient) > 0.0001)
        expected = initial - 0.02 * gradient / (np.abs(gradient) + 1e-08)
        result = fit(
            [region],
            {region.name: {"species": observations}},
            distance_model=WeightedEmbedding(jnp.zeros(2)),
            config=TrainingConfig(
                objective="mlpe",
                steps=1,
                jit=jit,
                learning_rate=0.02,
                mlpe_initial_variances=(0.05, 0.1),
            ),
        )
        assert result.raw_variances is not None
        assert isinstance(result.distance_model, WeightedEmbedding)
        actual = np.concatenate((result.distance_model.log_weights, result.raw_variances[0]))
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        moment = adam_moments(result)
        actual_moment = np.concatenate((moment[0].log_weights, moment[1][0]))
        np.testing.assert_allclose(actual_moment, 0.1 * gradient, rtol=1e-07, atol=1e-10)
        distances = np.asarray(result.model.ecological_distances(region)["species"])[left, right]
    assert isinstance(result.model.calibration, UnpooledMLPECalibrationModule)
    head = result.model.calibration.groups[region.name]["species"]
    expected_nll, beta, mean, posterior = dense_profile(
        distances, targets, left, right, 6, (head.unit_variance, head.residual_variance)
    )
    np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-12)
    np.testing.assert_allclose(head.effect_mean, mean, atol=1e-12)
    np.testing.assert_allclose(head.effect_covariance, posterior, atol=1e-12)
    np.testing.assert_allclose(-head.ml_log_likelihood, expected_nll, atol=1e-12)


def test_validation_targets_cannot_change_training_calibration_and_one_pair_is_supported():
    from ilg_toolkit import ObservationPartition

    with jax.enable_x64():
        region, observations = embedding_problem()
        query_pair = observations.observed_pairs[-1:]
        train_pairs = observations.observed_pairs[:-1]
        train_partition = ObservationPartition(region.name, "species", train_pairs, role="training")
        query_partition = ObservationPartition(
            region.name, "species", query_pair, role="validation"
        )
        altered = PairwiseObservations.from_pairs(
            observations.observed_pairs,
            observations.observed_values + 20,
            target=observations.target,
            sampling_unit_ids=region.sampling_units["species"].ids,
        )
        config = TrainingConfig(
            objective="mlpe", steps=4, learning_rate=0.03, mlpe_initial_variances=(0.05, 0.1)
        )
        runs = [
            fit(
                [region],
                {region.name: {"species": observations}},
                distance_model=WeightedEmbedding(jnp.zeros(2)),
                config=config,
                partition={region.name: {"species": train_partition}},
                validation=([region], {region.name: {"species": validation_observations}}),
                validation_partition={region.name: {"species": query_partition}},
            )
            for validation_observations in (observations, altered)
        ]
        distances = np.asarray(
            runs[0].latest_distance_model.ecological_distances(region)["species"]
        )
    first, second = runs
    assert isinstance(first.distance_model, WeightedEmbedding)
    assert isinstance(second.distance_model, WeightedEmbedding)
    assert first.raw_variances is not None and second.raw_variances is not None
    assert [r.training_loss for r in first.history] == [r.training_loss for r in second.history]
    np.testing.assert_array_equal(
        first.distance_model.log_weights, second.distance_model.log_weights
    )
    np.testing.assert_array_equal(first.raw_variances, second.raw_variances)
    assert first.history[-1].validation_loss != second.history[-1].validation_loss
    assert first.rng_key.dtype == np.uint32
    assert first.model.training_pairs[region.name]["species"] == train_partition.pairs
    assert first.model.validation_pairs[region.name]["species"] == query_partition.pairs
    (left, right), train_values = observations.aligned_pairs(region, "species", train_partition)
    train_distances = distances[left, right]
    variances = np.logaddexp(0, first.raw_variances[0]) + config.mlpe_variance_floor
    _, beta, _, _ = dense_profile(train_distances, train_values, left, right, 6, variances)
    (left, right), values = observations.aligned_pairs(region, "species", query_partition)
    expected_mean = beta[0] + beta[1] * (
        distances[left, right] - train_distances.mean()
    ) / train_distances.std(ddof=1)
    variance = 2 * variances[0] + variances[1]
    expected_nll = 0.5 * (
        np.log(2 * np.pi * variance) + (values[0] - expected_mean[0]) ** 2 / variance
    )
    assert first.history[-1].validation_loss is not None
    np.testing.assert_allclose(first.history[-1].validation_loss, expected_nll, rtol=1e-12)


def test_shared_mlpe_fit_has_one_encoder_and_separate_regional_heads():
    with jax.enable_x64():
        problems = (embedding_problem(), embedding_problem("valley", offset=3, seed=71))
        regions = tuple(problem[0] for problem in problems)
        observations = tuple(problem[1] for problem in problems)
        result = fit(
            regions,
            {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)},
            distance_model=WeightedEmbedding(jnp.zeros(2)),
            config=TrainingConfig(
                objective="mlpe", steps=15, learning_rate=0.03, mlpe_initial_variances=(0.05, 0.1)
            ),
        )
        assert isinstance(result, TrainingState)
        assert result.raw_variances is not None
        assert result.history[-1].training_loss < result.history[0].training_loss - 0.1
        for region in regions:
            assert np.isfinite(result.model.predict(region)["species"].values).all()
    assert result.raw_variances.shape == (2, 2)
    assert result.region_names == ("alpine", "valley")
    assert isinstance(result.model.calibration, UnpooledMLPECalibrationModule)
    heads = {name: groups["species"] for name, groups in result.model.calibration.groups.items()}
    assert heads["valley"].intercept - heads["alpine"].intercept > 2
    assert not np.array_equal(result.raw_variances[0], result.raw_variances[1])
    assert result.history[-1].training_loss == np.mean(
        [groups["species"] for groups in result.history[-1].training_by_group.values()]
    )


@final
class CovariateConductance(ConductanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        return jnp.exp(self.weight * features[..., 0])


def test_public_mlpe_conductance_fit_improves_with_actual_graph_solver():
    from test_multiregion import dense_graph_resistances

    with jax.enable_x64():
        features = np.array([[[0.0], [0.2], [0.8]], [[-0.3], [1.4], [-0.7]]])
        ids = tuple(f"population-{i}" for i in range(6))
        region = RegionBatch(
            "graph",
            cx.field(features, "row", "column", "feature"),
            {"species": SamplingUnits(ids, np.array(list(np.ndindex(2, 3))))},
            feature_names=("habitat",),
        )
        left, right = np.triu_indices(6, 1)
        distances = dense_graph_resistances(
            np.exp(1.2 * region.feature_array[..., 0]),
            np.arange(len(region.sampling_units["species"].ids)),
        )[left, right]
        target = 2 + 0.5 * (distances - distances.mean()) / distances.std(ddof=1)
        target += np.random.default_rng(17).normal(0, 0.01, size=len(target))
        observations = PairwiseObservations.from_pairs(
            [(ids[i], ids[j]) for i, j in zip(left, right, strict=True)],
            target,
            target=TargetSpec("graph divergence", units="index"),
            sampling_unit_ids=ids,
        )
        result = fit(
            [region],
            {region.name: {"species": observations}},
            distance_model=CovariateConductance(jnp.asarray(0.1)),
            config=TrainingConfig(
                objective="mlpe", steps=20, learning_rate=0.08, mlpe_initial_variances=(0.02, 0.05)
            ),
        )
        assert result.raw_variances is not None
        assert result.history[-1].training_loss < result.history[0].training_loss - 0.2
        assert isinstance(result.model.distance_model, CovariateConductance)
        assert float(result.model.distance_model.weight) > 0.6
        assert np.isfinite(result.model.predict(region)["species"].values).all()
        surface = result.model.conductance_surface(region)["species"]
        actual_distances = result.model.ecological_distances(region)["species"]
    np.testing.assert_allclose(
        actual_distances,
        dense_graph_resistances(surface, np.arange(len(region.sampling_units["species"].ids))),
        atol=1e-07,
    )


def test_in_memory_continuation_preserves_optimizer_rng_and_selected_heads():
    from dataclasses import replace

    with jax.enable_x64():
        region, observations = embedding_problem()
        config = TrainingConfig(objective="mlpe", steps=6, seed=31, learning_rate=0.03)
        model = WeightedEmbedding(jnp.zeros(2))
        uninterrupted = fit(
            [region], {region.name: {"species": observations}}, distance_model=model, config=config
        )
        assert uninterrupted.raw_variances is not None
        partial = fit(
            [region],
            {region.name: {"species": observations}},
            distance_model=model,
            config=replace(config, steps=2),
        )
        assert partial.raw_variances is not None
        resumed = fit(
            [region], {region.name: {"species": observations}}, state=partial, config=config
        )
        assert resumed.raw_variances is not None
    assert resumed.history == uninterrupted.history
    assert isinstance(resumed.model.calibration, UnpooledMLPECalibrationModule)
    assert isinstance(uninterrupted.model.calibration, UnpooledMLPECalibrationModule)
    assert resumed.model.calibration.groups == uninterrupted.model.calibration.groups
    assert resumed.selected_step == uninterrupted.selected_step == 6
    for expected, actual in zip(
        jax.tree.leaves(
            (
                uninterrupted.distance_model,
                uninterrupted.raw_variances,
                uninterrupted.optimizer_state,
                uninterrupted.rng_key,
            )
        ),
        jax.tree.leaves(
            (
                resumed.distance_model,
                resumed.raw_variances,
                resumed.optimizer_state,
                resumed.rng_key,
            )
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_mlpe_training_reports_precision_identifiability_and_numerical_failures():
    from dataclasses import replace

    import pytest

    region, observations = embedding_problem()
    model = WeightedEmbedding(jnp.zeros(2))
    config = TrainingConfig(objective="mlpe", steps=0)
    with jax.enable_x64(False), pytest.raises(RuntimeError, match="float64.*JAX_ENABLE_X64"):
        fit([region], {region.name: {"species": observations}}, distance_model=model, config=config)
    with jax.enable_x64():
        disjoint = PairwiseObservations.from_pairs(
            [("unit-0", "unit-1"), ("unit-2", "unit-3"), ("unit-4", "unit-5")],
            [1, 2, 3],
            target=observations.target,
            sampling_unit_ids=region.sampling_units["species"].ids,
        )
        with pytest.raises(ValueError, match="unidentifiable"):
            fit([region], {region.name: {"species": disjoint}}, distance_model=model, config=config)
        individual = replace(
            region,
            sampling_units={
                "species": replace(region.sampling_units["species"], kinds=("individual",) * 6)
            },
        )
        with pytest.raises(ValueError, match="individual"):
            fit(
                [individual],
                {individual.name: {"species": observations}},
                distance_model=model,
                config=config,
            )
        constant_region = replace(
            region,
            features=cx.field(np.ones_like(region.feature_array), "row", "column", "feature"),
        )
        with pytest.raises(
            FloatingPointError, match="alpine.*nonfinite MLPE.*nonconstant distances"
        ):
            fit(
                [constant_region],
                {constant_region.name: {"species": observations}},
                distance_model=model,
                config=config,
            )
        extreme = replace(config, mlpe_initial_variances=(1000000000000.0, 1e-09))
        with pytest.raises(FloatingPointError, match="variance conditioning"):
            fit(
                [region],
                {region.name: {"species": observations}},
                distance_model=model,
                config=extreme,
            )


def test_continuation_rejects_changed_optimization_policy():
    from dataclasses import replace

    import pytest

    with jax.enable_x64():
        region, observations = embedding_problem()
        config = TrainingConfig(objective="mlpe", steps=1)
        result = fit(
            [region],
            {region.name: {"species": observations}},
            distance_model=WeightedEmbedding(jnp.zeros(2)),
            config=config,
        )
        assert result.raw_variances is not None
        with pytest.raises(ValueError, match="policy"):
            fit(
                [region],
                {region.name: {"species": observations}},
                state=result,
                config=replace(config, learning_rate=0.1),
            )
        with pytest.raises(ValueError, match="already contains"):
            fit(
                [region],
                {region.name: {"species": observations}},
                state=result,
                distance_model=result.distance_model,
            )


def test_shared_likelihood_gradient_weights_regions_equally_with_unequal_pair_counts():
    with jax.enable_x64():
        first = embedding_problem()
        second = embedding_problem("valley", offset=3, seed=71)
        short_observations = PairwiseObservations.from_pairs(
            second[1].observed_pairs[:7],
            second[1].observed_values[:7],
            target=second[1].target,
            sampling_unit_ids=second[0].sampling_units["species"].ids,
        )
        regions = (first[0], second[0])
        observations = (first[1], short_observations)
        raw = np.log(np.expm1(np.array([0.05, 0.1]) - 1e-10))
        initial = np.concatenate((np.zeros(2), raw, raw))

        def objective(parameters):
            losses = []
            for i, (region, observed) in enumerate(zip(regions, observations, strict=True)):
                (left, right), targets = observed.aligned_pairs(region, "species")
                embedding = region.feature_array.reshape(6, 2) * np.exp(parameters[:2])
                distances = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)[left, right]
                variances = np.logaddexp(0, parameters[2 + 2 * i : 4 + 2 * i]) + 1e-10
                losses.append(
                    dense_profile(distances, targets, left, right, 6, variances)[0] / len(distances)
                )
            return np.mean(losses)

        step = 1e-05
        gradient = np.array(
            [
                (objective(initial + delta) - objective(initial - delta)) / (2 * step)
                for delta in np.eye(6) * step
            ]
        )
        result = fit(
            regions,
            {r.name: {"species": o} for r, o in zip(regions, observations, strict=True)},
            distance_model=WeightedEmbedding(jnp.zeros(2)),
            config=TrainingConfig(
                objective="mlpe", steps=1, mlpe_initial_variances=(0.05, 0.1), learning_rate=0.02
            ),
        )
        assert result.raw_variances is not None
        moment = adam_moments(result)
        actual_moment = np.concatenate((moment[0].log_weights, moment[1].reshape(-1)))
    np.testing.assert_allclose(result.history[0].training_loss, objective(initial), atol=1e-12)
    np.testing.assert_allclose(actual_moment, 0.1 * gradient, rtol=1e-07, atol=1e-10)
