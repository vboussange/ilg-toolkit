"""Scientific behavior of training."""

from collections.abc import Sequence
from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    MLPECalibrationModule,
    MLPEConfig,
    PairwiseObservations,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import ConductanceModel


def test_failed_solver_preserves_solver_failure(PointConductance):
    with jax.enable_x64():
        region = RegionBatch(
            "bad-solve",
            cx.field(np.linspace(0.1, 2, 12).reshape(3, 4, 1), "row", "column", "feature"),
            ("first", "last"),
            np.array([[0, 0], [2, 3]]),
        )
        observations = PairwiseObservations.from_matrix(
            region.sampling_unit_ids, [[0, 1], [1, 0]], target=TargetSpec("dissimilarity")
        )
        with pytest.raises(RuntimeError):
            fit(
                region,
                observations,
                distance_model=PointConductance(
                    jnp.array(0.0),
                    solver_config=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=1),
                ),
                config=TrainingConfig(
                    objective="direct_log1p",
                    steps=0,
                ),
            )


@pytest.mark.parametrize("jit", [False, True])
def test_nonfinite_training_is_reported_at_evaluation(
    ScalarEmbedding, regions_and_observations, jit
):
    regions, observations = regions_and_observations()
    completed = []
    with pytest.raises(FloatingPointError):
        fit(
            regions[0],
            observations[0],
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            config=TrainingConfig(
                objective="direct_log1p", steps=3, eval_every=2, learning_rate=1e200, jit=jit
            ),
            on_step=lambda state: completed.append(state.step),
        )
    assert completed == [0, 1]


@pytest.mark.parametrize(
    ("jit", "num_species", "region_count", "region_batch_size", "species_batch_size"),
    [
        (False, 1, 1, None, None),
        (True, 1, 1, None, None),
        (True, 1, 2, None, None),
        (True, 1, 2, 1, None),
        (True, 2, 1, None, None),
        (True, 2, 1, None, 1),
    ],
)
def test_mlpe_updates_match_dense_reference(
    WeightedEmbedding,
    dense_profile,
    embedding_problem,
    jit,
    region_count,
    num_species,
    region_batch_size,
    species_batch_size,
    monkeypatch,
):
    from dataclasses import replace

    problems = [embedding_problem(columns=2)]
    unit_count = len(problems[0][0].sampling_unit_ids)
    if region_count == 2:
        region, observed = embedding_problem("valley", offset=3, seed=71, columns=2)
        # Unequal observation counts must still give each region equal influence.
        sparse = PairwiseObservations.from_pairs(
            observed.observed_pairs[:5],
            observed.observed_values[:5],
            target=observed.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        problems.append((region, sparse))
    species = None if num_species == 1 else ("newt", "frog")
    if num_species > 1:
        region, observed = problems[0]
        problems.append((region, replace(observed, values=observed.values * 1.3)))
    group_count = len(problems)
    groups_species = [0] * region_count if num_species == 1 else [0, 1]
    regions = tuple(problem[0] for problem in problems)
    observations = tuple(problem[1] for problem in problems)
    if species is not None:
        regions = (
            replace(
                regions[0],
                sampling_unit_ids=regions[0].sampling_unit_ids * 2,
                grid_positions=np.tile(regions[0].grid_positions, (2, 1)),
                sampling_unit_kinds=None,
                species=(species[0],) * unit_count + (species[1],) * unit_count,
            ),
        )
        observations = (
            tuple(
                replace(observed, species=name)
                for name, observed in zip(species, observations, strict=True)
            ),
        )
    parameter_count = 2 * num_species
    raw = np.log(np.expm1(np.array([0.05, 0.1]) - 1e-10))
    initial = np.concatenate((np.zeros(parameter_count), np.tile(raw, group_count)))

    def objective(parameters, indices: Sequence[int] | None = None):
        losses = []
        for index in range(group_count) if indices is None else indices:
            region, observed = problems[index]
            (left, right), targets = observed.aligned_pairs(region)
            embedding = np.asarray(region.feature_array).reshape(unit_count, 2) * np.exp(
                parameters[2 * groups_species[index] : 2 * groups_species[index] + 2]
            )
            distances = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)[left, right]
            variances = (
                np.logaddexp(
                    0, parameters[parameter_count + 2 * index : parameter_count + 2 * index + 2]
                )
                + 1e-10
            )
            losses.append(
                dense_profile(distances, np.asarray(targets), left, right, unit_count, variances)[0]
                / len(distances)
            )
        return np.mean(losses)

    if species_batch_size == 1:
        selections = [(1,), (0,)]
    elif region_batch_size == 1:
        selections = [(0,), (1,)]
    else:
        selections = [range(group_count)] * 2
    if region_batch_size == 1 or species_batch_size == 1:
        choices = iter(selections)
        monkeypatch.setattr(jax.random, "choice", lambda *args, **kwargs: jnp.array(next(choices)))

    # Two Adam updates retain sensitivity to gradient magnitude, using only the
    # dense Gaussian objective and finite differences as the numerical reference.
    expected = initial.copy()
    first_moment, second_moment = np.zeros_like(initial), np.zeros_like(initial)
    for step in (1, 2):
        delta = np.eye(len(initial)) * 1e-5
        gradient = np.array(
            [
                (
                    objective(expected + offset, selections[step - 1])
                    - objective(expected - offset, selections[step - 1])
                )
                / 2e-5
                for offset in delta
            ]
        )
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        expected -= (
            0.02
            * (first_moment / (1 - 0.9**step))
            / (np.sqrt(second_moment / (1 - 0.999**step)) + 1e-8)
        )
    result = fit(
        regions,
        observations,
        distance_model=WeightedEmbedding(
            jnp.zeros(2) if num_species == 1 else jnp.zeros((2, 2)), num_species=num_species
        ),
        species=species,
        config=TrainingConfig(
            objective="mlpe",
            steps=2,
            jit=jit,
            learning_rate=0.02,
            region_batch_size=region_batch_size,
            species_batch_size=species_batch_size,
            mlpe=MLPEConfig(initial_variances=(0.05, 0.1)),
        ),
    )
    np.testing.assert_allclose(result.history[0].training_loss, objective(initial), atol=1e-12)
    np.testing.assert_allclose(result.history[-1].training_loss, objective(expected), atol=1e-8)
    assert result.history[-1].training_loss < result.history[0].training_loss
    assert result.step == result.selected_step == 2
    calibration = cast(MLPECalibrationModule, result.model.calibration)
    for index, (region, observed) in enumerate(problems):
        embedding = np.asarray(region.feature_array).reshape(unit_count, 2) * np.exp(
            expected[2 * groups_species[index] : 2 * groups_species[index] + 2]
        )
        distances = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)
        np.fill_diagonal(distances, np.nan)
        np.testing.assert_allclose(
            result.model.distance_model.predict(
                region
                if species is None
                else replace(region, species=(species[groups_species[index]],) * unit_count),
                species=None if species is None else species[groups_species[index]],
                species_order=species,
            ).as_matrix(),
            distances,
            rtol=1e-8,
            atol=1e-9,
        )
        variances = (
            np.logaddexp(0, expected[parameter_count + 2 * index : parameter_count + 2 * index + 2])
            + 1e-10
        )
        head = calibration.for_region(
            region.name, None if species is None else groups_species[index]
        )
        np.testing.assert_allclose(
            [head.unit_variance, head.residual_variance],
            variances,
            rtol=1e-8,
            atol=1e-10,
        )
        (left, right), targets = observed.aligned_pairs(region)
        nll, beta, effect_mean, effect_covariance = dense_profile(
            distances[left, right],
            np.asarray(targets),
            left,
            right,
            unit_count,
            variances,
        )
        np.testing.assert_allclose([head.intercept, head.slope], beta, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(head.effect_mean, effect_mean, atol=1e-9)
        np.testing.assert_allclose(head.effect_covariance, effect_covariance, atol=1e-10)
        np.testing.assert_allclose(-head.ml_log_likelihood, nll, atol=1e-8)
        assert head.slope < 0
        assert set(head.calibration_roles) == {"training"}
        assert not np.allclose(variances, [0.05, 0.1])
        observed_distances = distances[left, right]
        np.testing.assert_allclose(head.distance_center, observed_distances.mean(), atol=1e-12)
        np.testing.assert_allclose(head.distance_scale, observed_distances.std(ddof=1), atol=1e-12)
        expected_values = beta[0] + beta[1] * (
            observed_distances - observed_distances.mean()
        ) / observed_distances.std(ddof=1)
        query = (
            region if species is None else replace(region, species=(species[index],) * unit_count)
        )
        predicted = result.model.predict(
            query,
            species=None if species is None else species[index],
            pairs=observed.observed_pairs,
        )
        np.testing.assert_allclose(predicted.values, expected_values, atol=1e-8)


@pytest.mark.parametrize("steps", [0, 4])
@pytest.mark.parametrize("species_count", [1, 2])
def test_validation_targets_cannot_change_training_calibration_and_one_pair_is_supported(
    WeightedEmbedding,
    embedding_problem,
    steps,
    species_count,
):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    with jax.enable_x64():
        region, observations = embedding_problem()
        single_region = region
        order = None if species_count == 1 else ("stoat", "hare")
        observed_groups = (observations,)
        weights = jnp.zeros(2)
        if order is not None:
            kinds = region.sampling_unit_kinds
            assert kinds is not None
            region = replace(
                region,
                sampling_unit_ids=region.sampling_unit_ids * species_count,
                grid_positions=jnp.tile(jnp.asarray(region.grid_positions), (species_count, 1)),
                sampling_unit_kinds=kinds * species_count,
                species=tuple(name for name in order for _ in region.sampling_unit_ids),
            )
            observed_groups = tuple(
                replace(observations, species=name, values=observations.values * (index + 1))
                for index, name in enumerate(order)
            )
            weights = jnp.log(jnp.array([[1.0, 1.0], [2.0, 0.5]]))
        query_pair = observations.observed_pairs[-1:]
        train_pairs = observations.observed_pairs[:-1]
        train_partition = ObservationPartition(region.name, train_pairs, role="training")
        query_partition = ObservationPartition(region.name, query_pair, role="validation")
        altered = tuple(
            PairwiseObservations.from_pairs(
                group.observed_pairs,
                group.observed_values + 20,
                target=group.target,
                sampling_unit_ids=group.sampling_unit_ids,
                species=group.species,
            )
            for group in observed_groups
        )
        config = TrainingConfig(
            objective="mlpe",
            steps=steps,
            eval_every=3,
            learning_rate=0.03,
            mlpe=MLPEConfig(initial_variances=(0.05, 0.1)),
        )
        runs = [
            fit(
                region,
                observed_groups[0] if species_count == 1 else observed_groups,
                distance_model=WeightedEmbedding(weights, num_species=species_count),
                species=order,
                config=config,
                partition=train_partition,
                validation=(
                    region,
                    validation_observations[0] if species_count == 1 else validation_observations,
                ),
                validation_partition=query_partition,
            )
            for validation_observations in (observed_groups, altered)
        ]
    first, second = runs

    assert first.raw_variances is not None
    assert second.raw_variances is not None
    assert [r.training_loss for r in first.history] == [r.training_loss for r in second.history]
    np.testing.assert_array_equal(
        cast(Any, first.distance_model).log_weights, cast(Any, second.distance_model).log_weights
    )
    assert eqx.tree_equal(first.raw_variances, second.raw_variances)
    assert eqx.tree_equal(first.latest_model.calibration, second.latest_model.calibration)
    assert first.history[-1].validation_loss != second.history[-1].validation_loss
    assert first.rng_key.dtype == np.uint32
    expected_losses = []
    for index, observed in enumerate(observed_groups):
        name = None if order is None else order[index]
        distances = np.asarray(
            first.latest_model.distance_model.predict(
                region, species=name, species_order=order
            ).as_matrix()
        )
        head = cast(MLPECalibrationModule, first.latest_model.calibration).for_region(
            region.name, None if species_count == 1 else index
        )
        key = region.name if species_count == 1 else (region.name, index)
        assert set(head.calibration_pairs) == set(train_partition.pairs)
        assert first.latest_model.validation_pairs[key] == query_partition.pairs
        (left, right), _ = observed.aligned_pairs(single_region, train_partition)
        np.testing.assert_allclose(head.distance_center, distances[left, right].mean(), atol=1e-12)
        np.testing.assert_allclose(
            head.distance_scale, distances[left, right].std(ddof=1), atol=1e-12
        )
        (left, right), values = observed.aligned_pairs(single_region, query_partition)
        expected_mean = (
            head.intercept
            + head.slope * (distances[left, right] - head.distance_center) / head.distance_scale
        )
        variance = 2 * head.unit_variance + head.residual_variance
        expected_losses.append(
            0.5 * (np.log(2 * np.pi * variance) + (values[0] - expected_mean[0]) ** 2 / variance)
        )
    assert first.history[-1].validation_loss is not None
    np.testing.assert_allclose(
        first.history[-1].validation_loss, np.mean(expected_losses), rtol=1e-12
    )


def test_public_mlpe_conductance_fit_improves_with_actual_graph_solver(
    CovariateConductance, dense_resistance
):

    with jax.enable_x64():
        features = np.array([[[0.0], [0.2], [0.8]], [[-0.3], [1.4], [-0.7]]])
        ids = tuple(f"population-{i}" for i in range(6))
        region = RegionBatch(
            "graph",
            cx.field(features, "row", "column", "feature"),
            ids,
            np.array(list(np.ndindex(2, 3))),
            feature_names=("habitat",),
        )
        left, right = np.triu_indices(6, 1)
        distances = dense_resistance(
            np.exp(1.2 * region.feature_array[..., 0]), region.pixel_nodes
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
            region,
            observations,
            distance_model=CovariateConductance(jnp.asarray(0.1)),
            config=TrainingConfig(
                objective="mlpe",
                steps=20,
                learning_rate=0.08,
                mlpe=MLPEConfig(initial_variances=(0.02, 0.05)),
            ),
        )

        assert result.raw_variances is not None
        assert result.history[-1].training_loss < result.history[0].training_loss - 0.2
        assert float(cast(Any, result.model.distance_model).weight) > 0.6
        assert np.isfinite(result.model.predict(region).values).all()
        surface = cast(ConductanceModel, result.model.distance_model).conductance_surface(region)
        actual_distances = result.model.distance_model.predict(region).as_matrix()
    expected = dense_resistance(surface, region.pixel_nodes)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(actual_distances, expected, atol=1e-7)


def test_mlpe_training_reports_precision_identifiability_and_numerical_failures(
    WeightedEmbedding, embedding_problem
):
    from dataclasses import replace

    import pytest

    region, observations = embedding_problem()
    model = WeightedEmbedding(jnp.zeros(2))
    config = TrainingConfig(objective="mlpe", steps=0)
    with jax.enable_x64(False), pytest.raises(RuntimeError):
        fit(region, observations, distance_model=model, config=config)
    with jax.enable_x64():
        disjoint = PairwiseObservations.from_pairs(
            [("unit-0", "unit-1"), ("unit-2", "unit-3"), ("unit-4", "unit-5")],
            [1, 2, 3],
            target=observations.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        with pytest.raises(ValueError):
            fit(region, disjoint, distance_model=model, config=config)
        individual = replace(region, sampling_unit_kinds=("individual",) * 6)
        with pytest.raises(ValueError):
            fit(individual, observations, distance_model=model, config=config)
        constant_region = replace(
            region,
            features=cx.field(np.ones_like(region.feature_array), "row", "column", "feature"),
        )
        with pytest.raises(FloatingPointError):
            fit(constant_region, observations, distance_model=model, config=config)
        extreme = replace(config, mlpe=MLPEConfig(initial_variances=(1e12, 1e-9)))
        with pytest.raises(FloatingPointError):
            fit(region, observations, distance_model=model, config=extreme)


def test_continuation_rejects_changed_optimization_policy(WeightedEmbedding, embedding_problem):
    from dataclasses import replace

    import pytest

    with jax.enable_x64():
        region, observations = embedding_problem()
        config = TrainingConfig(objective="mlpe", steps=1)
        result = fit(
            region, observations, distance_model=WeightedEmbedding(jnp.zeros(2)), config=config
        )

        assert result.raw_variances is not None
        with pytest.raises(ValueError):
            fit(region, observations, state=result, config=replace(config, learning_rate=0.1))
        with pytest.raises(ValueError):
            fit(region, observations, state=result, config=replace(config, region_batch_size=1))
        with pytest.raises(ValueError):
            fit(region, observations, state=result, distance_model=result.distance_model)


@pytest.mark.parametrize(
    ("num_species", "species_batch_size", "region_batch_size"),
    [(1, None, size) for size in (None, 1, 2, 3)]
    + [(2, species, region) for species in (None, 1) for region in (None, 1)],
)
def test_shared_embedding_update_matches_independent_equal_region_objective(
    ScalarEmbedding,
    regions_and_observations,
    region_batch_size,
    num_species,
    species_batch_size,
    monkeypatch,
):
    from dataclasses import replace

    with jax.enable_x64():
        regions, observations = regions_and_observations()
        species = None if num_species == 1 else ("frog", "newt")
        if species is not None:
            # Frog occurs in both regions; newt only in the short region.
            regions = [
                replace(regions[0], species=("frog",) * 3),
                replace(
                    regions[1],
                    sampling_unit_ids=regions[1].sampling_unit_ids * 2,
                    grid_positions=np.tile(regions[1].grid_positions, (2, 1)),
                    sampling_unit_kinds=None,
                    species=("frog", "frog", "newt", "newt"),
                ),
            ]
            observations = [
                replace(observations[0], species="frog"),
                (
                    replace(observations[1], species="frog"),
                    replace(observations[1], species="newt", values=observations[1].values / 5),
                ),
            ]
        initial = 1.0

        def independent_objective(weight, indices: Sequence[int] = (0, 1), short_species=None):
            predictions = [np.array([4]) * weight**2, np.array([1, 4, 1]) * weight**2]
            targets = [np.array([15]), np.array([0.1, 0.1, 0.1])]
            losses = [
                np.mean((np.log1p(p) - np.log1p(t)) ** 2)
                for p, t in zip(predictions, targets, strict=True)
            ]
            if num_species == 2:
                newt = (np.log1p(4 * weight**2) - np.log1p(3)) ** 2
                short = losses[0] / 2 + newt
                if short_species is not None:
                    short = losses[0] if short_species == 0 else 2 * newt
                losses = [short, losses[1] / 2]
            return np.mean([losses[i] for i in indices])

        step = 1e-5
        dense_gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        gradients = []
        region_selections = [(0,), (1,)] if region_batch_size == 1 else [(0, 1)]
        selections = [
            (indices, short, 1 / len(region_selections) / len(short_choices))
            for indices in region_selections
            for short_choices in ([0, 1] if species_batch_size == 1 and 0 in indices else [None],)
            for short in short_choices
        ]
        probabilities = []
        for selected, short_species, probability in selections:
            draws = iter(
                ([selected] if region_batch_size == 1 else [])
                + ([(short_species,)] if short_species is not None else [])
            )
            monkeypatch.setattr(
                jax.random, "choice", lambda *a, draws=draws, **kw: jnp.array(next(draws))
            )
            gradient = (
                independent_objective(initial + step, selected, short_species)
                - independent_objective(initial - step, selected, short_species)
            ) / (2 * step)
            expected = initial - 0.01 * gradient / (abs(gradient) + 1e-8)
            result = fit(
                regions,
                observations,
                distance_model=ScalarEmbedding(
                    jnp.asarray(initial, dtype=jnp.float64), num_species=num_species
                ),
                species=species,
                config=TrainingConfig(
                    objective="direct_log1p",
                    steps=1,
                    learning_rate=0.01,
                    region_batch_size=region_batch_size,
                    species_batch_size=species_batch_size,
                ),
            )
            # Adam's first moment records 0.1 times the first applied gradient.
            gradients.append(cast(Any, result.optimizer_state)[0].mu[0].weight / 0.1)
            probabilities.append(probability)
            np.testing.assert_allclose(gradients[-1], gradient, rtol=1e-8)
            np.testing.assert_allclose(
                cast(Any, result.model.distance_model).weight, expected, rtol=1e-8
            )
            np.testing.assert_allclose(
                result.history[0].training_loss, independent_objective(initial), rtol=1e-8
            )
            assert result.region_names == ("short", "wide")
            for region in regions:
                assert np.isfinite(
                    result.model.predict(region, species=None if species is None else "frog").values
                ).all()
        np.testing.assert_allclose(np.dot(probabilities, gradients), dense_gradient, rtol=1e-8)


@pytest.mark.parametrize(("num_species", "jit"), [(1, True), (2, False), (2, True)])
@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_species_microbatches_preserve_updates_with_shared_parameters(
    training_problem, SharedParameterEmbedding, dense_profile, num_species, jit, objective
):
    from dataclasses import replace

    region, observations, _ = training_problem("embedding", num_species=num_species)
    species = None if num_species == 1 else ("species-0", "species-1")
    initial = np.array([0.6, 0.9])[:num_species]
    model = SharedParameterEmbedding(jnp.asarray(initial))
    config = TrainingConfig(
        objective=objective, steps=2, jit=jit, mlpe=MLPEConfig(initial_variances=(0.05, 0.1))
    )
    states = []
    reference = fit(
        region,
        observations,
        distance_model=model,
        species=species,
        config=config,
        on_step=states.append,
    )
    observed = (observations,) if num_species == 1 else observations
    features = np.asarray(region.feature_array).reshape(4, 2)
    left, right = np.triu_indices(4, 1)

    def reference_objective(weights):
        values = []
        for index, target in enumerate(observed):
            embeddings = features * weights.sum() * weights[index]
            distances = np.sum((embeddings[left] - embeddings[right]) ** 2, axis=1)
            order = [
                target.sampling_unit_ids.index(label) for label in region.sampling_unit_ids[:4]
            ]
            targets = np.asarray(target.values)[np.ix_(order, order)][left, right]
            values.append(
                np.mean((np.log1p(distances) - np.log1p(targets)) ** 2)
                if objective == "direct_log1p"
                else dense_profile(distances, targets, left, right, 4, (0.05, 0.1))[0] / 6
            )
        return np.mean(values)

    differences = np.eye(num_species) * 1e-5
    gradient = np.array(
        [
            (reference_objective(initial + delta) - reference_objective(initial - delta)) / 2e-5
            for delta in differences
        ]
    )
    np.testing.assert_allclose(
        states[0].history[0].training_loss, reference_objective(initial), atol=1e-9
    )
    np.testing.assert_allclose(
        cast(Any, states[1].optimizer_state)[0].mu[0].weights / 0.1, gradient, atol=1e-8
    )
    for size in (2,) if num_species == 1 else (1, 2):
        traces = []
        observed_model = SharedParameterEmbedding(
            jnp.asarray(initial), on_embedding=lambda traces=traces: traces.append(None)
        )
        result = fit(
            region,
            observations,
            distance_model=observed_model,
            species=species,
            config=replace(config, species_microbatch_size=size),
        )
        if jit:
            # Compatible species reuse evaluation and microbatch gradient programs.
            assert 0 < len(traces) <= 1 + min(size, num_species)
        for actual, expected in zip(
            jax.tree.leaves((result.distance_model, result.optimizer_state)),
            jax.tree.leaves((reference.distance_model, reference.optimizer_state)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_shared_conductance_update_matches_independent_combined_graph_objective(
    PointConductance, dense_resistance
):
    with jax.enable_x64():
        target = TargetSpec("synthetic graph divergence", units="index")
        regions = [
            RegionBatch(
                "large",
                cx.field(np.array([[[1.0], [2.0], [3.0]]]), "row", "column", "feature"),
                ("a", "b", "c"),
                np.array([[0, 0], [0, 1], [0, 2]]),
                feature_names=("covariate",),
            ),
            RegionBatch(
                "small",
                cx.field(np.array([[[1.0], [3.0]]]), "row", "column", "feature"),
                ("x", "y"),
                np.array([[0, 0], [0, 1]]),
                feature_names=("covariate",),
            ),
        ]
        observations = [
            PairwiseObservations.from_matrix(
                ("a", "b", "c"), [[0, 0.01, 0.01], [0.01, 0, 0.01], [0.01, 0.01, 0]], target=target
            ),
            PairwiseObservations.from_pairs([("x", "y")], [5], target=target),
        ]

        def independent_objective(log_scale):
            losses = []
            for region, observations_for_region in zip(regions, observations, strict=True):
                surface = np.exp(log_scale) * region.feature_array[..., 0].astype(np.float64)
                distances = dense_resistance(surface, region.pixel_nodes)
                pairs = np.triu_indices(len(region.sampling_unit_ids), 1)
                losses.append(
                    np.mean(
                        (
                            np.log1p(distances[pairs])
                            - np.log1p(observations_for_region.values[pairs])
                        )
                        ** 2
                    )
                )
            return np.mean(losses)

        initial, step = 0.2, 1e-5
        gradient = (
            independent_objective(initial + step) - independent_objective(initial - step)
        ) / (2 * step)
        expected = initial - 0.01 * gradient / (abs(gradient) + 1e-8)
        result = fit(
            regions,
            observations,
            distance_model=PointConductance(jnp.asarray(initial, dtype=jnp.float64)),
            config=TrainingConfig(objective="direct_log1p", steps=1, learning_rate=0.01),
        )
        np.testing.assert_allclose(
            cast(Any, result.model.distance_model).log_scale, expected, rtol=1e-8
        )
        np.testing.assert_allclose(
            result.history[0].training_loss, independent_objective(initial), rtol=1e-8
        )
        for region in regions:
            surface = cast(ConductanceModel, result.model.distance_model).conductance_surface(
                region
            )
            expected_distances = dense_resistance(surface, region.pixel_nodes)
            np.fill_diagonal(expected_distances, np.nan)
            np.testing.assert_allclose(
                result.model.predict(region).as_matrix(),
                expected_distances,
                rtol=1e-7,
            )


def test_region_order_and_validation_do_not_change_training_random_trajectory(
    StochasticEmbedding, regions_and_observations
):
    from dataclasses import replace

    regions, observations = regions_and_observations()
    config = TrainingConfig(objective="direct_log1p", steps=3, seed=7, region_batch_size=1)
    model = StochasticEmbedding(jnp.asarray(1.0, dtype=jnp.float32))
    baseline = fit(regions, observations, distance_model=model, config=config)
    reversed_regions = {region.name: region for region in reversed(regions)}
    reversed_observations = {
        region.name: observed
        for region, observed in reversed(list(zip(regions, observations, strict=True)))
    }
    validation_region = replace(regions[0], name="validation-only")
    validated = fit(
        reversed_regions,
        reversed_observations,
        distance_model=model,
        config=config,
        validation=(validation_region, observations[0]),
    )
    assert [step.training_by_region for step in baseline.history] == [
        step.training_by_region for step in validated.history
    ]
    reordered = fit(reversed_regions, reversed_observations, distance_model=model, config=config)
    np.testing.assert_array_equal(
        cast(Any, baseline.model.distance_model).weight,
        cast(Any, reordered.model.distance_model).weight,
    )


def test_incompatible_regional_inputs_fail_before_optimization(
    ScalarEmbedding, regions_and_observations
):
    from dataclasses import replace

    import pytest

    from ilg_toolkit import ObservationPartition

    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    cases = [
        ([regions[0], regions[0]], observations, None),
        (
            [
                replace(
                    region,
                    features=cx.field(region.feature_array, "row", "column", "feature"),
                    feature_names=None,
                )
                for region in regions
            ],
            observations,
            None,
        ),
        (
            [
                regions[0],
                replace(
                    regions[1],
                    features=cx.field(regions[1].feature_array, "row", "column", "feature"),
                    feature_names=("canopy",),
                ),
            ],
            observations,
            None,
        ),
        ({"wrong-name": regions[0]}, {"wrong-name": observations[0]}, None),
        (regions, {regions[0].name: observations[0]}, None),
        (regions, observations, {regions[0].name: None}),
        (regions, observations, [ObservationPartition("wrong-region", (("a", "b"),)), None]),
    ]
    for supplied_regions, supplied_observations, supplied_partitions in cases:
        with pytest.raises(ValueError):
            fit(
                supplied_regions,
                supplied_observations,
                distance_model=model,
                partition=supplied_partitions,
                config=TrainingConfig(objective="direct_log1p", steps=0),
            )


def test_validation_on_another_region_requires_declared_feature_meanings(
    ScalarEmbedding, regions_and_observations
):
    from dataclasses import replace

    import pytest

    regions, observations = regions_and_observations()
    regions = [
        replace(
            region,
            features=cx.field(region.feature_array, "row", "column", "feature"),
            feature_names=None,
        )
        for region in regions
    ]
    with pytest.raises(ValueError):
        fit(
            regions[0],
            observations[0],
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            validation=(regions[1], observations[1]),
            config=TrainingConfig(objective="direct_log1p", steps=0),
        )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("validation_used", [False, True])
def test_evaluation_cadence_selects_model_and_reports_every_step(
    ScalarEmbedding, regions_and_observations, jit, validation_used
):
    regions, observed = regions_and_observations()
    region, observations = regions[0], observed[0]
    model = ScalarEmbedding(jnp.asarray(1.0))
    # This deliberately prefers the initial model while training targets differ.
    initial = model.predict(region)
    validation = PairwiseObservations.from_pairs(
        initial.pairs, initial.values, target=observations.target
    )
    from dataclasses import replace

    validation_region = replace(region, name="independent-validation-region")
    callbacks = []
    result = fit(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(
            objective="direct_log1p", steps=5, eval_every=2, learning_rate=0.01, jit=jit
        ),
        validation=(validation_region, validation) if validation_used else None,
        on_step=callbacks.append,
    )
    assert result.selected_step == (0 if validation_used else 5)
    assert [record.step for record in result.history] == [0, 2, 4, 5]
    assert [state.step for state in callbacks] == [0, 1, 2, 3, 4, 5]
    if validation_used:
        final_loss = result.history[-1].validation_loss
        initial_loss = result.history[0].validation_loss
        assert final_loss is not None and initial_loss is not None
        assert final_loss > initial_loss
        np.testing.assert_allclose(
            result.selected_model.predict(region).values, initial.values, atol=1e-5
        )
    else:
        assert eqx.tree_equal(result.selected_model, result.latest_model)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("species_count", [1, 2])
def test_zero_step_recalibration_fits_only_supplied_region_and_preserves_distances(
    WeightedEmbedding, embedding_problem, dense_profile, jit, species_count
):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    region, observations = embedding_problem()
    transferred_region, transferred_observations = embedding_problem("valley", offset=3)
    single_transferred = transferred_region
    order = None if species_count == 1 else ("stoat", "hare")
    selected_species = None if order is None else order[1]
    weights = jnp.zeros(2)
    training_observations = observations
    if order is not None:
        prepared_regions = []
        for prepared in (region, transferred_region):
            kinds = prepared.sampling_unit_kinds
            assert kinds is not None
            prepared_regions.append(
                replace(
                    prepared,
                    sampling_unit_ids=prepared.sampling_unit_ids * species_count,
                    grid_positions=jnp.tile(
                        jnp.asarray(prepared.grid_positions), (species_count, 1)
                    ),
                    sampling_unit_kinds=kinds * species_count,
                    species=tuple(name for name in order for _ in prepared.sampling_unit_ids),
                )
            )
        region, transferred_region = prepared_regions
        training_observations = tuple(replace(observations, species=name) for name in order)
        transferred_observations = replace(transferred_observations, species=selected_species)
        weights = jnp.log(jnp.array([[1.0, 1.0], [2.0, 0.5]]))
    frozen = WeightedEmbedding(weights, num_species=species_count)
    if species_count > 1:
        with pytest.raises(ValueError):
            fit(
                region,
                training_observations,
                distance_model=frozen,
                config=TrainingConfig(objective="mlpe", steps=0, jit=jit),
            )
    initial = fit(
        region,
        training_observations,
        distance_model=frozen,
        species=order,
        config=TrainingConfig(objective="mlpe", steps=0, jit=jit),
    ).selected_model
    partition = ObservationPartition(
        "valley", transferred_observations.observed_pairs[:-1], role="training"
    )
    callbacks = []
    result = fit(
        transferred_region,
        transferred_observations,
        distance_model=initial.distance_model,
        species=order,
        partition=partition,
        config=TrainingConfig(
            objective="mlpe",
            steps=0,
            eval_every=7,
            jit=jit,
            mlpe=MLPEConfig(variance_floor=1e-8, jitter=1e-7, initial_variances=(0.05, 0.1)),
        ),
        on_step=callbacks.append,
    )
    assert result.step == result.selected_step == 0
    assert [record.step for record in result.history] == [0]
    assert len(callbacks) == 1
    assert eqx.tree_equal(result.selected_model.distance_model, frozen)
    model = result.selected_model
    distances = frozen.predict(
        transferred_region, species=selected_species, species_order=order, pairs=partition.pairs
    ).values
    (left, right), targets = transferred_observations.aligned_pairs(single_transferred, partition)
    nll, beta, effects, covariance = dense_profile(
        distances,
        targets,
        left,
        right,
        len(single_transferred.sampling_unit_ids),
        (0.05, 0.1),
        jitter=1e-7,
    )
    calibration = cast(MLPECalibrationModule, model.calibration).for_region(
        "valley", None if species_count == 1 else 1
    )
    np.testing.assert_allclose(result.history[0].training_loss, nll / len(targets), atol=1e-9)
    np.testing.assert_allclose([calibration.intercept, calibration.slope], beta, atol=1e-9)
    np.testing.assert_allclose(calibration.effect_mean, effects, atol=1e-9)
    np.testing.assert_allclose(calibration.effect_covariance, covariance, atol=1e-9)
    assert set(calibration.calibration_pairs) == set(partition.pairs)
    assert np.isfinite(model.predict(transferred_region, species=selected_species).values).all()
    with pytest.raises(ValueError):
        model.predict(region, species=selected_species)
    if order is not None:
        with pytest.raises(ValueError):
            model.predict(transferred_region, species=order[0])


def test_fit_validates_species_order_and_derives_a_single_name(
    observation_problem, ScalarEmbedding
):
    from dataclasses import replace

    region, target, _ = observation_problem()
    region = replace(region, species=("a",) * 3)
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [0.5], target=target, species="a"
    )
    config = TrainingConfig(objective="direct_log1p", steps=0)
    for order in ("ab", {"a", "b"}, ("a",), ("a", "a"), ("", "a")):
        with pytest.raises(ValueError):
            fit(
                region,
                observations,
                distance_model=ScalarEmbedding(jnp.array(1.0), num_species=2),
                species=order,  # type: ignore[arg-type]
                config=config,
            )
    result = fit(
        region, observations, distance_model=ScalarEmbedding(jnp.array(1.0)), config=config
    )
    assert result.model.species == ("a",)
    assert result.history[0].training_by_species is None
    np.testing.assert_array_equal(
        result.model.predict(region).values, result.model.predict(region, species="a").values
    )
