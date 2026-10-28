"""Scientific behavior of training."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast, final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ilg_toolkit import (
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationModule,
    MLPELoss,
    PairwiseObservations,
    Prediction,
    RegionBatch,
    ResistanceSolverConfig,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import AbstractDistanceModel, ConductanceModel, EmbeddingDistanceModel


def _single_species_prediction(
    model: LandscapeGeneticsModel | AbstractDistanceModel,
    region: RegionBatch,
    name: str | None,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> Prediction:
    """A named species' prediction, or the prediction of an unnamed single-species model."""
    if name is None:
        return model.predict(region, pairs)
    return model.predict(region, {name: pairs})[name]


class _Jitter(eqx.Module):
    """Inference-dependent layer, other than dropout, that rescales by random noise."""

    inference: bool = False

    def __call__(self, x, *, key: jax.Array | None = None):
        if self.inference:
            return x
        if key is None:
            raise ValueError("Training a jitter layer requires a random key")
        return x * jax.random.uniform(key, x.shape, minval=0.5, maxval=1.5)


@final
class _JitteredEmbedding(EmbeddingDistanceModel):
    weight: jax.Array
    jitter: _Jitter
    species: tuple[str, ...] | None = eqx.field(static=True, default=None)
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight
        self.jitter = _Jitter()

    def backbone(self, features):
        return self.jitter(features)

    def embedding(self, latent, species_index):
        return self.weight * latent


def test_failed_solver_preserves_solver_failure(PointConductance):
    with jax.enable_x64():
        region = RegionBatch(
            "bad-solve",
            cx.field(np.linspace(0.1, 2, 12).reshape(3, 4, 1), "row", "column", "feature"),
            ("first", "last"),
            np.array([[0, 0], [2, 3]]),
        )
        observations = PairwiseObservations(
            region.sampling_unit_ids, [[0, 1], [1, 0]], target=TargetSpec("dissimilarity")
        )
        with pytest.raises(RuntimeError):
            train(
                region,
                observations,
                distance_model=PointConductance(
                    jnp.array(0.0),
                    solver_config=ResistanceSolverConfig(rtol=1e-12, atol=1e-12, max_steps=1),
                ),
                config=TrainingConfig(
                    loss=Log1pLoss(),
                    steps=0,
                ),
            )


def test_nonfinite_training_is_reported_at_evaluation(ScalarEmbedding, regions_and_observations):
    regions, observations = regions_and_observations()
    completed = []
    with pytest.raises(FloatingPointError):
        train(
            regions[0],
            observations[0],
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            config=TrainingConfig(loss=Log1pLoss(), steps=3, eval_every=2, learning_rate=1e200),
            on_step=lambda state: completed.append(state.step),
        )
    assert completed == [0]


@pytest.mark.parametrize(
    ("num_species", "region_count", "species_batch_size"),
    [
        # Two regions with unequal pair counts keep equal influence in full batches.
        (1, 2, None),
        # A sampled species batch updates only the selected group's MLPE state.
        (2, 1, 1),
    ],
)
def test_mlpe_updates_match_dense_reference(
    WeightedEmbedding,
    dense_profile,
    embedding_problem,
    num_species,
    region_count,
    species_batch_size,
    monkeypatch,
):
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
    variances_by_group = [
        np.var(np.asarray(observed.observed_values), ddof=1) * np.array([0.25, 0.5])
        for _, observed in problems
    ]
    raw = np.concatenate([np.log(np.expm1(values - 1e-10)) for values in variances_by_group])
    initial = np.concatenate((np.zeros(parameter_count), raw))

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
        choices = iter(selections)
        monkeypatch.setattr(jax.random, "choice", lambda *args, **kwargs: jnp.array(next(choices)))
    else:
        selections = [range(group_count)] * 2

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
    result = train(
        regions,
        observations,
        distance_model=WeightedEmbedding(
            jnp.zeros(2) if num_species == 1 else jnp.zeros((2, 2)), species=species
        ),
        config=TrainingConfig(
            loss=MLPELoss(),
            steps=2,
            learning_rate=0.02,
            species_batch_size=species_batch_size,
        ),
    )
    np.testing.assert_allclose(result.history[0].training_loss, objective(initial), atol=1e-12)
    np.testing.assert_allclose(result.history[-1].training_loss, objective(expected), atol=1e-8)
    assert result.history[-1].training_loss < result.history[0].training_loss
    assert result.step == result.selected_step == 2
    calibration = cast(MLPECalibrationModule, result.selected_model.calibration)
    for index, (region, observed) in enumerate(problems):
        embedding = np.asarray(region.feature_array).reshape(unit_count, 2) * np.exp(
            expected[2 * groups_species[index] : 2 * groups_species[index] + 2]
        )
        distances = ((embedding[:, None] - embedding[None, :]) ** 2).sum(-1)
        np.fill_diagonal(distances, np.nan)
        np.testing.assert_allclose(
            _single_species_prediction(
                result.selected_model.distance_model,
                region
                if species is None
                else replace(region, species=(species[groups_species[index]],) * unit_count),
                None if species is None else species[groups_species[index]],
            ).as_matrix(),
            distances,
            rtol=1e-8,
            atol=1e-9,
        )
        variances = (
            np.logaddexp(0, expected[parameter_count + 2 * index : parameter_count + 2 * index + 2])
            + 1e-10
        )
        head = calibration.for_region(region.name, 0 if species is None else groups_species[index])
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
        assert not np.allclose(variances, variances_by_group[index])
        observed_distances = distances[left, right]
        np.testing.assert_allclose(head.distance_center, observed_distances.mean(), atol=1e-12)
        np.testing.assert_allclose(head.distance_scale, observed_distances.std(ddof=1), atol=1e-12)
        expected_values = beta[0] + beta[1] * (
            observed_distances - observed_distances.mean()
        ) / observed_distances.std(ddof=1)
        query = (
            region if species is None else replace(region, species=(species[index],) * unit_count)
        )
        predicted = _single_species_prediction(
            result.selected_model,
            query,
            None if species is None else species[index],
            observed.observed_pairs,
        )
        np.testing.assert_allclose(predicted.values, expected_values, atol=1e-8)


def test_validation_targets_cannot_change_training_calibration_and_one_pair_is_supported(
    WeightedEmbedding, embedding_problem
):
    with jax.enable_x64():
        region, observations = embedding_problem(columns=2)
        single_region = region
        order = ("stoat", "hare")
        kinds = region.sampling_unit_kinds
        assert kinds is not None
        region = replace(
            region,
            sampling_unit_ids=region.sampling_unit_ids * 2,
            grid_positions=jnp.tile(jnp.asarray(region.grid_positions), (2, 1)),
            sampling_unit_kinds=kinds * 2,
            species=tuple(name for name in order for _ in region.sampling_unit_ids),
        )
        observed_groups = tuple(
            replace(observations, species=name, values=observations.values * (index + 1))
            for index, name in enumerate(order)
        )
        weights = jnp.log(jnp.array([[1.0, 1.0], [2.0, 0.5]]))
        query_pair = observations.observed_pairs[-1:]
        train_pairs = observations.observed_pairs[:-1]
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
            loss=MLPELoss(),
            steps=4,
            eval_every=3,
            learning_rate=0.03,
        )
        runs = [
            train(
                region,
                tuple(group.select_pairs(train_pairs) for group in observed_groups),
                distance_model=WeightedEmbedding(weights, species=order),
                config=config,
                validation=(
                    region,
                    tuple(group.select_pairs(query_pair) for group in validation_observations),
                ),
            )
            for validation_observations in (observed_groups, altered)
        ]
    first, second = runs

    assert first.loss_state is not None
    assert second.loss_state is not None
    assert [r.training_loss for r in first.history] == [r.training_loss for r in second.history]
    np.testing.assert_array_equal(
        cast(Any, first.latest_model.distance_model).log_weights,
        cast(Any, second.latest_model.distance_model).log_weights,
    )
    assert eqx.tree_equal(first.loss_state, second.loss_state)
    assert eqx.tree_equal(first.latest_model.calibration, second.latest_model.calibration)
    assert first.history[-1].validation_loss != second.history[-1].validation_loss
    assert jnp.issubdtype(first.rng_key.dtype, jax.dtypes.prng_key)
    expected_losses = []
    for index, (name, observed) in enumerate(zip(order, observed_groups, strict=True)):
        distances = np.asarray(
            first.latest_model.distance_model.predict(region, {name: None})[name].as_matrix()
        )
        head = cast(MLPECalibrationModule, first.latest_model.calibration).for_region(
            region.name, index
        )
        assert set(head.calibration_pairs) == set(train_pairs)
        (left, right), _ = observed.select_pairs(train_pairs).aligned_pairs(single_region)
        np.testing.assert_allclose(head.distance_center, distances[left, right].mean(), atol=1e-12)
        np.testing.assert_allclose(
            head.distance_scale, distances[left, right].std(ddof=1), atol=1e-12
        )
        (left, right), values = observed.select_pairs(query_pair).aligned_pairs(single_region)
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
        # Four populations give the same MLPE pair design as the embedding problems.
        features = np.array([[[0.0], [0.2], [0.8]], [[-0.3], [1.4], [-0.7]]])
        ids = tuple(f"population-{i}" for i in range(4))
        region = RegionBatch(
            "graph",
            cx.field(features, "row", "column", "feature"),
            ids,
            np.array([[0, 0], [0, 2], [1, 0], [1, 2]]),
            feature_names=("habitat",),
        )
        left, right = np.triu_indices(4, 1)
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
        result = train(
            region,
            observations,
            distance_model=CovariateConductance(jnp.asarray(0.1)),
            config=TrainingConfig(
                loss=MLPELoss(),
                steps=20,
                learning_rate=0.08,
            ),
        )

        assert result.loss_state is not None
        assert result.history[-1].training_loss < result.history[0].training_loss - 0.2
        assert float(cast(Any, result.selected_model.distance_model).weight) > 0.6
        assert np.isfinite(result.selected_model.predict(region).values).all()
        surface = cast(ConductanceModel, result.selected_model.distance_model).conductance_surface(
            region
        )
        actual_distances = result.selected_model.distance_model.predict(region).as_matrix()
    expected = dense_resistance(surface, region.pixel_nodes)
    np.fill_diagonal(expected, np.nan)
    np.testing.assert_allclose(actual_distances, expected, atol=1e-7)


def test_mlpe_training_reports_precision_identifiability_and_numerical_failures(
    WeightedEmbedding, embedding_problem
):
    region, observations = embedding_problem(columns=2)
    model = WeightedEmbedding(jnp.zeros(2))
    config = TrainingConfig(loss=MLPELoss(), steps=0)
    with jax.enable_x64():
        six_units, _ = embedding_problem(columns=3)
        disjoint = PairwiseObservations.from_pairs(
            [("unit-0", "unit-1"), ("unit-2", "unit-3"), ("unit-4", "unit-5")],
            [1, 2, 3],
            target=observations.target,
            sampling_unit_ids=six_units.sampling_unit_ids,
        )
        with pytest.raises(ValueError):
            train(six_units, disjoint, distance_model=model, config=config)
        individual = replace(region, sampling_unit_kinds=("individual",) * 4)
        with pytest.raises(ValueError):
            train(individual, observations, distance_model=model, config=config)
        constant_region = replace(
            region,
            features=cx.field(np.ones_like(region.feature_array), "row", "column", "feature"),
        )
        with pytest.raises(FloatingPointError):
            train(constant_region, observations, distance_model=model, config=config)


def test_continuation_rejects_changed_optimization_policy(mlpe_embedding_fit):
    region, observations, result = (
        mlpe_embedding_fit.region,
        mlpe_embedding_fit.observations,
        mlpe_embedding_fit.state,
    )
    assert result.loss_state is not None
    with pytest.raises(ValueError):
        train(region, observations, state=result, config=replace(result.config, learning_rate=0.1))
    with pytest.raises(ValueError):
        train(
            region, observations, state=result, config=replace(result.config, region_batch_size=1)
        )
    with pytest.raises(ValueError):
        train(
            region,
            observations,
            state=result,
            distance_model=result.latest_model.distance_model,
        )


@pytest.mark.parametrize(
    ("num_species", "species_batch_size", "region_batch_size"),
    [
        # A region batch larger than the region count is the full batch.
        (1, None, 3),
        # Sampled regions weight their species groups equally.
        (2, None, 1),
        # Sampled species within full region batches.
        (2, 1, None),
    ],
)
def test_shared_embedding_update_matches_independent_equal_region_objective(
    ScalarEmbedding,
    regions_and_observations,
    region_batch_size,
    num_species,
    species_batch_size,
    monkeypatch,
):
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
            result = train(
                regions,
                observations,
                distance_model=ScalarEmbedding(
                    jnp.asarray(initial, dtype=jnp.float64), species=species
                ),
                config=TrainingConfig(
                    loss=Log1pLoss(),
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
                cast(Any, result.selected_model.distance_model).weight, expected, rtol=1e-8
            )
            np.testing.assert_allclose(
                result.history[0].training_loss, independent_objective(initial), rtol=1e-8
            )
            assert set(result.history[0].training_by_region) == {"short", "wide"}
            for region in regions:
                assert np.isfinite(
                    _single_species_prediction(
                        result.selected_model, region, None if species is None else "frog"
                    ).values
                ).all()
        np.testing.assert_allclose(np.dot(probabilities, gradients), dense_gradient, rtol=1e-8)


@pytest.mark.parametrize("objective", ["direct_log1p", "mlpe"])
def test_species_microbatches_preserve_updates_with_shared_parameters(
    training_problem, SharedParameterEmbedding, dense_profile, objective
):
    region, observations, _ = training_problem("embedding", num_species=2)
    species = ("species-0", "species-1")
    initial = np.array([0.6, 0.9])
    model = SharedParameterEmbedding(jnp.asarray(initial), species)
    config = TrainingConfig(
        loss=Log1pLoss() if objective == "direct_log1p" else MLPELoss(), steps=2
    )
    states = []
    reference = train(
        region,
        observations,
        distance_model=model,
        config=config,
        on_step=states.append,
    )
    features = np.asarray(region.feature_array).reshape(4, 2)
    left, right = np.triu_indices(4, 1)

    def reference_objective(weights):
        values = []
        for index, target in enumerate(observations):
            embeddings = features * weights.sum() * weights[index]
            distances = np.sum((embeddings[left] - embeddings[right]) ** 2, axis=1)
            order = [
                target.sampling_unit_ids.index(label) for label in region.sampling_unit_ids[:4]
            ]
            targets = np.asarray(target.values)[np.ix_(order, order)][left, right]
            values.append(
                np.mean((np.log1p(distances) - np.log1p(targets)) ** 2)
                if objective == "direct_log1p"
                else dense_profile(
                    distances,
                    targets,
                    left,
                    right,
                    4,
                    np.var(targets, ddof=1) * np.array([0.25, 0.5]),
                )[0]
                / 6
            )
        return np.mean(values)

    differences = np.eye(2) * 1e-5
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
    for size in (1, 2):
        traces = []
        observed_model = SharedParameterEmbedding(
            jnp.asarray(initial), species, on_embedding=lambda traces=traces: traces.append(None)
        )
        result = train(
            region,
            observations,
            distance_model=observed_model,
            config=replace(config, species_microbatch_size=size),
        )
        # Compatible species reuse evaluation and microbatch gradient programs.
        assert 0 < len(traces) <= 1 + size
        for actual, expected in zip(
            jax.tree.leaves((result.latest_model.distance_model, result.optimizer_state)),
            jax.tree.leaves((reference.latest_model.distance_model, reference.optimizer_state)),
            strict=True,
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)


def test_region_order_and_validation_do_not_change_training_random_trajectory(
    StochasticEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    config = TrainingConfig(loss=Log1pLoss(), steps=3, seed=7, region_batch_size=1)
    model = StochasticEmbedding(jnp.asarray(1.0, dtype=jnp.float32))
    baseline = train(regions, observations, distance_model=model, config=config)
    reversed_regions = {region.name: region for region in reversed(regions)}
    reversed_observations = {
        region.name: observed
        for region, observed in reversed(list(zip(regions, observations, strict=True)))
    }
    validation_region = replace(regions[0], name="validation-only")
    validated = train(
        reversed_regions,
        reversed_observations,
        distance_model=model,
        config=config,
        validation=(validation_region, observations[0]),
    )
    assert [step.training_by_region for step in baseline.history] == [
        step.training_by_region for step in validated.history
    ]
    reordered = train(reversed_regions, reversed_observations, distance_model=model, config=config)
    np.testing.assert_array_equal(
        cast(Any, baseline.selected_model.distance_model).weight,
        cast(Any, reordered.selected_model.distance_model).weight,
    )


def test_incompatible_regional_inputs_fail_before_optimization(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    cases = [
        ([regions[0], regions[0]], observations),
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
        ),
        ({"wrong-name": regions[0]}, {"wrong-name": observations[0]}),
        (regions, {regions[0].name: observations[0]}),
    ]
    for supplied_regions, supplied_observations in cases:
        with pytest.raises(ValueError):
            train(
                supplied_regions,
                supplied_observations,
                distance_model=model,
                config=TrainingConfig(loss=Log1pLoss(), steps=0),
            )


def test_validation_on_another_region_requires_declared_feature_meanings(
    ScalarEmbedding, regions_and_observations
):
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
        train(
            regions[0],
            observations[0],
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            validation=(regions[1], observations[1]),
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )


@pytest.mark.parametrize("validation_used", [False, True])
def test_evaluation_cadence_selects_model_and_reports_every_step(
    ScalarEmbedding, regions_and_observations, validation_used
):
    regions, observed = regions_and_observations()
    region, observations = regions[0], observed[0]
    model = ScalarEmbedding(jnp.asarray(1.0))
    # This deliberately prefers the initial model while training targets differ.
    initial = model.predict(region)
    validation = PairwiseObservations.from_pairs(
        initial.pairs, initial.values, target=observations.target
    )
    validation_region = replace(region, name="independent-validation-region")
    callbacks = []
    result = train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=5, eval_every=2, learning_rate=0.01),
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


def test_training_reads_species_names_from_the_distance_model(
    training_problem, observation_problem, ScalarEmbedding
):
    region, observations, distance_model = training_problem("embedding", num_species=2)
    config = TrainingConfig(loss=Log1pLoss(), steps=1)
    result = train(region, observations, distance_model=distance_model, config=config)
    names = ("species-0", "species-1")
    assert result.selected_model.species == result.latest_model.species == names
    assert set(cast(dict, result.history[0].training_by_species)) == set(names)
    continued = train(region, observations, state=result, config=replace(config, steps=2))
    assert continued.selected_model.species == names
    predictions = continued.selected_model.predict(region, dict.fromkeys(names))
    assert [prediction.species for prediction in predictions.values()] == list(names)
    with pytest.raises(ValueError, match="species"):
        train(
            region,
            (observations[0], replace(observations[1], species="species-2")),
            distance_model=distance_model,
            config=config,
        )

    region, target, _ = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    unnamed = train(
        region, observations, distance_model=ScalarEmbedding(jnp.array(1.0)), config=config
    )
    assert unnamed.selected_model.species is None
    assert unnamed.history[0].training_by_species is None
    assert unnamed.selected_model.predict(region).species is None
    with pytest.raises(ValueError, match="distance model with species"):
        train(
            region,
            replace(observations, species="a"),
            distance_model=ScalarEmbedding(jnp.array(1.0)),
            config=config,
        )
    labeled = replace(region, species=("a",) * 3)
    named = train(
        labeled,
        observations,
        distance_model=ScalarEmbedding(jnp.array(1.0), species=("a",)),
        config=config,
    )
    assert named.selected_model.species == ("a",)
    assert named.history[0].training_by_species is None
    np.testing.assert_array_equal(
        named.selected_model.predict(labeled).values,
        named.selected_model.predict(labeled, {"a": None})["a"].values,
    )


def test_callback_models_keep_matching_distance_and_calibration_state(
    mlpe_embedding_fit, dense_profile
):
    region, observations = mlpe_embedding_fit.region, mlpe_embedding_fit.observations
    assert [record.step for record in mlpe_embedding_fit.state.history] == [0, 3]
    assert [state.step for state in mlpe_embedding_fit.callbacks] == [0, 1, 2, 3]
    (left, right), targets = observations.aligned_pairs(region)
    for current in mlpe_embedding_fit.callbacks:
        model = current.latest_model
        distances = model.distance_model.predict(region, observations.observed_pairs).values
        calibration = cast(MLPECalibrationModule, model.calibration).for_region(region.name)
        nll, beta, effects, covariance = dense_profile(
            distances,
            targets,
            left,
            right,
            len(region.sampling_unit_ids),
            (calibration.unit_variance, calibration.residual_variance),
        )
        np.testing.assert_allclose(-calibration.ml_log_likelihood, nll, atol=1e-10)
        np.testing.assert_allclose([calibration.intercept, calibration.slope], beta, atol=1e-10)
        np.testing.assert_allclose(calibration.effect_mean, effects, atol=1e-10)
        np.testing.assert_allclose(calibration.effect_covariance, covariance, atol=1e-10)


def test_training_rejects_stateful_layers(BatchNormConductance, regions_and_observations):
    regions, observations = regions_and_observations()
    with pytest.raises(TypeError, match="stateful layers"):
        train(
            regions,
            observations,
            distance_model=BatchNormConductance(),
            config=TrainingConfig(loss=Log1pLoss(), steps=1),
        )


def test_training_passes_random_keys_to_custom_inference_layers(regions_and_observations):
    regions, observations = regions_and_observations()
    model = _JitteredEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(loss=Log1pLoss(), steps=2, seed=3)
    trained = train(regions, observations, distance_model=model, config=config)
    repeated = train(regions, observations, distance_model=model, config=config)
    reseeded = train(regions, observations, distance_model=model, config=replace(config, seed=4))
    weight = cast(Any, trained.latest_model.distance_model).weight
    assert weight == cast(Any, repeated.latest_model.distance_model).weight
    assert weight != cast(Any, reseeded.latest_model.distance_model).weight
    region = regions[0]
    assert cast(Any, trained.latest_model.distance_model).jitter.inference is False
    np.testing.assert_allclose(
        trained.latest_model.distance_model.predict(region).values,
        weight**2 * np.array([1.0, 4.0, 1.0]),
    )


def test_training_requires_float64(observation_problem, float64_required):
    region, target, model = observation_problem()
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20"), ("unit/20", "unit/99")], [1.0, 2.0], target=target
    )
    with float64_required():
        train(
            region,
            observations,
            distance_model=model,
            config=TrainingConfig(loss=Log1pLoss(), steps=0),
        )
