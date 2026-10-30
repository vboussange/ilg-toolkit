"""Tests of training: dense-reference updates, batching, penalties, selection and continuation."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast, final, NamedTuple

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from ilg_toolkit import (
    LandscapeGeneticsModel,
    Log1pLoss,
    MLPECalibrationHead,
    MLPELoss,
    PairwiseObservations,
    Prediction,
    RegionBatch,
    ResistanceSolverConfig,
    RidgePenalty,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import (
    AbstractDistanceModel,
    ConductanceModel,
    EmbeddingDistanceModel,
    LogLinearConductance,
)


def _single_species_prediction(
    model: LandscapeGeneticsModel | AbstractDistanceModel,
    region: RegionBatch,
    name: str | None,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> Prediction:
    """Returns a named species' prediction, or that of an unnamed single-species model."""
    if name is None:
        return model.predict(region, pairs)
    return model.predict(region, {name: pairs})[name]


class _Jitter(eqx.Module):
    """An inference-dependent layer, other than dropout, that rescales by random noise."""

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
            cx.field(
                np.linspace(0.1, 2, 12).reshape(3, 4, 1),
                "row",
                "column",
                cx.LabeledAxis("feature", np.array(["elevation"])),
            ),
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
            config=TrainingConfig(
                loss=Log1pLoss(), steps=3, eval_every=2, optimizer=optax.adam(1e200)
            ),
            on_step=lambda state: completed.append(state.step),
        )
    assert completed == [0]


@pytest.mark.parametrize(
    ("species", "region_count", "species_batch_size"),
    [
        # Two regions with unequal pair counts keep equal influence in full batches.
        (None, 2, None),
        # A sampled species batch updates only the selected group's MLPE state.
        (("newt", "frog"), 1, 1),
    ],
)
def test_mlpe_updates_match_dense_reference(
    WeightedEmbedding,
    dense_profile,
    embedding_problem,
    species,
    region_count,
    species_batch_size,
    monkeypatch,
):
    problems = [embedding_problem(columns=2)]
    unit_count = len(problems[0][0].sampling_unit_ids)
    if region_count == 2:
        region, observed = embedding_problem("valley", offset=3, seed=71, columns=2)
        # The valley region keeps five pairs and receives equal influence.
        sparse = PairwiseObservations.from_pairs(
            observed.observed_pairs[:5],
            observed.observed_values[:5],
            target=observed.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        problems.append((region, sparse))
    if species is not None:
        region, observed = problems[0]
        problems.append((region, replace(observed, values=observed.values * 1.3)))
    group_count = len(problems)
    groups_species = [0] * region_count if species is None else [0, 1]
    regions = tuple(problem[0] for problem in problems)
    observations = tuple(problem[1] for problem in problems)
    if species is not None:
        regions = (
            replace(
                regions[0],
                sampling_unit_ids=regions[0].sampling_unit_ids * 2,
                grid_positions=np.tile(regions[0].grid_positions, (2, 1)),
                species=(species[0],) * unit_count + (species[1],) * unit_count,
            ),
        )
        observations = (
            tuple(
                replace(observed, species=name)
                for name, observed in zip(species, observations, strict=True)
            ),
        )
    parameter_count = 2 * (1 if species is None else len(species))
    variances_by_group = [
        np.var(np.asarray(observed.observed_values), ddof=1) * np.array([0.25, 0.5])
        for _, observed in problems
    ]
    raw = np.concatenate([np.log(np.expm1(values - 1e-10)) for values in variances_by_group])
    initial = np.concatenate((np.zeros(parameter_count), raw))

    def reference_loss(parameters, indices: Sequence[int] | None = None):
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

    # The reference applies two Adam updates to finite-difference gradients of the
    # dense Gaussian loss.
    expected = initial.copy()
    first_moment, second_moment = np.zeros_like(initial), np.zeros_like(initial)
    for step in (1, 2):
        delta = np.eye(len(initial)) * 1e-5
        gradient = np.array(
            [
                (
                    reference_loss(expected + offset, selections[step - 1])
                    - reference_loss(expected - offset, selections[step - 1])
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
            jnp.zeros(2) if species is None else jnp.zeros((2, 2)), species=species
        ),
        config=TrainingConfig(
            loss=MLPELoss(),
            steps=2,
            optimizer=optax.adam(0.02),
            species_batch_size=species_batch_size,
        ),
    )
    np.testing.assert_allclose(result.history[0].training_loss, reference_loss(initial), atol=1e-12)
    np.testing.assert_allclose(
        result.history[-1].training_loss, reference_loss(expected), atol=1e-8
    )
    assert result.history[-1].training_loss < result.history[0].training_loss
    assert result.step == result.selected_step == 2
    calibration_head = cast(MLPECalibrationHead, result.selected_model.calibration_head)
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
        calibration = calibration_head.for_region(
            region.name, None if species is None else species[groups_species[index]]
        )
        np.testing.assert_allclose(
            [calibration.unit_variance, calibration.residual_variance],
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
        np.testing.assert_allclose(
            [calibration.intercept, calibration.slope], beta, rtol=1e-8, atol=1e-10
        )
        np.testing.assert_allclose(calibration.effect_mean, effect_mean, atol=1e-9)
        np.testing.assert_allclose(calibration.effect_covariance, effect_covariance, atol=1e-10)
        np.testing.assert_allclose(-calibration.ml_log_likelihood, nll, atol=1e-8)
        assert calibration.slope < 0
        assert not np.allclose(variances, variances_by_group[index])
        observed_distances = distances[left, right]
        np.testing.assert_allclose(
            calibration.distance_center, observed_distances.mean(), atol=1e-12
        )
        np.testing.assert_allclose(
            calibration.distance_scale, observed_distances.std(ddof=1), atol=1e-12
        )
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
        species = ("stoat", "hare")
        region = replace(
            region,
            sampling_unit_ids=region.sampling_unit_ids * 2,
            grid_positions=jnp.tile(jnp.asarray(region.grid_positions), (2, 1)),
            species=tuple(name for name in species for _ in region.sampling_unit_ids),
        )
        observed_groups = tuple(
            replace(observations, species=name, values=observations.values * (index + 1))
            for index, name in enumerate(species)
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
            optimizer=optax.adam(0.03),
        )
        runs = [
            train(
                region,
                tuple(group.select_pairs(train_pairs) for group in observed_groups),
                distance_model=WeightedEmbedding(weights, species=species),
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
    assert eqx.tree_equal(first.latest_model.calibration_head, second.latest_model.calibration_head)
    assert first.history[-1].validation_loss != second.history[-1].validation_loss
    assert jnp.issubdtype(first.rng_key.dtype, jax.dtypes.prng_key)
    expected_losses = []
    for index, (name, observed) in enumerate(zip(species, observed_groups, strict=True)):
        distances = np.asarray(
            first.latest_model.distance_model.predict(region, {name: None})[name].as_matrix()
        )
        calibration = cast(MLPECalibrationHead, first.latest_model.calibration_head).for_region(
            region.name, name
        )
        assert set(calibration.calibration_pairs) == set(train_pairs)
        (left, right), _ = observed.select_pairs(train_pairs).aligned_pairs(single_region)
        np.testing.assert_allclose(
            calibration.distance_center, distances[left, right].mean(), atol=1e-12
        )
        np.testing.assert_allclose(
            calibration.distance_scale, distances[left, right].std(ddof=1), atol=1e-12
        )
        (left, right), values = observed.select_pairs(query_pair).aligned_pairs(single_region)
        expected_mean = (
            calibration.intercept
            + calibration.slope
            * (distances[left, right] - calibration.distance_center)
            / calibration.distance_scale
        )
        variance = 2 * calibration.unit_variance + calibration.residual_variance
        expected_losses.append(
            0.5 * (np.log(2 * np.pi * variance) + (values[0] - expected_mean[0]) ** 2 / variance)
        )
    assert first.history[-1].validation_loss is not None
    np.testing.assert_allclose(
        first.history[-1].validation_loss, np.mean(expected_losses), rtol=1e-12
    )


def test_mlpe_conductance_training_improves_through_effective_resistance_layer(
    FeatureConductance, dense_resistance
):

    with jax.enable_x64():
        # Four populations give the same MLPE pair design as the embedding problems.
        features = np.array([[[0.0], [0.2], [0.8]], [[-0.3], [1.4], [-0.7]]])
        ids = tuple(f"population-{i}" for i in range(4))
        region = RegionBatch(
            "graph",
            cx.field(features, "row", "column", cx.LabeledAxis("feature", np.array(("habitat",)))),
            ids,
            np.array([[0, 0], [0, 2], [1, 0], [1, 2]]),
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
            distance_model=FeatureConductance(jnp.asarray(0.1)),
            config=TrainingConfig(
                loss=MLPELoss(),
                steps=20,
                optimizer=optax.adam(0.08),
            ),
        )

        assert result.loss_state is not None
        assert result.history[-1].training_loss < result.history[0].training_loss - 0.2
        assert float(cast(Any, result.selected_model.distance_model).weight) > 0.6
        assert np.isfinite(result.selected_model.predict(region).values).all()
        surface = (
            cast(ConductanceModel, result.selected_model.distance_model)
            .conductance_surface(region)
            .data
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
        constant_region = replace(
            region,
            features=cx.field(
                np.ones_like(region.feature_array),
                "row",
                "column",
                region.features.axes["feature"],
            ),
        )
        with pytest.raises(FloatingPointError):
            train(constant_region, observations, distance_model=model, config=config)


def test_continuation_rejects_changed_optimization_policy(mlpe_embedding_training):
    region, observations, result = (
        mlpe_embedding_training.region,
        mlpe_embedding_training.observations,
        mlpe_embedding_training.state,
    )
    assert result.loss_state is not None
    # The optimizer may change while its state keeps the saved structure.
    continued = train(
        region,
        observations,
        state=result,
        config=replace(result.config, steps=result.step + 1, optimizer=optax.adam(0.1)),
    )
    assert continued.step == result.step + 1
    with pytest.raises(ValueError, match="optimizer"):
        train(
            region,
            observations,
            state=result,
            config=replace(result.config, steps=result.step + 1, optimizer=optax.sgd(0.1)),
        )
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
    ("species", "species_batch_size", "region_batch_size"),
    [
        # A region batch larger than the region count is the full batch.
        (None, None, 3),
        # Sampled regions weight their species groups equally.
        (("frog", "newt"), None, 1),
        # Sampled species within full region batches.
        (("frog", "newt"), 1, None),
    ],
)
def test_shared_embedding_update_matches_independent_equal_region_loss(
    ScalarEmbedding,
    regions_and_observations,
    region_batch_size,
    species,
    species_batch_size,
    monkeypatch,
):
    with jax.enable_x64():
        regions, observations = regions_and_observations()
        if species is not None:
            # Frog occurs in both regions; newt only in the short region.
            regions = [
                replace(regions[0], species=("frog",) * 3),
                replace(
                    regions[1],
                    sampling_unit_ids=regions[1].sampling_unit_ids * 2,
                    grid_positions=np.tile(regions[1].grid_positions, (2, 1)),
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

        def independent_loss(weight, indices: Sequence[int] = (0, 1), short_species=None):
            predictions = [np.array([4]) * weight**2, np.array([1, 4, 1]) * weight**2]
            targets = [np.array([15]), np.array([0.1, 0.1, 0.1])]
            losses = [
                np.mean((np.log1p(p) - np.log1p(t)) ** 2)
                for p, t in zip(predictions, targets, strict=True)
            ]
            if species is not None:
                newt = (np.log1p(4 * weight**2) - np.log1p(3)) ** 2
                short = losses[0] / 2 + newt
                if short_species is not None:
                    short = losses[0] if short_species == 0 else 2 * newt
                losses = [short, losses[1] / 2]
            return np.mean([losses[i] for i in indices])

        step = 1e-5
        dense_gradient = (independent_loss(initial + step) - independent_loss(initial - step)) / (
            2 * step
        )
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
                independent_loss(initial + step, selected, short_species)
                - independent_loss(initial - step, selected, short_species)
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
                    optimizer=optax.adam(0.01),
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
                result.history[0].training_loss, independent_loss(initial), rtol=1e-8
            )
            assert set(result.history[0].training_by_region) == {"short", "wide"}
            for region in regions:
                assert np.isfinite(
                    _single_species_prediction(
                        result.selected_model, region, None if species is None else "frog"
                    ).values
                ).all()
        np.testing.assert_allclose(np.dot(probabilities, gradients), dense_gradient, rtol=1e-8)


@pytest.mark.parametrize("loss_kind", ["direct_log1p", "mlpe"])
def test_species_microbatches_preserve_updates_with_shared_parameters(
    training_problem, SharedParameterEmbedding, dense_profile, loss_kind
):
    species = ("frog", "newt")
    region, observations, _ = training_problem("embedding", species=species)
    initial = np.array([0.6, 0.9])
    model = SharedParameterEmbedding(jnp.asarray(initial), species)
    config = TrainingConfig(
        loss=Log1pLoss() if loss_kind == "direct_log1p" else MLPELoss(), steps=2
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

    def reference_loss(weights):
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
                if loss_kind == "direct_log1p"
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
            (reference_loss(initial + delta) - reference_loss(initial - delta)) / 2e-5
            for delta in differences
        ]
    )
    np.testing.assert_allclose(
        states[0].history[0].training_loss, reference_loss(initial), atol=1e-9
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
                regions[0],
                replace(
                    regions[1],
                    features=cx.field(
                        regions[1].feature_array,
                        "row",
                        "column",
                        cx.LabeledAxis("feature", np.array(["canopy"])),
                    ),
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


def test_validation_on_another_region_requires_matching_feature_names(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    canopy = cx.LabeledAxis("feature", np.array(["canopy"]))
    regions[1] = replace(
        regions[1], features=cx.field(regions[1].feature_array, "row", "column", canopy)
    )
    with pytest.raises(ValueError, match="feature"):
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
    # Validation observations equal the initial predictions; training targets differ.
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
        config=TrainingConfig(loss=Log1pLoss(), steps=5, eval_every=2, optimizer=optax.adam(0.01)),
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
    names = ("frog", "newt")
    region, observations, distance_model = training_problem("embedding", species=names)
    config = TrainingConfig(loss=Log1pLoss(), steps=1)
    result = train(region, observations, distance_model=distance_model, config=config)
    assert result.selected_model.species == result.latest_model.species == names
    assert set(cast(dict, result.history[0].training_by_species)) == set(names)
    continued = train(region, observations, state=result, config=replace(config, steps=2))
    assert continued.selected_model.species == names
    predictions = continued.selected_model.predict(region, dict.fromkeys(names))
    assert [prediction.species for prediction in predictions.values()] == list(names)
    with pytest.raises(ValueError, match="species"):
        train(
            region,
            (observations[0], replace(observations[1], species="toad")),
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
    mlpe_embedding_training, dense_profile
):
    region, observations = mlpe_embedding_training.region, mlpe_embedding_training.observations
    assert [record.step for record in mlpe_embedding_training.state.history] == [0, 3]
    assert [state.step for state in mlpe_embedding_training.callbacks] == [0, 1, 2, 3]
    (left, right), targets = observations.aligned_pairs(region)
    for current in mlpe_embedding_training.callbacks:
        model = current.latest_model
        distances = model.distance_model.predict(region, observations.observed_pairs).values
        calibration = cast(MLPECalibrationHead, model.calibration_head).for_region(region.name)
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


@final
class _SequentialEmbedding(EmbeddingDistanceModel):
    """A per-pixel embedding through a sequential network ending in dropout."""

    network: eqx.nn.Sequential
    species: tuple[str, ...] | None = eqx.field(static=True, default=None)
    patch_size: int = eqx.field(static=True, default=1)

    def backbone(self, features):
        return jax.vmap(jax.vmap(self.network))(features)

    def embedding(self, latent, species_index):
        return latent


def test_training_passes_keys_through_sequential_containers(regions_and_observations):
    regions, observations = regions_and_observations()
    network = eqx.nn.Sequential([eqx.nn.Linear(1, 1, key=jax.random.key(0)), eqx.nn.Dropout(0.2)])
    model = _SequentialEmbedding(network=network)
    state = train(
        regions,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=2, seed=3),
    )
    assert state.step == 2
    assert not eqx.tree_equal(state.latest_model.distance_model, model)


@final
class _DropoutEmbedding(EmbeddingDistanceModel):
    """A scaled embedding of a constant field passed through dropout per pixel or per grid."""

    weight: jax.Array
    dropout: eqx.nn.Dropout
    per_pixel: bool = eqx.field(static=True)
    species: tuple[str, ...] | None = eqx.field(static=True, default=None)
    patch_size: int = eqx.field(static=True, default=1)

    def backbone(self, features):
        ones = jnp.ones_like(features[..., :1])
        if self.per_pixel:
            return jax.vmap(jax.vmap(self.dropout))(ones)
        return self.dropout(ones)

    def embedding(self, latent, species_index):
        return self.weight * latent


@pytest.mark.parametrize("per_pixel", [True, False])
def test_bound_dropout_keys_draw_one_mask_per_layer_call(regions_and_observations, per_pixel):
    # Mapped calls share one mask, so equal inputs stay equal and give zero distances.
    regions, observations = regions_and_observations()
    model = _DropoutEmbedding(
        weight=jnp.asarray(1.0), dropout=eqx.nn.Dropout(0.5), per_pixel=per_pixel
    )
    state = train(
        regions,
        observations,
        distance_model=model,
        config=TrainingConfig(loss=Log1pLoss(), steps=1, seed=0, optimizer=optax.sgd(0.1)),
    )
    weight = cast(Any, state.latest_model.distance_model).weight
    assert (weight == 1.0) == per_pixel


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


def _scalar_region_losses(weight):
    """Returns the log1p losses of the short and wide regions for a scalar embedding weight."""
    predictions = [np.array([4]) * weight**2, np.array([1, 4, 1]) * weight**2]
    targets = [np.array([15]), np.array([0.1, 0.1, 0.1])]
    return [
        np.mean((np.log1p(p) - np.log1p(t)) ** 2) for p, t in zip(predictions, targets, strict=True)
    ]


@pytest.mark.parametrize("optimizer", [optax.adam(0.1), optax.lbfgs()], ids=["adam", "lbfgs"])
def test_initial_optimizer_state_has_the_leaf_types_of_updated_states(
    ScalarEmbedding, regions_and_observations, optimizer
):
    regions, observations = regions_and_observations()
    config = TrainingConfig(loss=Log1pLoss(), steps=1, optimizer=optimizer)
    states = []
    train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0, dtype=jnp.float64)),
        config=config,
        on_step=states.append,
    )
    initial, updated = (
        [(leaf.dtype, leaf.weak_type) for leaf in jax.tree.leaves(state.optimizer_state)]
        for state in states
    )
    assert initial == updated


def test_adam_with_learning_rate_1e_3_is_the_default_optimizer(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(loss=Log1pLoss(), steps=3)
    default, explicit = (
        train(regions, observations, distance_model=model, config=configured)
        for configured in (config, replace(config, optimizer=optax.adam(1e-3)))
    )
    assert eqx.tree_equal(default, explicit)
    assert not default.converged


def test_lbfgs_trains_until_the_gradient_tolerance(ScalarEmbedding, regions_and_observations):
    regions, observations = regions_and_observations()
    config = TrainingConfig(
        loss=Log1pLoss(), steps=100, optimizer=optax.lbfgs(), gradient_tolerance=1e-9
    )
    states = []
    result = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=config,
        on_step=states.append,
    )
    assert result.converged
    assert 0 < result.step < config.steps
    assert [state.step for state in states] == list(range(result.step + 1))
    assert [state.converged for state in states] == [False] * result.step + [True]
    assert eqx.tree_equal(states[-1], result)
    # The final step is evaluated although it is not on the evaluation cadence.
    assert result.history[-1].step == result.selected_step == result.step
    weight = float(cast(Any, result.selected_model.distance_model).weight)

    def mean_loss(value):
        return np.mean(_scalar_region_losses(value))

    np.testing.assert_allclose(
        (mean_loss(weight + 1e-6) - mean_loss(weight - 1e-6)) / 2e-6, 0, atol=1e-7
    )
    np.testing.assert_allclose(result.history[-1].training_loss, mean_loss(weight), rtol=1e-12)
    # Line searches are deterministic: the same configuration reproduces the run.
    repeated = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=config,
    )
    assert eqx.tree_equal(repeated, result)

    capped = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=replace(config, steps=1),
    )
    assert not capped.converged
    assert capped.step == 1

    callbacks = []
    continued = train(
        regions,
        observations,
        state=result,
        config=replace(config, steps=200),
        on_step=callbacks.append,
    )
    assert callbacks == []
    assert continued.converged
    assert continued.step == result.step
    assert eqx.tree_equal(continued.latest_model, result.latest_model)


def test_line_searches_evaluate_the_regions_sampled_for_the_update(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    result = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(
            loss=Log1pLoss(), steps=1, optimizer=optax.lbfgs(), region_batch_size=1
        ),
    )
    weight = float(cast(Any, result.latest_model.distance_model).weight)
    # The line search's accepted loss value at the updated weight covers the one
    # sampled region, reweighted to the scale of the full training loss.
    value = float(cast(Any, optax.tree.get(result.optimizer_state, "value")))
    losses = _scalar_region_losses(weight)
    assert min(abs(value - loss) for loss in losses) < 1e-12
    assert abs(value - np.mean(losses)) > 1e-3


def test_training_checks_the_patch_grid_of_training_and_validation_regions(patch_check_problem):
    problem = patch_check_problem
    config = TrainingConfig(loss=Log1pLoss(), steps=1)
    misaligned, shared = problem.misaligned, problem.shared
    with problem.trim_required():
        train(
            misaligned,
            problem.observations(misaligned),
            distance_model=problem.model,
            config=config,
        )
    with problem.trim_required():
        train(
            shared,
            problem.observations(shared),
            distance_model=problem.model,
            config=config,
            validation=(misaligned, problem.observations(misaligned)),
        )
    observations = problem.observations(shared)
    pairs = observations.observed_pairs
    with problem.shared_vertices_warned():
        state = train(
            shared,
            observations.select_pairs(pairs[:8]),
            distance_model=problem.model,
            config=config,
            validation=(shared, observations.select_pairs(pairs[8:])),
        )
    assert np.isfinite(state.selected_loss)
    # A rebuilt copy of the training region is reported once.
    with problem.shared_vertices_warned():
        train(
            shared,
            observations.select_pairs(pairs[:8]),
            distance_model=problem.model,
            config=replace(config, steps=0),
            validation=(replace(shared), observations.select_pairs(pairs[8:])),
        )


def test_training_objective_equals_the_training_loss_without_a_penalty(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    result = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.1)),
    )
    assert len(result.history) == 3
    for record in result.history:
        assert record.training_objective == record.training_loss


class _RecordedUpdate(NamedTuple):
    """The arguments of an optimizer's latest update."""

    value: jax.Array
    gradient: Any
    shifted_value: jax.Array
    value_gradient: Any


def _recording_optimizer() -> optax.GradientTransformationExtraArgs:
    """Returns an optimizer that keeps the parameters and records each update's arguments.

    Its state holds the update's value and gradient, ``value_fn`` at the parameters
    plus one, and the gradient of ``value_fn`` at the parameters.
    """

    def init(params):
        zeros = jax.tree.map(jnp.zeros_like, params)
        return _RecordedUpdate(jnp.zeros(()), zeros, jnp.zeros(()), zeros)

    def update(updates, state, params, *, value, grad, value_fn, **extra):
        shifted = jax.tree.map(lambda leaf: leaf + 1.0, params)
        recorded = _RecordedUpdate(
            value, grad, value_fn(shifted), eqx.filter_grad(value_fn)(params)
        )
        return jax.tree.map(jnp.zeros_like, updates), recorded

    return optax.GradientTransformationExtraArgs(init, cast(Any, update))


_RECORDING_OPTIMIZER = _recording_optimizer()


def _recorded_update(regions, observations, distance_model, **settings) -> _RecordedUpdate:
    """Returns the optimizer arguments of one update from a distance model."""
    state = train(
        regions,
        observations,
        distance_model=distance_model,
        config=TrainingConfig(
            loss=settings.pop("loss", Log1pLoss()),
            steps=1,
            optimizer=_RECORDING_OPTIMIZER,
            **settings,
        ),
    )
    return cast(_RecordedUpdate, state.optimizer_state)


def test_a_custom_penalty_enters_the_value_the_gradient_and_the_line_search_function(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(2.0))
    plain = _recorded_update(regions, observations, model)
    penalized = _recorded_update(
        regions, observations, model, penalty=lambda distance_model: distance_model.weight**4
    )
    # The two regions hold 3 + 1 pairs. At weight 2, w**4 is 16 and its derivative 32; at
    # the shifted weight 3, w**4 is 81.
    np.testing.assert_allclose(penalized.value - plain.value, 16 / 4, rtol=1e-12)
    np.testing.assert_allclose(
        penalized.gradient[0].weight - plain.gradient[0].weight, 32 / 4, rtol=1e-12
    )
    np.testing.assert_allclose(penalized.shifted_value - plain.shifted_value, 81 / 4, rtol=1e-12)
    # The value that line searches evaluate and the update's gradient belong together.
    np.testing.assert_allclose(
        penalized.value_gradient[0].weight, penalized.gradient[0].weight, rtol=1e-9
    )


def _separation_problem() -> tuple[RegionBatch, PairwiseObservations]:
    """Returns sampling units on both sides of a wall, more dissimilar across it.

    Lower wall conductance always fits the dissimilarities better, so the
    coefficient of the wall feature has no finite optimum under the MLPE loss.
    """
    rng = np.random.default_rng(0)
    wall = np.zeros((6, 7))
    wall[:, 3] = 1.0
    features = np.stack([wall, rng.normal(size=wall.shape)], axis=-1)
    positions = np.array([[0, 0], [2, 1], [4, 0], [5, 2], [0, 5], [2, 6], [4, 5], [5, 4]])
    units = tuple(f"unit-{index}" for index in range(len(positions)))
    region = RegionBatch(
        "valley",
        cx.field(features, "row", "column", cx.LabeledAxis("feature", np.array(["wall", "noise"]))),
        units,
        positions,
    )
    east = positions[:, 1] > 3
    left, right = np.triu_indices(len(units), 1)
    observations = PairwiseObservations.from_pairs(
        [(units[a], units[b]) for a, b in zip(left, right, strict=True)],
        0.1 + 1.0 * (east[left] != east[right]) + rng.normal(scale=0.01, size=len(left)),
        target=TargetSpec("synthetic divergence", units="index"),
        sampling_unit_ids=units,
    )
    return region, observations


_SEPARATION_SOLVER = ResistanceSolverConfig(rtol=1e-10, atol=1e-10, max_steps=5000)


@pytest.mark.parametrize(
    ("where", "selected"),
    [
        (lambda model: model.coefficients, np.array([[1.0, 1.0]])),
        (lambda model: model.coefficients[:, 0], np.array([[1.0, 0.0]])),
    ],
    ids=["coefficients", "one-column"],
)
def test_ridge_gradient_is_the_selected_leaf_over_the_squared_prior_sd_and_the_pair_count(
    where, selected
):
    region, observations = _separation_problem()
    coefficients = jnp.asarray([[0.5, -1.0]])
    model = eqx.tree_at(
        lambda current: (current.coefficients, current.intercepts),
        LogLinearConductance(2, patch_size=1, solver_config=_SEPARATION_SOLVER),
        (coefficients, jnp.asarray([0.3])),
    )
    plain = _recorded_update(region, observations, model, loss=MLPELoss())
    penalized = _recorded_update(
        region,
        observations,
        model,
        loss=MLPELoss(),
        penalty=RidgePenalty(prior_sd=1.5, where=where),
    )
    # Eight sampling units form 28 training pairs.
    np.testing.assert_allclose(
        penalized.gradient[0].coefficients - plain.gradient[0].coefficients,
        selected * np.asarray(coefficients) / (1.5**2 * 28),
        rtol=1e-9,
        atol=1e-14,
    )
    np.testing.assert_array_equal(penalized.gradient[0].intercepts, plain.gradient[0].intercepts)
    assert eqx.tree_equal(penalized.gradient[1], plain.gradient[1])


@pytest.fixture(scope="module")
def shared_ridge_penalty() -> RidgePenalty:
    """Returns one penalty object that the tests share across training sets."""
    return RidgePenalty(prior_sd=0.5, where=lambda model: model.weight)


def _two_species(regions, observations):
    """Returns frog observations in two regions and newt observations in the short region."""
    return (
        [
            replace(regions[0], species=("frog",) * 3),
            replace(
                regions[1],
                sampling_unit_ids=regions[1].sampling_unit_ids * 2,
                grid_positions=np.tile(regions[1].grid_positions, (2, 1)),
                species=("frog", "frog", "newt", "newt"),
            ),
        ],
        [
            replace(observations[0], species="frog"),
            (
                replace(observations[1], species="frog"),
                replace(observations[1], species="newt", values=observations[1].values / 5),
            ),
        ],
    )


@pytest.mark.parametrize(
    ("groups", "pair_count", "settings"),
    [
        ("wide", 3, {}),
        ("short", 1, {}),
        ("both", 4, {}),
        ("species", 5, {}),
        # A sampled update takes the penalty once and divides by every group's pairs.
        (
            "species",
            5,
            {"region_batch_size": 1, "species_batch_size": 1, "species_microbatch_size": 1},
        ),
        ("species", 5, {"species_microbatch_size": 1}),
    ],
    ids=["three-pairs", "one-pair", "two-regions", "two-species", "sampled", "microbatched"],
)
def test_one_penalty_object_divides_by_the_pairs_of_every_training_group(
    ScalarEmbedding, regions_and_observations, groups, pair_count, settings, shared_ridge_penalty
):
    regions, observations = regions_and_observations()
    selected = {
        "wide": (regions[0], observations[0], jnp.asarray(2.0)),
        "short": (regions[1], observations[1], jnp.asarray(2.0)),
        "both": (regions, observations, jnp.asarray(2.0)),
        "species": (*_two_species(regions, observations), jnp.asarray([1.0, 3.0])),
    }
    region, observed, weight = selected[groups]
    species = ("frog", "newt") if groups == "species" else None
    model = ScalarEmbedding(weight, species=species)
    plain = _recorded_update(region, observed, model, **settings)
    penalized = train(
        region,
        observed,
        distance_model=model,
        config=TrainingConfig(
            loss=Log1pLoss(),
            steps=1,
            optimizer=_RECORDING_OPTIMIZER,
            penalty=shared_ridge_penalty,
            **settings,
        ),
    )
    recorded = cast(_RecordedUpdate, penalized.optimizer_state)
    # A prior standard deviation of 0.5 gives 2 * sum(w**2) and the gradient 4 * w.
    penalty_value = 2 * np.sum(np.asarray(weight) ** 2)
    np.testing.assert_allclose(
        recorded.gradient[0].weight - plain.gradient[0].weight,
        4 * np.asarray(weight) / pair_count,
        rtol=1e-12,
    )
    np.testing.assert_allclose(recorded.value - plain.value, penalty_value / pair_count, rtol=1e-12)
    record = penalized.history[-1]
    np.testing.assert_allclose(
        record.training_objective - record.training_loss, penalty_value / pair_count, rtol=1e-9
    )


def test_training_objective_adds_the_penalty_over_the_pairs_to_the_unpenalized_training_loss(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    states = []
    train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(
            loss=Log1pLoss(),
            steps=2,
            optimizer=optax.adam(0.1),
            penalty=RidgePenalty(prior_sd=0.5, where=lambda model: model.weight),
        ),
        on_step=states.append,
    )
    assert [state.history[-1].step for state in states] == [0, 1, 2]
    for state in states:
        weight = float(cast(Any, state.latest_model.distance_model).weight)
        record = state.history[-1]
        np.testing.assert_allclose(
            record.training_loss, np.mean(_scalar_region_losses(weight)), rtol=1e-12
        )
        # A prior standard deviation of 0.5 gives 2 * w**2, over 3 + 1 pairs.
        np.testing.assert_allclose(
            record.training_objective, record.training_loss + 2 * weight**2 / 4, rtol=1e-12
        )


@pytest.mark.parametrize(
    ("penalty", "settings", "completed"),
    [
        (lambda model: jnp.asarray(jnp.nan), {}, []),
        (lambda model: jnp.asarray(jnp.inf), {}, []),
        # Finite at the initial weight only, at evaluations on and off the cadence.
        (lambda model: jnp.where(model.weight == 1.0, 0.0, jnp.inf), {}, [0]),
        (lambda model: jnp.where(model.weight == 1.0, 0.0, jnp.inf), {"eval_every": 2}, [0]),
    ],
    ids=["nan", "inf", "after-an-update", "after-an-update-off-cadence"],
)
def test_a_nonfinite_penalty_at_an_evaluation_raises(
    ScalarEmbedding, regions_and_observations, penalty, settings, completed
):
    regions, observations = regions_and_observations()
    steps = []
    with pytest.raises(FloatingPointError, match="penalty"):
        train(
            regions,
            observations,
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            config=TrainingConfig(
                loss=Log1pLoss(), steps=3, optimizer=optax.adam(0.1), penalty=penalty, **settings
            ),
            on_step=lambda state: steps.append(state.step),
        )
    assert steps == completed


def test_a_nonfinite_penalty_is_reported_at_the_next_evaluation_without_a_callback(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    with pytest.raises(FloatingPointError, match="step 2"):
        train(
            regions,
            observations,
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            config=TrainingConfig(
                loss=Log1pLoss(),
                steps=3,
                eval_every=2,
                optimizer=optax.adam(0.1),
                penalty=lambda model: jnp.where(model.weight == 1.0, 0.0, jnp.inf),
            ),
        )


@pytest.mark.parametrize(
    "penalty",
    [
        lambda model: jnp.ones(2),
        lambda model: jnp.ones(1),
        lambda model: (model.weight,),
        lambda model: None,
        lambda model: "zero",
        lambda model: 1j * model.weight,
    ],
    ids=["vector", "one-element", "tuple", "none", "string", "complex"],
)
def test_a_penalty_that_is_not_a_real_scalar_is_rejected_before_the_first_update(
    ScalarEmbedding, regions_and_observations, penalty
):
    regions, observations = regions_and_observations()
    callbacks = []
    with pytest.raises(ValueError, match="penalty"):
        train(
            regions,
            observations,
            distance_model=ScalarEmbedding(jnp.asarray(1.0)),
            config=TrainingConfig(loss=Log1pLoss(), steps=1, penalty=penalty),
            on_step=callbacks.append,
        )
    assert callbacks == []


def _embedding_regions(embedding_problem):
    """Returns two regions with different pair counts and their observations."""
    alpine, alpine_observations = embedding_problem("alpine", columns=3)
    valley, valley_observations = embedding_problem("valley", offset=0.2, seed=9, columns=2)
    return [alpine, valley], [alpine_observations, valley_observations]


@pytest.mark.parametrize("optimizer", [optax.adam(0.03), optax.lbfgs()], ids=["adam", "lbfgs"])
@pytest.mark.parametrize(
    "penalty",
    [
        lambda model: jnp.zeros(()),
        lambda model: 0.0,
        lambda model: 0,
        RidgePenalty(prior_sd=1.0, where=lambda model: ()),
    ],
    ids=["array", "float", "integer", "empty-ridge"],
)
def test_a_zero_penalty_and_an_absent_penalty_give_identical_trajectories(
    WeightedEmbedding, embedding_problem, optimizer, penalty
):
    regions, observations = _embedding_regions(embedding_problem)
    model = WeightedEmbedding(jnp.zeros(2))
    config = TrainingConfig(loss=MLPELoss(), steps=3, optimizer=optimizer)
    absent = train(regions, observations, distance_model=model, config=config)
    zero = train(
        regions, observations, distance_model=model, config=replace(config, penalty=penalty)
    )
    assert absent.step == 3
    assert eqx.tree_equal(replace(zero, config=absent.config), absent)


def test_continuation_keeps_the_penalty_and_rejects_a_changed_one(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()

    def where(model):
        return model.weight

    penalty = RidgePenalty(prior_sd=0.5, where=where)
    model = ScalarEmbedding(jnp.asarray(1.0))
    config = TrainingConfig(loss=Log1pLoss(), steps=1, optimizer=optax.adam(0.1), penalty=penalty)
    state = train(regions, observations, distance_model=model, config=config)
    continued = train(regions, observations, state=state, config=replace(state.config, steps=2))
    assert continued.step == 2
    assert continued.config.penalty is penalty
    uninterrupted = train(
        regions, observations, distance_model=model, config=replace(config, steps=2)
    )
    assert eqx.tree_equal(continued, uninterrupted)
    # A separately built penalty with the same selector and strength is unchanged.
    assert (
        train(
            regions,
            observations,
            state=state,
            config=replace(config, steps=2, penalty=RidgePenalty(0.5, where)),
        ).step
        == 2
    )
    for changed in (RidgePenalty(1.0, where), RidgePenalty(0.5, lambda model: model.weight), None):
        with pytest.raises(ValueError, match="Continuation"):
            train(
                regions,
                observations,
                state=state,
                config=replace(config, steps=2, penalty=changed),
            )
    unpenalized = train(
        regions, observations, distance_model=model, config=replace(config, penalty=None)
    )
    with pytest.raises(ValueError, match="Continuation"):
        train(regions, observations, state=unpenalized, config=replace(config, steps=2))


def test_validation_losses_and_the_selected_model_ignore_the_penalty(
    ScalarEmbedding, regions_and_observations
):
    regions, observed = regions_and_observations()
    region, observations = regions[0], observed[0]
    model = ScalarEmbedding(jnp.asarray(1.0))
    # Validation observations equal the initial predictions; training targets differ.
    initial = model.predict(region)
    validation = (
        replace(region, name="independent-validation-region"),
        PairwiseObservations.from_pairs(initial.pairs, initial.values, target=observations.target),
    )
    config = TrainingConfig(loss=Log1pLoss(), steps=3, optimizer=optax.adam(0.01))
    plain = train(region, observations, distance_model=model, config=config, validation=validation)
    # A constant penalty changes the training objective and leaves every update unchanged.
    constant = train(
        region,
        observations,
        distance_model=model,
        config=replace(config, penalty=lambda model: jnp.asarray(1000.0)),
        validation=validation,
    )
    assert [record.validation_loss for record in constant.history] == [
        record.validation_loss for record in plain.history
    ]
    assert [record.training_loss for record in constant.history] == [
        record.training_loss for record in plain.history
    ]
    for record in constant.history:
        np.testing.assert_allclose(
            record.training_objective, record.training_loss + 1000 / 3, rtol=1e-12
        )
    assert constant.selected_step == plain.selected_step == 0
    assert constant.selected_loss == plain.selected_loss
    assert eqx.tree_equal(constant.selected_model, plain.selected_model)

    # Under a ridge penalty, each validation loss is the unpenalized loss of its model.
    states = []
    ridge = train(
        region,
        observations,
        distance_model=model,
        config=replace(config, penalty=RidgePenalty(0.1, lambda model: model.weight)),
        validation=validation,
        on_step=states.append,
    )
    for state in states:
        frozen = train(
            region,
            observations,
            distance_model=state.latest_model.distance_model,
            config=replace(config, steps=0),
            validation=validation,
        )
        assert state.history[-1].validation_loss == frozen.history[0].validation_loss
    validation_losses = [cast(float, record.validation_loss) for record in ridge.history]
    assert ridge.selected_step == int(np.argmin(validation_losses))
    assert ridge.selected_loss == min(validation_losses)


def test_the_selected_loss_without_validation_is_the_unpenalized_training_loss(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    result = train(
        regions,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(
            loss=Log1pLoss(),
            steps=2,
            optimizer=optax.adam(0.1),
            penalty=RidgePenalty(prior_sd=0.5, where=lambda model: model.weight),
        ),
    )
    record = result.history[-1]
    assert result.selected_loss == record.training_loss
    assert record.training_objective > record.training_loss


def test_the_gradient_tolerance_tests_the_gradient_norm_with_the_penalty(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    model = ScalarEmbedding(jnp.asarray(1.0))
    recorded = _recorded_update(regions, observations, model)
    # A learning rate of zero keeps the weight; only the penalty changes the gradient norm.
    config = TrainingConfig(
        loss=Log1pLoss(),
        steps=2,
        optimizer=optax.sgd(0.0),
        gradient_tolerance=1.5 * abs(float(recorded.gradient[0].weight)),
    )
    plain = train(regions, observations, distance_model=model, config=config)
    assert plain.converged and plain.step == 1
    penalized = train(
        regions,
        observations,
        distance_model=model,
        config=replace(config, penalty=lambda model: 100 * model.weight**2),
    )
    assert not penalized.converged and penalized.step == 2


@final
class _CenteredPenalty(eqx.Module):
    """A penalty holding an array: the squared distance of the weight from a center."""

    center: jax.Array

    def __call__(self, distance_model):
        return jnp.sum((distance_model.weight - self.center) ** 2)


@pytest.mark.parametrize("optimizer", [optax.adam(0.1), optax.lbfgs()], ids=["adam", "lbfgs"])
def test_a_penalty_module_holding_arrays_trains_and_compares_by_value(
    ScalarEmbedding, regions_and_observations, optimizer
):
    regions, observations = regions_and_observations()
    config = TrainingConfig(
        loss=Log1pLoss(), steps=1, optimizer=optimizer, penalty=_CenteredPenalty(jnp.asarray(0.5))
    )
    state = train(
        regions, observations, distance_model=ScalarEmbedding(jnp.asarray(1.0)), config=config
    )
    weight = float(cast(Any, state.latest_model.distance_model).weight)
    record = state.history[-1]
    np.testing.assert_allclose(
        record.training_objective, record.training_loss + (weight - 0.5) ** 2 / 4, rtol=1e-12
    )
    same = replace(config, steps=2, penalty=_CenteredPenalty(jnp.asarray(0.5)))
    assert train(regions, observations, state=state, config=same).step == 2
    with pytest.raises(ValueError, match="Continuation"):
        train(
            regions,
            observations,
            state=state,
            config=replace(same, penalty=_CenteredPenalty(jnp.asarray(0.6))),
        )


def test_a_penalty_on_every_parameter_trains_a_distance_model_with_random_layers(training_problem):
    region, observations, model = training_problem("dropout")
    penalty = RidgePenalty(
        prior_sd=1.0, where=lambda current: eqx.filter(current, eqx.is_inexact_array)
    )
    states = []
    train(
        region,
        observations,
        distance_model=model,
        config=TrainingConfig(
            loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.002), penalty=penalty
        ),
        on_step=states.append,
    )
    for state in states:
        parameters = jax.tree.leaves(
            eqx.filter(state.latest_model.distance_model, eqx.is_inexact_array)
        )
        expected = sum(float(np.sum(np.asarray(leaf) ** 2)) for leaf in parameters) / 2
        record = state.history[-1]
        pair_count = len(observations.observed_pairs)
        np.testing.assert_allclose(
            record.training_objective, record.training_loss + expected / pair_count, rtol=1e-9
        )


def test_reusing_one_penalty_object_reuses_the_compiled_update(
    ScalarEmbedding, regions_and_observations
):
    regions, observations = regions_and_observations()
    traces = []

    def penalty(model):
        traces.append(None)
        return jnp.sum(model.weight**2)

    config = TrainingConfig(loss=Log1pLoss(), steps=2, optimizer=optax.adam(0.1), penalty=penalty)
    model = ScalarEmbedding(jnp.asarray(1.0))
    train(regions, observations, distance_model=model, config=config)
    assert len(traces) > 1
    traces.clear()
    train(regions, observations, distance_model=model, config=config)
    # Only the check of the penalty's output shape runs the penalty again.
    assert len(traces) == 1


def test_lbfgs_with_a_ridge_penalty_converges_where_the_coefficients_run_away_without_it():
    region, observations = _separation_problem()
    model = LogLinearConductance(2, patch_size=1, solver_config=_SEPARATION_SOLVER)
    config = TrainingConfig(loss=MLPELoss(), steps=30, optimizer=optax.lbfgs())
    plain = train(region, observations, distance_model=model, config=config)
    plain_coefficients = np.asarray(cast(Any, plain.latest_model.distance_model).coefficients)
    assert np.abs(plain_coefficients).max() > 15

    penalized = train(
        region,
        observations,
        distance_model=model,
        config=replace(
            config,
            steps=100,
            gradient_tolerance=1e-7,
            penalty=RidgePenalty(prior_sd=1.0, where=lambda model: model.coefficients),
        ),
    )
    assert penalized.converged
    assert penalized.step < penalized.config.steps
    coefficients = np.asarray(cast(Any, penalized.latest_model.distance_model).coefficients)
    assert np.isfinite(coefficients).all()
    assert np.abs(coefficients).max() < 10
    # L-BFGS's line search decreases the training objective and accepts its value and gradient.
    objectives = [record.training_objective for record in penalized.history]
    assert np.diff(objectives).max() <= 1e-8
    state = penalized.optimizer_state
    np.testing.assert_allclose(cast(Any, optax.tree.get(state, "value")), objectives[-1], rtol=1e-8)
    assert optax.tree.norm(cast(Any, optax.tree.get(state, "grad"))) < 1e-6
    assert penalized.history[-1].training_loss < objectives[-1]
