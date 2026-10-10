"""Independent fold fits and original-target-scale ensemble aggregation."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from test_mlpe_training import embedding_problem

from ilg_toolkit import (
    CalibratedModel,
    EnsembleMember,
    FitResult,
    TrainingConfig,
    TrainingState,
    fit_ensemble,
    generate_population_folds,
)
from ilg_toolkit.models import ConductanceModel, EmbeddingDistanceModel


@final
class InitializedEmbedding(EmbeddingDistanceModel):
    log_weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features * jnp.exp(self.log_weights)


def model_factory(key):
    return InitializedEmbedding(log_weights=jax.random.normal(key, (2,)) * 0.1)


def completed_fit(member: EnsembleMember) -> tuple[CalibratedModel, FitResult, TrainingState]:
    assert member.status == "completed"
    assert member.model is not None
    assert member.fit_result is not None
    assert member.fit_result.state is not None
    return member.model, member.fit_result, member.fit_result.state


def test_public_fold_ensemble_fits_independent_members_and_averages_calibrated_predictions():
    with jax.enable_x64():
        region, observations = embedding_problem()
        folds = generate_population_folds(region, observations, n_folds=2, holdout_size=2, seed=47)
        result = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13, 29),
            config=TrainingConfig(objective="mlpe", epochs=3, learning_rate=0.03),
            model_factory=model_factory,
        )
        prediction = result.predict(region)
        fits = [completed_fit(member) for member in result.members]
        member_values = np.stack([model.predict(region).values for model, _, _ in fits])
    assert len(result.members) == 4
    assert all(member.status == "completed" for member in result.members)
    assert len({member.identity.member_id for member in result.members}) == 4
    assert len({member.identity.effective_seed for member in result.members}) == 4
    assert len({id(state.optimizer_state) for _, _, state in fits}) == 4
    encoders = [model.encoder for model, _, _ in fits]
    assert all(isinstance(encoder, InitializedEmbedding) for encoder in encoders)
    weights = []
    for encoder in encoders:
        assert isinstance(encoder, InitializedEmbedding)
        weights.append(tuple(np.asarray(encoder.log_weights)))
    assert len(set(weights)) == 4
    np.testing.assert_allclose(prediction.values, member_values.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(prediction.member_spread, member_values.std(axis=0), atol=1e-12)
    np.testing.assert_array_equal(prediction.member_values, member_values)
    assert prediction.target.units == "index"
    for member in result.members:
        model, fitted, _ = completed_fit(member)
        assert fitted.history[-1].training_loss < fitted.history[0].training_loss
        head = model.calibrations[region.name]
        heldout = set(member.fold.held_out_units[region.name])
        assert not any(heldout & set(pair) for pair in head.calibration_pairs)
        assert model.validation_pairs == {}
        assert not any(heldout & set(pair) for pair in model.training_pairs[region.name])


def test_generated_folds_are_identity_stable_and_query_values_cannot_change_members():
    from ilg_toolkit import PairwiseObservations

    with jax.enable_x64():
        region, observations = embedding_problem()
        other, other_observations = embedding_problem("valley", seed=71)
        first = generate_population_folds(
            [region, other],
            [observations, other_observations],
            n_folds=2,
            holdout_size={region.name: 2, other.name: 2},
            seed=23,
        )
        reordered = generate_population_folds(
            [other, region],
            [other_observations, observations],
            n_folds=2,
            holdout_size={other.name: 2, region.name: 2},
            seed=23,
        )
        assert first == reordered
        fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=23)[
            0
        ]
        changed_values = observations.observed_values.copy()
        for i, pair in enumerate(observations.observed_pairs):
            if set(pair) & set(fold.held_out_units[region.name]):
                changed_values[i] += 100
        changed = PairwiseObservations.from_pairs(
            observations.observed_pairs,
            changed_values,
            target=observations.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        config = TrainingConfig(objective="mlpe", epochs=2)
        baseline = fit_ensemble(
            region, observations, folds=[fold], config=config, model_factory=model_factory
        )
        perturbed = fit_ensemble(
            region, changed, folds=[fold], config=config, model_factory=model_factory
        )
        np.testing.assert_array_equal(
            baseline.predict(region).values, perturbed.predict(region).values
        )
    baseline_model, baseline_fit, _ = completed_fit(baseline.members[0])
    perturbed_model, perturbed_fit, _ = completed_fit(perturbed.members[0])
    assert baseline_fit.history == perturbed_fit.history
    assert baseline_model.calibrations == perturbed_model.calibrations


def test_original_scale_known_outputs_are_averaged_after_inverse_transformation():
    from ilg_toolkit import Prediction, TargetSpec, aggregate_predictions

    target = TargetSpec("divergence", units="index", transform="log1p")
    transformed = [
        np.array([[0, np.log(2)], [np.log(2), 0]]),
        np.array([[0, np.log(10)], [np.log(10), 0]]),
    ]
    with jax.enable_x64():
        predictions = {
            str(i): Prediction(jnp.asarray(np.expm1(value)), ("a", "b"), target)
            for i, value in enumerate(transformed)
        }
        result = aggregate_predictions(predictions, expected_member_ids=("0", "1"))
    assert all(
        isinstance(value, jax.Array) and value.dtype == jnp.float64
        for value in (result.values, result.member_values, result.member_spread)
    )
    np.testing.assert_allclose(result.values, [[0, 5], [5, 0]], atol=1e-14)
    np.testing.assert_allclose(result.member_spread, [[0, 4], [4, 0]], atol=1e-14)
    assert not np.allclose(result.values, np.expm1(np.mean(transformed, axis=0)))
    assert result.scale == "original"
    assert result.target == target


def test_original_scale_aggregation_preserves_large_finite_means_and_spreads():
    from ilg_toolkit import Prediction, TargetSpec, aggregate_predictions

    with jax.enable_x64():
        target = TargetSpec("signed synthetic measurement")
        for levels, expected_mean, expected_spread in [
            ((1e308, 1e308), 1e308, 0),
            ((-1e308, 1e308), 0, 1e308),
        ]:
            predictions = {
                str(i): Prediction(jnp.asarray([level]), ("a",), target)
                for i, level in enumerate(levels)
            }
            result = aggregate_predictions(predictions)
            np.testing.assert_allclose(result.values, [expected_mean], rtol=1e-14)
            np.testing.assert_allclose(result.member_spread, [expected_spread], rtol=1e-14)


def test_failed_members_and_missing_composition_cannot_be_silently_dropped():
    import pytest

    from ilg_toolkit import Ensemble

    region, observations = embedding_problem()
    outcomes = []
    calls = []

    def partly_broken_factory(key):
        calls.append(key)
        if len(calls) == 2:
            raise ArithmeticError("deliberate initialization failure")
        return model_factory(key)

    result = fit_ensemble(
        region,
        observations,
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(1, 2, 3),
        config=TrainingConfig(epochs=0),
        model_factory=partly_broken_factory,
        on_member=outcomes.append,
    )
    assert [member.status for member in result.members] == ["completed", "failed", "completed"]
    assert outcomes == list(result.members)
    failed = result.members[1]
    assert result.failures == {failed.identity.member_id: failed.failure}
    assert failed.failure is not None
    assert failed.failure.stage == "initialization"
    assert failed.failure.error_type == "ArithmeticError"
    assert "deliberate" in failed.failure.message
    with pytest.raises(RuntimeError, match="unavailable members.*failed"):
        result.predict(region)
    with pytest.raises(ValueError, match="composition cannot shrink"):
        Ensemble((result.members[0], result.members[2]), result.expected_member_ids)


def test_explicit_validation_retains_selection_target_access_on_every_member():
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    with jax.enable_x64():
        region, observations = embedding_problem()
        fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)[
            0
        ]
        # Explicitly reuse held-out observations for selection. Later OOF
        # eligibility must see this target access even though fold.query remains.
        validation = {
            region.name: ObservationPartition(
                region.name, fold.query[region.name].pairs, "validation"
            )
        }
        fold = replace(fold, validation=validation)
        result = fit_ensemble(
            region,
            observations,
            folds=[fold],
            config=TrainingConfig(objective="mlpe", epochs=2),
            model_factory=model_factory,
        )
    member = result.members[0]
    model, fitted, _ = completed_fit(member)
    assert model.validation_pairs[region.name] == validation[region.name].pairs
    assert fitted.selection == "validation"
    assert len(model.calibrations[region.name].calibration_pairs) == 6


def test_one_member_continuation_matches_uninterrupted_fit_and_does_not_reinitialize():
    from dataclasses import replace

    from ilg_toolkit import fit_ensemble_member

    with jax.enable_x64():
        region, observations = embedding_problem()
        fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)[
            0
        ]
        config = TrainingConfig(objective="mlpe", epochs=4)
        complete = fit_ensemble_member(
            region,
            observations,
            fold=fold,
            initialization_seed=31,
            config=config,
            model_factory=model_factory,
        )
        partial = fit_ensemble_member(
            region,
            observations,
            fold=fold,
            initialization_seed=31,
            config=replace(config, epochs=1),
            model_factory=model_factory,
        )

        def cannot_initialize(key):
            raise AssertionError("continuation must use checkpoint encoder")

        _, _, partial_state = completed_fit(partial)
        resumed = fit_ensemble_member(
            region,
            observations,
            fold=fold,
            initialization_seed=31,
            config=config,
            model_factory=cannot_initialize,
            state=partial_state,
        )
        assert resumed.status == complete.status == "completed"
        resumed_model, resumed_fit, resumed_state = completed_fit(resumed)
        complete_model, complete_fit, complete_state = completed_fit(complete)
        np.testing.assert_array_equal(
            resumed_model.predict(region).values, complete_model.predict(region).values
        )
    assert resumed_fit.history == complete_fit.history
    np.testing.assert_array_equal(resumed_state.rng_key, complete_state.rng_key)


def test_incompatible_scales_and_invalid_fold_membership_fail_clearly():
    from dataclasses import replace

    import pytest

    from ilg_toolkit import (
        ObservationPartition,
        PopulationFold,
        Prediction,
        TargetSpec,
        aggregate_predictions,
    )

    target = TargetSpec("divergence", units="index")
    first = Prediction(jnp.zeros((2, 2)), ("a", "b"), target)
    for second in [
        replace(first, target=replace(target, units="percent")),
        replace(first, scale="model"),
        replace(first, sampling_unit_ids=("b", "a")),
    ]:
        with pytest.raises(ValueError, match="compatible.*scales|align sampling-unit"):
            aggregate_predictions({"first": first, "second": second})
    with pytest.raises(ValueError, match="composition cannot shrink"):
        aggregate_predictions({"first": first}, expected_member_ids=("first", "absent"))
    with pytest.raises(ValueError, match="held-out endpoints"):
        PopulationFold(
            "invalid",
            {"alpine": ("a",)},
            {"alpine": ObservationPartition("alpine", (("a", "b"),), "training")},
        )
    region, observations = embedding_problem()
    fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2)[0]
    with pytest.raises(ValueError, match="fold_id.*unique"):
        fit_ensemble(region, observations, folds=[fold, fold])
    with pytest.raises(ValueError, match="initialization_seeds.*unique"):
        fit_ensemble(region, observations, folds=[fold], initialization_seeds=(1, 1))


def test_shared_region_members_fit_one_encoder_with_distinct_regional_calibrations():
    with jax.enable_x64():
        pairs = (embedding_problem(), embedding_problem("valley", offset=3, seed=71))
        regions, observations = zip(*pairs, strict=True)
        result = fit_ensemble(
            regions,
            observations,
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            config=TrainingConfig(objective="mlpe", epochs=2),
            model_factory=model_factory,
        )
    member = result.members[0]
    model, fitted, state = completed_fit(member)
    assert fitted.region_names == ("alpine", "valley")
    assert set(model.calibrations) == {"alpine", "valley"}
    assert state.raw_variances is not None and state.raw_variances.shape == (2, 2)
    assert fitted.history[-1].training_loss < fitted.history[0].training_loss
    assert model.calibrations["valley"].intercept - model.calibrations["alpine"].intercept > 2


@final
class ConstantConductance(ConductanceModel):
    level: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def conductance(self, features, *, patch_batch_size=None):
        return jnp.ones(features.shape[:2]) * self.level


def test_known_graph_outputs_keep_surface_mean_separate_from_distance_mean():
    from ilg_toolkit import (
        CalibratedModel,
        Ensemble,
        EnsembleMember,
        ObservationPartition,
        PopulationFold,
        RegionBatch,
        TargetSpec,
        ensemble_member_identity,
    )

    with jax.enable_x64():
        region = RegionBatch(
            "graph",
            np.ones((1, 4, 1)),
            ("a", "b", "c", "d"),
            np.array([[0, 0], [0, 1], [0, 2], [0, 3]]),
        )
        members = []
        for i, level in enumerate((1.0, 4.0)):
            fold = PopulationFold(
                f"fold-{i}",
                {"graph": ("c", "d")},
                {"graph": ObservationPartition("graph", (("a", "b"),), "training")},
                {"graph": ObservationPartition("graph", (("c", "d"),), "query")},
            )
            model = CalibratedModel(
                ConstantConductance(level=jnp.asarray(level)), TargetSpec("divergence"), 1
            )
            members.append(
                EnsembleMember(ensemble_member_identity(fold.fold_id, 0), fold, "completed", model)
            )
        ensemble = Ensemble(tuple(members))
        prediction = ensemble.predict(region)
        summary = ensemble.conductance_surfaces(region)
        averaged_surface_model = CalibratedModel(
            ConstantConductance(level=jnp.asarray(2.5)), TargetSpec("divergence"), 1
        )
        surface_prediction = averaged_surface_model.predict(region)
    np.testing.assert_allclose(prediction.values[0, 1], 0.625, atol=1e-7)
    np.testing.assert_allclose(summary.values, 2.5, atol=1e-12)
    assert all(
        isinstance(value, jax.Array) and value.dtype == jnp.float64
        for value in (summary.values, summary.member_values, summary.member_spread)
    )
    np.testing.assert_allclose(summary.member_spread, 1.5, atol=1e-12)
    np.testing.assert_allclose(surface_prediction.values[0, 1], 0.4, atol=1e-7)
    assert not np.allclose(prediction.values, surface_prediction.values)


def test_member_callback_failure_interrupts_before_starting_another_member():
    import pytest

    region, observations = embedding_problem()
    initialized = []

    def tracked_factory(key):
        initialized.append(key)
        return model_factory(key)

    def interrupted_save(member):
        assert member.status == "completed"
        raise OSError("interrupted member save")

    with pytest.raises(OSError, match="interrupted member save"):
        fit_ensemble(
            region,
            observations,
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            initialization_seeds=(1, 2),
            config=TrainingConfig(epochs=0),
            model_factory=tracked_factory,
            on_member=interrupted_save,
        )
    assert len(initialized) == 1


def test_epoch_hook_interrupts_and_member_states_continue_the_exact_ensemble():
    import pytest

    with jax.enable_x64():
        region, observations = embedding_problem()
        folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
        config = TrainingConfig(objective="mlpe", epochs=3)
        complete = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13, 29),
            config=config,
            model_factory=model_factory,
        )
        checkpoints = {}
        completed_members = []

        def stop_after_one_update(identity, state):
            checkpoints[identity.member_id] = state
            if state.epoch == 1:
                raise RuntimeError("stop epoch for checkpoint")

        with pytest.raises(RuntimeError, match="stop epoch for checkpoint"):
            fit_ensemble(
                region,
                observations,
                folds=folds,
                initialization_seeds=(13, 29),
                config=config,
                model_factory=model_factory,
                on_epoch=stop_after_one_update,
                on_member=completed_members.append,
            )
        assert completed_members == []
        assert len(checkpoints) == 1
        resumed = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13, 29),
            config=config,
            model_factory=model_factory,
            member_states=checkpoints,
        )
        np.testing.assert_array_equal(
            resumed.predict(region).values, complete.predict(region).values
        )
    assert [completed_fit(member)[1].history for member in resumed.members] == [
        completed_fit(member)[1].history for member in complete.members
    ]
