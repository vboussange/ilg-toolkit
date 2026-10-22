"""Scientific behavior of ensemble."""

from typing import Any, cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    DirectCalibrationModule,
    EnsembleMember,
    LandscapeGeneticsModel,
    MLPECalibrationModule,
    TrainingConfig,
    fit_ensemble,
    fit_ensemble_member,
    generate_population_folds,
)


def test_declared_seeds_initialize_readable_members_directly(
    completed_fit, embedding_problem, model_factory
):
    region, observations = embedding_problem()
    folds = generate_population_folds(region, observations, n_folds=2, holdout_size=2, seed=47)
    result = fit_ensemble(
        region,
        observations,
        folds=folds,
        initialization_seeds=(13, 29),
        config=TrainingConfig(objective="mlpe", epochs=0),
        model_factory=model_factory,
    )
    assert result.expected_member_ids == (
        "fold-0000:seed=13",
        "fold-0000:seed=29",
        "fold-0001:seed=13",
        "fold-0001:seed=29",
    )
    for member in result.members:
        model, state = completed_fit(member)
        expected = model_factory(jax.random.PRNGKey(member.identity.initialization_seed))
        assert eqx.tree_equal(model.distance_model, expected)
        assert state.config.seed == member.identity.initialization_seed
    assert len({id(completed_fit(member)[1].optimizer_state) for member in result.members}) == 4


def test_fresh_ensemble_requires_factory_and_explicit_training_configuration(
    embedding_problem, model_factory
):
    region, observations = embedding_problem()
    fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2)[0]
    config = TrainingConfig(objective="mlpe", epochs=0)
    with pytest.raises(TypeError):
        fit_ensemble(  # pyright: ignore[reportCallIssue]
            region, observations, folds=[fold], model_factory=model_factory
        )
    with pytest.raises(ValueError):
        fit_ensemble(region, observations, folds=[fold], config=config)
    with pytest.raises(ValueError):
        fit_ensemble_member(region, observations, fold=fold, initialization_seed=13, config=config)
    with pytest.raises(ValueError):
        fit_ensemble_member(
            region, observations, fold=fold, initialization_seed=13, model_factory=model_factory
        )


def test_factory_cannot_silently_select_a_builtin_model(embedding_problem):
    region, observations = embedding_problem()
    result = fit_ensemble(
        region,
        observations,
        n_folds=1,
        holdout_size=2,
        config=TrainingConfig(objective="direct_log1p", epochs=0),
        model_factory=lambda key: None,  # pyright: ignore[reportArgumentType]
    )
    member = result.members[0]
    assert member.status == "failed"
    assert member.failure is not None and member.failure.stage == "initialization"
    assert member.failure.error_type == "TypeError"


def test_public_fold_ensemble_fits_independent_members_and_averages_calibrated_predictions(
    completed_fit, embedding_problem, model_factory
):
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
        member_values = np.stack([model.predict(region).values for model, _ in fits])
    assert len(result.members) == 4
    assert all(member.status == "completed" for member in result.members)
    assert len({member.identity.member_id for member in result.members}) == 4
    assert {member.identity.initialization_seed for member in result.members} == {13, 29}
    assert len({id(state.optimizer_state) for _, state in fits}) == 4
    encoders = [model.distance_model for model, _ in fits]
    weights = []
    for encoder in encoders:
        weights.append(tuple(np.asarray(cast(Any, encoder).log_weights)))
    assert len(set(weights)) == 4
    np.testing.assert_allclose(prediction.prediction.values, member_values.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(prediction.member_spread, member_values.std(axis=0), atol=1e-12)
    np.testing.assert_array_equal(prediction.member_values, member_values)
    assert prediction.prediction.target.units == "index"
    for member in result.members:
        model, fitted = completed_fit(member)
        assert fitted.history[-1].training_loss < fitted.history[0].training_loss
        head = cast(MLPECalibrationModule, model.calibration).regions[region.name]
        heldout = set(member.fold.held_out_units[region.name])
        assert not any(heldout & set(pair) for pair in head.calibration_pairs)
        assert model.validation_pairs == {}
        assert not any(heldout & set(pair) for pair in model.training_pairs[region.name])


def test_generated_folds_are_identity_stable_and_query_values_cannot_change_members(
    completed_fit, embedding_problem, model_factory
):
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
        changed_values = np.asarray(observations.observed_values).copy()
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
            baseline.predict(region).prediction.values, perturbed.predict(region).prediction.values
        )
    baseline_model, baseline_fit = completed_fit(baseline.members[0])
    perturbed_model, perturbed_fit = completed_fit(perturbed.members[0])
    assert baseline_fit.history == perturbed_fit.history
    assert eqx.tree_equal(
        cast(MLPECalibrationModule, baseline_model.calibration).regions,
        cast(MLPECalibrationModule, perturbed_model.calibration).regions,
    )


def test_original_scale_known_outputs_are_averaged_after_inverse_transformation():
    from ilg_toolkit import Prediction, TargetSpec, aggregate_predictions

    target = TargetSpec("divergence", units="index", transform="log1p")
    transformed = [np.log(2), np.log(10)]
    with jax.enable_x64():
        predictions = {
            str(i): Prediction(
                jnp.asarray([np.expm1(value)]), (("b", "a"),), "alpine", target, ("a", "b", "extra")
            )
            for i, value in enumerate(transformed)
        }
        result = aggregate_predictions(predictions, expected_member_ids=("0", "1"))
    assert all(
        value.dtype == jnp.float64
        for value in (result.prediction.values, result.member_values, result.member_spread)
    )
    np.testing.assert_allclose(result.prediction.values, [5], atol=1e-14)
    np.testing.assert_allclose(result.member_spread, [4], atol=1e-14)
    np.testing.assert_allclose(result.member_values, [[1], [9]], atol=1e-14)
    np.testing.assert_allclose(
        result.prediction.as_matrix(), [[0, 5, np.nan], [5, 0, np.nan], [np.nan, np.nan, 0]]
    )
    assert result.prediction.pairs == (("b", "a"),)
    assert result.prediction.region_name == "alpine"
    assert not np.allclose(result.prediction.values, np.expm1(np.mean(transformed)))
    assert result.prediction.scale == "original"
    assert result.prediction.target == target


def test_original_scale_aggregation_preserves_large_finite_means_and_spreads():
    from ilg_toolkit import Prediction, TargetSpec, aggregate_predictions

    with jax.enable_x64():
        target = TargetSpec("signed synthetic measurement")
        for levels, expected_mean, expected_spread in [
            ((1e308, 1e308), 1e308, 0),
            ((-1e308, 1e308), 0, 1e308),
        ]:
            predictions = {
                str(i): Prediction(
                    jnp.asarray([level]), (("a", "b"),), "alpine", target, ("a", "b")
                )
                for i, level in enumerate(levels)
            }
            result = aggregate_predictions(predictions)
            np.testing.assert_allclose(result.prediction.values, [expected_mean], rtol=1e-14)
            np.testing.assert_allclose(result.member_spread, [expected_spread], rtol=1e-14)


def test_failed_members_and_missing_composition_cannot_be_silently_dropped(
    embedding_problem, model_factory
):
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
        config=TrainingConfig(objective="direct_log1p", epochs=0),
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
    with pytest.raises(RuntimeError):
        result.predict(region)
    with pytest.raises(ValueError):
        Ensemble((result.members[0], result.members[2]), result.expected_member_ids)


def test_explicit_validation_retains_selection_target_access_on_every_member(
    completed_fit, embedding_problem, model_factory
):
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
    model, fitted = completed_fit(member)
    assert model.validation_pairs[region.name] == validation[region.name].pairs
    assert fitted.selection == "validation"
    assert (
        len(cast(MLPECalibrationModule, model.calibration).regions[region.name].calibration_pairs)
        == 6
    )


def test_incompatible_scales_and_invalid_fold_membership_fail_clearly(
    embedding_problem, model_factory
):
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
    first = Prediction(jnp.zeros(1), (("a", "b"),), "alpine", target, ("a", "b"))
    with pytest.raises(ValueError):
        replace(first, scale="model")
    for second in [
        replace(first, target=replace(target, units="percent")),
        replace(first, sampling_unit_ids=("b", "a")),
        replace(first, pairs=(("b", "a"),)),
        replace(first, region_name="other"),
    ]:
        with pytest.raises(ValueError):
            aggregate_predictions({"first": first, "second": second})
    with pytest.raises(ValueError):
        aggregate_predictions({"first": first}, expected_member_ids=("first", "absent"))
    with pytest.raises(ValueError):
        PopulationFold(
            "invalid",
            {"alpine": ("a",)},
            {"alpine": ObservationPartition("alpine", (("a", "b"),), "training")},
        )
    region, observations = embedding_problem()
    fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2)[0]
    with pytest.raises(ValueError):
        fit_ensemble(
            region,
            observations,
            folds=[fold, fold],
            config=TrainingConfig(objective="mlpe"),
            model_factory=model_factory,
        )
    with pytest.raises(ValueError):
        fit_ensemble(
            region,
            observations,
            folds=[fold],
            initialization_seeds=(1, 1),
            config=TrainingConfig(objective="mlpe"),
            model_factory=model_factory,
        )


def test_shared_region_members_fit_one_encoder_with_distinct_regional_calibrations(
    completed_fit, embedding_problem, model_factory
):
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
    model, state = completed_fit(member)
    assert state.region_names == ("alpine", "valley")
    assert set(cast(MLPECalibrationModule, model.calibration).regions) == {"alpine", "valley"}
    assert state.raw_variances is not None
    assert set(state.raw_variances) == {"alpine", "valley"}
    assert state.history[-1].training_loss < state.history[0].training_loss
    assert (
        cast(MLPECalibrationModule, model.calibration).regions["valley"].intercept
        - cast(MLPECalibrationModule, model.calibration).regions["alpine"].intercept
        > 2
    )


def test_known_graph_outputs_keep_surface_mean_separate_from_distance_mean(ConstantConductance):
    from ilg_toolkit import (
        Ensemble,
        ObservationPartition,
        PopulationFold,
        RegionBatch,
        TargetSpec,
        ensemble_member_identity,
    )

    with jax.enable_x64():
        region = RegionBatch(
            "graph",
            cx.field(np.ones((1, 4, 1)), "row", "column", "feature"),
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
            model = LandscapeGeneticsModel(
                ConstantConductance(level=jnp.asarray(level)),
                DirectCalibrationModule(TargetSpec("divergence")),
                1,
                training_objective="direct_log1p",
            )
            members.append(
                EnsembleMember(ensemble_member_identity(fold.fold_id, 0), fold, "completed", model)
            )
        ensemble = Ensemble(tuple(members))
        prediction = ensemble.predict(region)
        summary = ensemble.conductance_surfaces(region)
        averaged_surface_model = LandscapeGeneticsModel(
            ConstantConductance(level=jnp.asarray(2.5)),
            DirectCalibrationModule(TargetSpec("divergence")),
            1,
            training_objective="direct_log1p",
        )
        surface_prediction = averaged_surface_model.predict(region)
    np.testing.assert_allclose(prediction.prediction.as_matrix()[0, 1], 0.625, atol=1e-7)
    np.testing.assert_allclose(summary.values, 2.5, atol=1e-12)
    assert all(
        value.dtype == jnp.float64
        for value in (summary.values, summary.member_values, summary.member_spread)
    )
    np.testing.assert_allclose(summary.member_spread, 1.5, atol=1e-12)
    np.testing.assert_allclose(surface_prediction.as_matrix()[0, 1], 0.4, atol=1e-7)
    assert not np.allclose(prediction.prediction.values, surface_prediction.values)


def test_member_callback_failure_interrupts_before_starting_another_member(
    embedding_problem, model_factory
):
    import pytest

    region, observations = embedding_problem()
    initialized = []

    def tracked_factory(key):
        initialized.append(key)
        return model_factory(key)

    def interrupted_save(member):
        assert member.status == "completed"
        raise OSError("interrupted member save")

    with pytest.raises(OSError):
        fit_ensemble(
            region,
            observations,
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            initialization_seeds=(1, 2),
            config=TrainingConfig(objective="direct_log1p", epochs=0),
            model_factory=tracked_factory,
            on_member=interrupted_save,
        )
    assert len(initialized) == 1


def test_epoch_hook_resumes_pending_members_without_reinitializing_completed_updates(
    completed_fit, embedding_problem, model_factory
):
    import pytest

    with jax.enable_x64():
        region, observations = embedding_problem()
        folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
        config = TrainingConfig(objective="mlpe", epochs=3)
        checkpoints = {}
        completed_members = []

        def stop_after_one_update(identity, state):
            checkpoints[identity.member_id] = state
            if state.epoch == 1:
                raise RuntimeError("stop epoch for checkpoint")

        with pytest.raises(RuntimeError):
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
        epochs = []
        resumed = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13, 29),
            config=config,
            model_factory=model_factory,
            member_states=checkpoints,
            on_epoch=lambda identity, state: epochs.append(
                (identity.initialization_seed, state.epoch)
            ),
        )
        assert np.isfinite(resumed.predict(region).prediction.values).all()
    assert epochs == [(13, 2), (13, 3), (29, 0), (29, 1), (29, 2), (29, 3)]
    assert [completed_fit(member)[1].epoch for member in resumed.members] == [3, 3]
