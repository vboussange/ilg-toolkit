"""Independent fold fits and original-target-scale ensemble aggregation."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_mlpe_training import embedding_problem

from ilg_toolkit import (
    DirectCalibrationModule,
    EnsembleMember,
    LandscapeGeneticsModel,
    ResistanceSolverConfig,
    SamplingUnits,
    TrainingConfig,
    TrainingState,
    UnpooledMLPECalibrationModule,
    fit_ensemble,
    fit_ensemble_member,
    generate_population_folds,
)
from ilg_toolkit.models import ConductanceModel, EmbeddingDistanceModel


@final
class InitializedEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    log_weights: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return features * jnp.exp(self.log_weights)


def model_factory(key):
    return InitializedEmbedding(log_weights=jax.random.normal(key, (2,)) * 0.1)


def completed_fit(member: EnsembleMember) -> tuple[LandscapeGeneticsModel, TrainingState]:
    assert member.status == "completed"
    assert member.model is not None
    assert member.training_state is not None
    return (member.model, member.training_state)


def test_declared_seeds_initialize_readable_members_directly():
    (region, observations) = embedding_problem()
    folds = generate_population_folds(
        [region], {region.name: {"species": observations}}, n_folds=2, holdout_size=2, seed=47
    )
    result = fit_ensemble(
        [region],
        {region.name: {"species": observations}},
        folds=folds,
        initialization_seeds=(13, 29),
        config=TrainingConfig(objective="mlpe", steps=0),
        model_factory=model_factory,
    )
    assert result.expected_member_ids == (
        "fold-0000:seed=13",
        "fold-0000:seed=29",
        "fold-0001:seed=13",
        "fold-0001:seed=29",
    )
    for member in result.members:
        (model, state) = completed_fit(member)
        expected = model_factory(jax.random.PRNGKey(member.identity.initialization_seed))
        assert eqx.tree_equal(model.distance_model, expected)
        assert state.config.seed == member.identity.initialization_seed
    assert len({id(completed_fit(member)[1].optimizer_state) for member in result.members}) == 4


def test_fresh_ensemble_requires_factory_and_explicit_training_configuration():
    (region, observations) = embedding_problem()
    fold = generate_population_folds(
        [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2
    )[0]
    config = TrainingConfig(objective="mlpe", steps=0)
    with pytest.raises(TypeError, match="config"):
        fit_ensemble(  # pyright: ignore[reportCallIssue]
            [region],
            {region.name: {"species": observations}},
            folds=[fold],
            model_factory=model_factory,
        )
    with pytest.raises(ValueError, match="model_factory"):
        fit_ensemble(
            [region], {region.name: {"species": observations}}, folds=[fold], config=config
        )
    with pytest.raises(ValueError, match="config and model_factory"):
        fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=13,
            config=config,
        )
    with pytest.raises(ValueError, match="config and model_factory"):
        fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=13,
            model_factory=model_factory,
        )


def test_factory_cannot_silently_select_a_builtin_model():
    (region, observations) = embedding_problem()
    result = fit_ensemble(
        [region],
        {region.name: {"species": observations}},
        n_folds=1,
        holdout_size=2,
        config=TrainingConfig(objective="direct_log1p", steps=0),
        model_factory=lambda key: None,  # pyright: ignore[reportArgumentType]
    )
    member = result.members[0]
    assert member.status == "failed"
    assert member.failure is not None and member.failure.stage == "initialization"
    assert member.failure.error_type == "TypeError"


def test_public_fold_ensemble_fits_independent_members_and_averages_calibrated_predictions():
    with jax.enable_x64():
        (region, observations) = embedding_problem()
        folds = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=2, holdout_size=2, seed=47
        )
        result = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=folds,
            initialization_seeds=(13, 29),
            config=TrainingConfig(objective="mlpe", steps=3, learning_rate=0.03),
            model_factory=model_factory,
        )
        prediction = result.predict(region)["species"]
        fits = [completed_fit(member) for member in result.members]
        member_values = np.stack([model.predict(region)["species"].values for (model, _) in fits])
    assert len(result.members) == 4
    assert all(member.status == "completed" for member in result.members)
    assert len({member.identity.member_id for member in result.members}) == 4
    assert {member.identity.initialization_seed for member in result.members} == {13, 29}
    assert len({id(state.optimizer_state) for (_, state) in fits}) == 4
    encoders = [model.distance_model for (model, _) in fits]
    assert all(isinstance(encoder, InitializedEmbedding) for encoder in encoders)
    weights = []
    for encoder in encoders:
        assert isinstance(encoder, InitializedEmbedding)
        weights.append(tuple(np.asarray(encoder.log_weights)))
    assert len(set(weights)) == 4
    np.testing.assert_allclose(prediction.prediction.values, member_values.mean(axis=0), atol=1e-12)
    np.testing.assert_allclose(prediction.member_spread, member_values.std(axis=0), atol=1e-12)
    np.testing.assert_array_equal(prediction.member_values, member_values)
    assert prediction.prediction.target.units == "index"
    for member in result.members:
        (model, fitted) = completed_fit(member)
        assert fitted.history[-1].training_loss < fitted.history[0].training_loss
        assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
        head = model.calibration.groups[region.name]["species"]
        heldout = set(member.fold.held_out_units[region.name]["species"])
        assert not any(heldout & set(pair) for pair in head.calibration_pairs)
        assert model.validation_pairs == {}
        assert not any(heldout & set(pair) for pair in model.training_pairs[region.name]["species"])


def test_generated_folds_are_identity_stable_and_query_values_cannot_change_members():
    from ilg_toolkit import PairwiseObservations

    with jax.enable_x64():
        (region, observations) = embedding_problem()
        (other, other_observations) = embedding_problem("valley", seed=71)
        first = generate_population_folds(
            [region, other],
            {region.name: {"species": observations}, other.name: {"species": other_observations}},
            n_folds=2,
            holdout_size={region.name: {"species": 2}, other.name: {"species": 2}},
            seed=23,
        )
        reordered = generate_population_folds(
            [other, region],
            {other.name: {"species": other_observations}, region.name: {"species": observations}},
            n_folds=2,
            holdout_size={other.name: {"species": 2}, region.name: {"species": 2}},
            seed=23,
        )
        assert first == reordered
        fold = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=23
        )[0]
        changed_values = np.asarray(observations.observed_values).copy()
        for i, pair in enumerate(observations.observed_pairs):
            if set(pair) & set(fold.held_out_units[region.name]["species"]):
                changed_values[i] += 100
        changed = PairwiseObservations.from_pairs(
            observations.observed_pairs,
            changed_values,
            target=observations.target,
            sampling_unit_ids=region.sampling_units["species"].ids,
        )
        config = TrainingConfig(objective="mlpe", steps=2)
        baseline = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=[fold],
            config=config,
            model_factory=model_factory,
        )
        perturbed = fit_ensemble(
            [region],
            {region.name: {"species": changed}},
            folds=[fold],
            config=config,
            model_factory=model_factory,
        )
        np.testing.assert_array_equal(
            baseline.predict(region)["species"].prediction.values,
            perturbed.predict(region)["species"].prediction.values,
        )
    (baseline_model, baseline_fit) = completed_fit(baseline.members[0])
    (perturbed_model, perturbed_fit) = completed_fit(perturbed.members[0])
    assert baseline_fit.history == perturbed_fit.history
    assert isinstance(baseline_model.calibration, UnpooledMLPECalibrationModule)
    assert isinstance(perturbed_model.calibration, UnpooledMLPECalibrationModule)
    assert eqx.tree_equal(baseline_model.calibration.groups, perturbed_model.calibration.groups)


def test_original_scale_known_outputs_are_averaged_after_inverse_transformation():
    from ilg_toolkit import Prediction, TargetSpec, aggregate_predictions

    target = TargetSpec("divergence", units="index", transform="log1p")
    transformed = [np.log(2), np.log(10)]
    with jax.enable_x64():
        predictions = {
            str(i): Prediction(
                jnp.asarray([np.expm1(value)]),
                (("b", "a"),),
                "alpine",
                "species",
                target,
                ("a", "b", "extra"),
            )
            for (i, value) in enumerate(transformed)
        }
        result = aggregate_predictions(predictions, expected_member_ids=("0", "1"))
    assert all(
        isinstance(value, jax.Array) and value.dtype == jnp.float64
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
                    jnp.asarray([level]), (("a", "b"),), "alpine", "species", target, ("a", "b")
                )
                for (i, level) in enumerate(levels)
            }
            result = aggregate_predictions(predictions)
            np.testing.assert_allclose(result.prediction.values, [expected_mean], rtol=1e-14)
            np.testing.assert_allclose(result.member_spread, [expected_spread], rtol=1e-14)


def test_failed_members_and_missing_composition_cannot_be_silently_dropped():
    import pytest

    from ilg_toolkit import Ensemble

    (region, observations) = embedding_problem()
    outcomes = []
    calls = []

    def partly_broken_factory(key):
        calls.append(key)
        if len(calls) == 2:
            raise ArithmeticError("deliberate initialization failure")
        return model_factory(key)

    result = fit_ensemble(
        [region],
        {region.name: {"species": observations}},
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(1, 2, 3),
        config=TrainingConfig(objective="direct_log1p", steps=0),
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
        result.predict(region)["species"]
    with pytest.raises(ValueError, match="composition cannot shrink"):
        Ensemble((result.members[0], result.members[2]), result.expected_member_ids)


def test_explicit_validation_retains_selection_target_access_on_every_member():
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition

    with jax.enable_x64():
        (region, observations) = embedding_problem()
        fold = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=47
        )[0]
        # Selection consumes these held-out targets; OOF must later exclude them.
        validation = {
            region.name: {
                "species": ObservationPartition(
                    region.name, "species", fold.query[region.name]["species"].pairs, "validation"
                )
            }
        }
        fold = replace(fold, validation=validation)
        result = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=[fold],
            config=TrainingConfig(objective="mlpe", steps=2),
            model_factory=model_factory,
        )
    member = result.members[0]
    (model, fitted) = completed_fit(member)
    assert (
        model.validation_pairs[region.name]["species"] == validation[region.name]["species"].pairs
    )
    assert fitted.selection == "validation"
    assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
    assert len(model.calibration.groups[region.name]["species"].calibration_pairs) == 6


def test_one_member_continuation_matches_uninterrupted_fit_and_does_not_reinitialize():
    from dataclasses import replace

    from ilg_toolkit import fit_ensemble_member

    with jax.enable_x64():
        (region, observations) = embedding_problem()
        fold = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=47
        )[0]
        config = TrainingConfig(objective="mlpe", steps=4)
        complete = fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=31,
            config=config,
            model_factory=model_factory,
        )
        partial = fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=31,
            config=replace(config, steps=1),
            model_factory=model_factory,
        )
        (_, partial_state) = completed_fit(partial)
        resumed = fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=31,
            config=config,
            state=partial_state,
        )
        retained = fit_ensemble_member(
            [region],
            {region.name: {"species": observations}},
            fold=fold,
            initialization_seed=31,
            state=partial_state,
        )
        assert eqx.tree_equal(completed_fit(retained)[1], partial_state)
        with pytest.raises(ValueError, match="declared initialization_seed"):
            fit_ensemble_member(
                [region],
                {region.name: {"species": observations}},
                fold=fold,
                initialization_seed=32,
                state=partial_state,
            )
        assert resumed.status == complete.status == "completed"
        (resumed_model, resumed_state) = completed_fit(resumed)
        (complete_model, complete_state) = completed_fit(complete)
        np.testing.assert_array_equal(
            resumed_model.predict(region)["species"].values,
            complete_model.predict(region)["species"].values,
        )
    assert resumed_state.history == complete_state.history
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
    first = Prediction(jnp.zeros(1), (("a", "b"),), "alpine", "species", target, ("a", "b"))
    with pytest.raises(ValueError, match="original-scale"):
        replace(first, scale="model")
    for second in [
        replace(first, target=replace(target, units="percent")),
        replace(first, sampling_unit_ids=("b", "a")),
        replace(first, pairs=(("b", "a"),)),
        replace(first, region_name="other"),
    ]:
        with pytest.raises(ValueError, match="compatible.*scales|align sampling-unit"):
            aggregate_predictions({"first": first, "second": second})
    with pytest.raises(ValueError, match="composition cannot shrink"):
        aggregate_predictions({"first": first}, expected_member_ids=("first", "absent"))
    with pytest.raises(ValueError, match="held-out endpoints"):
        PopulationFold(
            "invalid",
            {"alpine": {"species": ("a",)}},
            {
                "alpine": {
                    "species": ObservationPartition("alpine", "species", (("a", "b"),), "training")
                }
            },
        )
    (region, observations) = embedding_problem()
    fold = generate_population_folds(
        [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2
    )[0]
    with pytest.raises(ValueError, match="fold_id.*unique"):
        fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=[fold, fold],
            config=TrainingConfig(objective="mlpe"),
            model_factory=model_factory,
        )
    with pytest.raises(ValueError, match="initialization_seeds.*unique"):
        fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=[fold],
            initialization_seeds=(1, 1),
            config=TrainingConfig(objective="mlpe"),
            model_factory=model_factory,
        )


def test_shared_region_members_fit_one_encoder_with_distinct_regional_calibrations():
    with jax.enable_x64():
        first, first_observed = embedding_problem()
        second, second_observed = embedding_problem("valley", offset=3, seed=71)
        regions = (first, second)
        observations = (first_observed, second_observed)
        result = fit_ensemble(
            regions,
            {r.name: {"species": o} for (r, o) in zip(regions, observations, strict=True)},
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            config=TrainingConfig(objective="mlpe", steps=2),
            model_factory=model_factory,
        )
    member = result.members[0]
    (model, state) = completed_fit(member)
    assert state.region_names == ("alpine", "valley")
    assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
    assert set(model.calibration.groups) == {"alpine", "valley"}
    assert state.raw_variances is not None
    assert state.raw_variances.shape == (2, 2)
    assert state.history[-1].training_loss < state.history[0].training_loss
    assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
    assert (
        model.calibration.groups["valley"]["species"].intercept
        - model.calibration.groups["alpine"]["species"].intercept
        > 2
    )


@final
class ConstantConductance(ConductanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    level: jax.Array
    patch_size: int = eqx.field(static=True, default=1)
    solver_config: ResistanceSolverConfig = eqx.field(
        static=True, kw_only=True, default=ResistanceSolverConfig()
    )

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        return jnp.ones(features.shape[:2]) * self.level


def test_known_graph_outputs_keep_surface_mean_separate_from_distance_mean():
    from ilg_toolkit import (
        Ensemble,
        EnsembleMember,
        LandscapeGeneticsModel,
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
            {
                "species": SamplingUnits(
                    ("a", "b", "c", "d"), np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
                )
            },
        )
        members = []
        for i, level in enumerate((1.0, 4.0)):
            fold = PopulationFold(
                f"fold-{i}",
                {"graph": {"species": ("c", "d")}},
                {
                    "graph": {
                        "species": ObservationPartition(
                            "graph", "species", (("a", "b"),), "training"
                        )
                    }
                },
                {
                    "graph": {
                        "species": ObservationPartition("graph", "species", (("c", "d"),), "query")
                    }
                },
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
        prediction = ensemble.predict(region)["species"]
        summary = ensemble.conductance_surfaces(region)["species"]
        averaged_surface_model = LandscapeGeneticsModel(
            ConstantConductance(level=jnp.asarray(2.5)),
            DirectCalibrationModule(TargetSpec("divergence")),
            1,
            training_objective="direct_log1p",
        )
        # Resistance of the mean surface differs from the mean resistance.
        surface_prediction = averaged_surface_model.predict(region)["species"]
    np.testing.assert_allclose(prediction.prediction.as_matrix()[0, 1], 0.625, atol=1e-07)
    np.testing.assert_allclose(summary.values, 2.5, atol=1e-12)
    assert all(
        isinstance(value, jax.Array) and value.dtype == jnp.float64
        for value in (summary.values, summary.member_values, summary.member_spread)
    )
    np.testing.assert_allclose(summary.member_spread, 1.5, atol=1e-12)
    np.testing.assert_allclose(surface_prediction.as_matrix()[0, 1], 0.4, atol=1e-07)
    assert not np.allclose(prediction.prediction.values, surface_prediction.values)


def test_member_callback_failure_interrupts_before_starting_another_member():
    import pytest

    (region, observations) = embedding_problem()
    initialized = []

    def tracked_factory(key):
        initialized.append(key)
        return model_factory(key)

    def interrupted_save(member):
        assert member.status == "completed"
        raise OSError("interrupted member save")

    with pytest.raises(OSError, match="interrupted member save"):
        fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            n_folds=1,
            holdout_size=2,
            fold_seed=47,
            initialization_seeds=(1, 2),
            config=TrainingConfig(objective="direct_log1p", steps=0),
            model_factory=tracked_factory,
            on_member=interrupted_save,
        )
    assert len(initialized) == 1


def test_step_hook_interrupts_and_member_states_continue_the_exact_ensemble():
    import pytest

    with jax.enable_x64():
        (region, observations) = embedding_problem()
        folds = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=47
        )
        config = TrainingConfig(objective="mlpe", steps=3)
        complete = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=folds,
            initialization_seeds=(13, 29),
            config=config,
            model_factory=model_factory,
        )
        checkpoints = {}
        completed_members = []

        def stop_after_one_update(identity, state):
            checkpoints[identity.member_id] = state
            if state.step == 1:
                raise RuntimeError("stop step for checkpoint")

        with pytest.raises(RuntimeError, match="stop step for checkpoint"):
            fit_ensemble(
                [region],
                {region.name: {"species": observations}},
                folds=folds,
                initialization_seeds=(13, 29),
                config=config,
                model_factory=model_factory,
                on_step=stop_after_one_update,
                on_member=completed_members.append,
            )
        assert completed_members == []
        assert len(checkpoints) == 1
        resumed = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=folds,
            initialization_seeds=(13, 29),
            config=config,
            model_factory=model_factory,
            member_states=checkpoints,
        )
        np.testing.assert_array_equal(
            resumed.predict(region)["species"].prediction.values,
            complete.predict(region)["species"].prediction.values,
        )
    assert [completed_fit(member)[1].history for member in resumed.members] == [
        completed_fit(member)[1].history for member in complete.members
    ]
