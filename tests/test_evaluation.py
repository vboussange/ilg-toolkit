"""OOF eligibility checks actual target access and pools unique pairs once."""

from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    DirectCalibrationModule,
    Ensemble,
    EnsembleMember,
    LandscapeGeneticsModel,
    ObservationPartition,
    PairwiseObservations,
    PopulationFold,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    UnpooledMLPECalibrationModule,
    ensemble_member_identity,
    evaluate_ensemble,
    predict_out_of_fold,
    score_out_of_fold,
)
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class ScalarEmbedding(EmbeddingDistanceModel):
    species: tuple[str, ...] = eqx.field(static=True, default=("species",), kw_only=True)
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def _embedding_grid(
        self, features, species_index, *, inference=True, key=None, patch_batch_size=None
    ):
        return features * self.weight


def completed_model(member: EnsembleMember) -> LandscapeGeneticsModel:
    assert member.model is not None
    return member.model


def known_ensemble() -> tuple[RegionBatch, Ensemble, TargetSpec]:
    region = RegionBatch(
        "alpine",
        cx.field(np.arange(4).reshape(1, 4, 1), "row", "column", "feature"),
        {
            "species": SamplingUnits(
                ("a", "b", "c", "d"), np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
            )
        },
    )
    target = TargetSpec("divergence", units="index")
    members = []
    for index, weight in enumerate((1.0, 2.0)):
        fold = PopulationFold(
            f"fold-{index}",
            {"alpine": {"species": ("c", "d")}},
            {
                "alpine": {
                    "species": ObservationPartition("alpine", "species", (("a", "b"),), "training")
                }
            },
            {
                "alpine": {
                    "species": ObservationPartition("alpine", "species", (("c", "d"),), "query")
                }
            },
        )
        model = LandscapeGeneticsModel(
            ScalarEmbedding(jnp.asarray(weight)),
            DirectCalibrationModule(target),
            1,
            training_pairs={"alpine": {"species": (("a", "b"),)}},
            training_objective="direct_log1p",
        )
        members.append(
            EnsembleMember(ensemble_member_identity(fold.fold_id, 0), fold, "completed", model)
        )
    return (region, Ensemble(tuple(members)), target)


def test_label_free_eligible_predictions_average_first_and_score_each_pair_once():
    (region, ensemble, target) = known_ensemble()
    prediction = predict_out_of_fold(
        ensemble, [region], {region.name: {"species": [("d", "c"), ("a", "b"), ("c", "d")]}}
    )
    assert tuple(prediction.predictions) == ("alpine",)
    points = prediction.predictions["alpine"]["species"]
    assert points.sampling_unit_ids == region.sampling_units["species"].ids
    assert points.pairs == (("a", "b"), ("c", "d"))
    assert points.region_name == "alpine"
    assert np.isnan(points.as_matrix()[0, 1])
    assert points.as_matrix()[2, 3] == 2.5
    for values in (prediction.values, prediction.member_values, prediction.member_spread):
        assert isinstance(values, jax.Array)
        assert values.dtype == jnp.float64
    assert prediction.keys == (("alpine", "species", ("a", "b")), ("alpine", "species", ("c", "d")))
    np.testing.assert_array_equal(prediction.eligible_counts, [0, 2])
    np.testing.assert_array_equal(prediction.covered_mask, [False, True])
    assert np.isnan(prediction.values[0])
    assert prediction.values[1] == 2.5
    with jax.enable_x64(False):
        assert prediction.values.dtype == jnp.float64
        np.testing.assert_allclose(prediction.values, [np.nan, 2.5])
        assert not jax.config.read("jax_enable_x64")
    assert prediction.member_spread[1] == 1.5
    assert prediction.coverage == 0.5
    observations = PairwiseObservations.from_pairs([("a", "b"), ("c", "d")], [1, 3], target=target)
    evaluation = score_out_of_fold(prediction, {"alpine": {"species": observations}})
    assert isinstance(evaluation.observed_values, jax.Array)
    assert evaluation.observed_values.dtype == jnp.float64
    assert evaluation.n_pairs == 1
    assert evaluation.n_query_pairs == 2
    assert evaluation.mse == 0.25
    assert evaluation.mae == evaluation.rmse == 0.5
    assert evaluation.predictions is prediction
    wrapped = evaluate_ensemble(ensemble, [region], {region.name: {"species": observations}})
    np.testing.assert_array_equal(wrapped.predictions.values, prediction.values)
    assert wrapped.mse == 0.25


def test_actual_mlpe_fit_keeps_prior_only_heldout_ids_eligible_and_query_perturbations_label_free():
    from test_ensemble import model_factory
    from test_mlpe_training import embedding_problem

    from ilg_toolkit import TrainingConfig, fit_ensemble, generate_population_folds

    with jax.enable_x64():
        (region, observations) = embedding_problem()
        folds = generate_population_folds(
            [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=47
        )
        ensemble = fit_ensemble(
            [region],
            {region.name: {"species": observations}},
            folds=folds,
            initialization_seeds=(13, 29),
            config=TrainingConfig(objective="mlpe", steps=2),
            model_factory=model_factory,
        )
        query = folds[0].query[region.name]["species"]
        first = evaluate_ensemble(
            ensemble,
            [region],
            {region.name: {"species": observations}},
            partitions={region.name: {"species": query}},
        )
        changed = PairwiseObservations.from_pairs(
            observations.observed_pairs,
            observations.observed_values + 100,
            target=observations.target,
            sampling_unit_ids=region.sampling_units["species"].ids,
        )
        perturbed = evaluate_ensemble(
            ensemble,
            [region],
            {region.name: {"species": changed}},
            partitions={region.name: {"species": query}},
        )
    assert first.n_pairs == first.n_query_pairs == 1
    assert first.coverage == 1
    np.testing.assert_array_equal(first.predictions.eligible_counts, [2])
    np.testing.assert_array_equal(first.predictions.values, perturbed.predictions.values)
    assert first.mse != perturbed.mse
    for member in ensemble.members:
        calibration = completed_model(member).calibration
        assert isinstance(calibration, UnpooledMLPECalibrationModule)
        head = calibration.groups[region.name]["species"]
        assert set(folds[0].held_out_units[region.name]["species"]).issubset(head.population_ids)
        assert not any(
            set(pair) & set(folds[0].held_out_units[region.name]["species"])
            for pair in head.calibration_pairs
        )
        assert (
            first.predictions.access[member.identity.member_id][region.name][
                "species"
            ].calibration_pairs
            == head.calibration_pairs
        )


def test_oof_values_follow_configured_precision_without_changing_it():
    with jax.enable_x64(False):
        (region, ensemble, _) = known_ensemble()
        prediction = predict_out_of_fold(
            ensemble, [region], {region.name: {"species": [("c", "d")]}}
        )
        assert prediction.values.dtype == jnp.float32
        np.testing.assert_allclose(prediction.values, [2.5])
        assert jax.config.read("jax_enable_x64") is False


def test_validation_selection_and_own_query_target_access_preclude_scoring():
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    (region, ensemble, target) = known_ensemble()
    contaminated = []
    for member in ensemble.members:
        model = eqx.tree_at(
            lambda model: model.validation_pairs,
            completed_model(member),
            {region.name: {"species": (("d", "c"),)}},
        )
        contaminated.append(replace(member, model=model))
    contaminated = Ensemble(tuple(contaminated))
    observed = PairwiseObservations.from_pairs([("c", "d")], [3], target=target)
    result = evaluate_ensemble(
        contaminated,
        [region],
        {region.name: {"species": observed}},
        regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
    )
    assert result.n_pairs == 0
    assert result.coverage == 0
    assert result.mse is result.rmse is result.mae is None
    assert result.predictions.coverage_status == "none"
    assert all(
        reasons == ("query_target_accessed",)
        for reasons in result.predictions.exclusion_reasons.values()
    )


def test_at_least_one_requires_the_same_endpoint_to_be_heldout_and_still_unseen():
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    (region, ensemble, _) = known_ensemble()
    # c is held out but accessed; b is untouched but not held out. The same
    # endpoint must satisfy both conditions, so (b,c) cannot become eligible.
    members = []
    for member in ensemble.members:
        model = eqx.tree_at(
            lambda model: (model.training_pairs, model.validation_pairs),
            completed_model(member),
            ({region.name: {"species": ()}}, {region.name: {"species": (("a", "c"),)}}),
        )
        members.append(replace(member, model=model))
    result = predict_out_of_fold(
        Ensemble(tuple(members)),
        [region],
        {region.name: {"species": [("b", "c"), ("c", "d")]}},
        regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
    )
    np.testing.assert_array_equal(result.eligible_counts, [0, 2])
    assert all(
        reason[0] == "insufficient_held_out_unseen_endpoints"
        for reason in result.exclusion_reasons.values()
    )


def test_failed_pending_and_unknown_access_members_are_explicit_in_coverage():
    from dataclasses import replace

    from ilg_toolkit import MemberFailure

    (region, ensemble, target) = known_ensemble()
    (first, second) = ensemble.members
    failed = replace(
        second,
        status="failed",
        model=None,
        failure=MemberFailure("fit", "RuntimeError", "failed fit"),
    )
    result = predict_out_of_fold(
        Ensemble((first, failed)), [region], {region.name: {"species": [("c", "d")]}}
    )
    np.testing.assert_array_equal(result.eligible_counts, [1])
    assert result.member_failures[failed.identity.member_id].message == "failed fit"
    assert result.exclusion_reasons[failed.identity.member_id] == ("member_failed",)
    pending = replace(failed, status="pending", failure=None)
    unavailable = predict_out_of_fold(
        Ensemble((pending,)), [region], {region.name: {"species": [("c", "d")]}}, target=target
    )
    assert unavailable.coverage == 0
    assert unavailable.member_statuses[pending.identity.member_id] == "pending"
    unknown = replace(
        first, model=eqx.tree_at(lambda model: model.training_pairs, completed_model(first), {})
    )
    unknown_result = predict_out_of_fold(
        Ensemble((unknown,)), [region], {region.name: {"species": [("c", "d")]}}
    )
    assert unknown_result.coverage == 0
    assert unknown_result.exclusion_reasons[unknown.identity.member_id] == (
        "unknown_encoder_access",
    )
    declared = replace(
        unknown,
        model=eqx.tree_at(
            lambda model: model.training_pairs,
            completed_model(unknown),
            {region.name: {"species": ()}},
        ),
    )
    declared_result = predict_out_of_fold(
        Ensemble((declared,)), [region], {region.name: {"species": [("c", "d")]}}
    )
    assert declared_result.coverage == 1


def fitted_member_problem():
    from test_ensemble import model_factory
    from test_mlpe_training import embedding_problem

    from ilg_toolkit import TrainingConfig, fit_ensemble, generate_population_folds

    (region, observations) = embedding_problem()
    fold = generate_population_folds(
        [region], {region.name: {"species": observations}}, n_folds=1, holdout_size=2, seed=47
    )[0]
    ensemble = fit_ensemble(
        [region],
        {region.name: {"species": observations}},
        folds=[fold],
        initialization_seeds=(13,),
        config=TrainingConfig(objective="mlpe", steps=1),
        model_factory=model_factory,
    )
    assert ensemble.members[0].status == "completed"
    return (region, observations, ensemble)


def test_actual_recalibration_changes_eligibility_and_query_calibration_always_excludes():
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    with jax.enable_x64():
        (region, observations, ensemble) = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]["species"]
        old = next(label for label in region.sampling_units["species"].ids if label not in held)
        cross = ObservationPartition(region.name, "species", ((held[0], old),), "calibration")
        training = member.fold.training[region.name]["species"]
        calibrated = completed_model(member).calibrate(
            region, {"species": observations}, partitions={"species": (training, cross)}
        )
        changed = Ensemble((replace(member, model=calibrated),))
        strict = predict_out_of_fold(changed, [region], {region.name: {"species": [held]}})
        partial = predict_out_of_fold(
            changed,
            [region],
            {region.name: {"species": [held]}},
            regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
        )
        own = ObservationPartition(region.name, "species", ((held[0], held[1]),), "calibration")
        contaminated = completed_model(member).calibrate(
            region, {"species": observations}, partitions={"species": (training, own)}
        )
        forbidden = predict_out_of_fold(
            Ensemble((replace(member, model=contaminated),)),
            [region],
            {region.name: {"species": [held]}},
            regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
        )
    assert eqx.tree_equal(calibrated.distance_model, completed_model(member).distance_model)
    assert strict.coverage == 0
    assert partial.coverage == 1
    assert forbidden.coverage == 0
    assert forbidden.exclusion_reasons[member.identity.member_id] == ("query_target_accessed",)


def test_calibrated_nominal_holdout_cannot_be_replaced_by_an_unused_nominal_training_endpoint():
    from dataclasses import replace

    from test_ensemble import model_factory
    from test_mlpe_training import embedding_problem

    from ilg_toolkit import EvaluationRegime, TrainingConfig, fit_ensemble

    with jax.enable_x64():
        (region, full_observations) = embedding_problem()
        held = ("unit-0", "unit-5")
        unused = "unit-1"
        training_pairs = (("unit-2", "unit-3"), ("unit-2", "unit-4"), ("unit-3", "unit-4"))
        cross = ("unit-0", "unit-2")
        query = ("unit-0", unused)
        selected = training_pairs + (cross, query)
        value_by_pair = {
            tuple(sorted(pair)): value
            for (pair, value) in zip(
                full_observations.observed_pairs, full_observations.observed_values, strict=True
            )
        }
        observed = PairwiseObservations.from_pairs(
            selected,
            [value_by_pair[tuple(sorted(pair))] for pair in selected],
            target=full_observations.target,
            sampling_unit_ids=region.sampling_units["species"].ids,
        )
        fold = PopulationFold(
            "incomplete",
            {region.name: {"species": held}},
            {
                region.name: {
                    "species": ObservationPartition(
                        region.name, "species", training_pairs, "training"
                    )
                }
            },
            {
                region.name: {
                    "species": ObservationPartition(region.name, "species", (query,), "query")
                }
            },
            query_regime="at_least_one_unseen",
        )
        ensemble = fit_ensemble(
            [region],
            {region.name: {"species": observed}},
            folds=[fold],
            model_factory=model_factory,
            config=TrainingConfig(objective="mlpe", steps=0),
        )
        member = ensemble.members[0]
        assert member.status == "completed"
        calibrated = completed_model(member).calibrate(
            region,
            {"species": observed},
            partitions={
                "species": (
                    fold.training[region.name]["species"],
                    ObservationPartition(region.name, "species", (cross,), "calibration"),
                )
            },
        )
        result = predict_out_of_fold(
            Ensemble((replace(member, model=calibrated),)),
            [region],
            {region.name: {"species": [query]}},
            regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
        )
    access = result.access[member.identity.member_id][region.name]["species"]
    assert not any(unused in pair for pair in access.training_pairs + access.calibration_pairs)
    assert any(held[0] in pair for pair in access.calibration_pairs)
    assert result.coverage == 0


def test_known_effects_and_support_require_explicit_modes_and_report_separate_variance():
    from ilg_toolkit import EvaluationRegime, EvaluationSupport

    with jax.enable_x64():
        (region, observations, ensemble) = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]["species"]
        old = next(label for label in region.sampling_units["species"].ids if label not in held)
        known_pair = (old, held[0])
        known = predict_out_of_fold(
            ensemble,
            [region],
            {region.name: {"species": [known_pair]}},
            regime=EvaluationRegime(
                endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
            ),
        )
        expected_known = completed_model(member).predict_known_effects(
            region, pairs={"species": [tuple(sorted(known_pair))]}
        )["species"]
        np.testing.assert_allclose(known.values, expected_known.prediction.values, atol=1e-12)
        assert known.member_model_variances is not None
        assert isinstance(known.member_model_variances, jax.Array)
        assert known.member_model_variances.dtype == jnp.float64
        np.testing.assert_allclose(
            known.member_model_variances[0], expected_known.model_variance, atol=1e-12
        )
        assert known.variance_scale == "model"
        assert (
            known.conditioning_provenance[member.identity.member_id][region.name]["species"].mode
            == "known_effects"
        )
        support_pair = (held[0], old)
        lookup = {
            tuple(sorted(pair)): value
            for (pair, value) in zip(
                observations.observed_pairs, observations.observed_values, strict=True
            )
        }
        support_observed = PairwiseObservations.from_pairs(
            [support_pair], [lookup[tuple(sorted(support_pair))]], target=observations.target
        )
        support = EvaluationSupport(
            support_observed,
            ObservationPartition(region.name, "species", (support_pair,), "support"),
        )
        with pytest.raises(ValueError, match="explicit support prediction"):
            predict_out_of_fold(
                ensemble,
                [region],
                {region.name: {"species": [held]}},
                support={region.name: {"species": support}},
            )
        strict = predict_out_of_fold(
            ensemble,
            [region],
            {region.name: {"species": [held]}},
            support={region.name: {"species": support}},
            regime=EvaluationRegime(prediction_mode="support"),
        )
        assert strict.coverage == 0
        permitted = EvaluationRegime(
            prediction_mode="support", support_endpoint_policy="allow_declared_support"
        )
        conditional = predict_out_of_fold(
            ensemble,
            [region],
            {region.name: {"species": [held]}},
            support={region.name: {"species": support}},
            regime=permitted,
        )
        expected = completed_model(member).predict_with_support(
            region,
            {"species": support_observed},
            support_partition={"species": support.partition},
            pairs={"species": [held]},
        )["species"]
        np.testing.assert_allclose(conditional.values, expected.prediction.values, atol=1e-12)
        assert conditional.member_model_variances is not None
        np.testing.assert_allclose(
            conditional.member_model_variances[0], expected.model_variance, atol=1e-12
        )
        assert conditional.coverage == 1
        np.testing.assert_array_equal(conditional.member_spread, [0])
        assert conditional.member_model_variances[0, 0] > 0
        assert conditional.regime.support_endpoint_policy == "allow_declared_support"
        assert (
            conditional.conditioning_provenance[member.identity.member_id][region.name][
                "species"
            ].support_pairs
            == support.partition.pairs
        )
        with pytest.raises(ValueError, match="Support/query overlap"):
            predict_out_of_fold(
                ensemble,
                [region],
                {region.name: {"species": [support_pair]}},
                support={region.name: {"species": support}},
                regime=permitted,
            )
        with pytest.raises(ValueError, match="declared support observations"):
            predict_out_of_fold(
                ensemble, [region], {region.name: {"species": [held]}}, regime=permitted
            )


def test_access_identity_includes_region_and_prediction_failures_remain_visible():
    from dataclasses import replace

    (region, ensemble, _) = known_ensemble()
    valley = replace(
        region,
        name="valley",
        sampling_units={
            "species": replace(region.sampling_units["species"], ids=("v-a", "v-b", "v-c", "v-d"))
        },
    )
    members = []
    for member in ensemble.members:
        fold = replace(
            member.fold,
            held_out_units={
                "alpine": {"species": ("c", "d")},
                "valley": {"species": ("v-c", "v-d")},
            },
            training={
                name: {
                    "species": ObservationPartition(
                        name,
                        "species",
                        (("a", "b"),) if name == "alpine" else (("v-a", "v-b"),),
                        "training",
                    )
                }
                for name in ("alpine", "valley")
            },
            query={
                name: {
                    "species": ObservationPartition(
                        name,
                        "species",
                        (("c", "d"),) if name == "alpine" else (("v-c", "v-d"),),
                        "query",
                    )
                }
                for name in ("alpine", "valley")
            },
        )
        model = eqx.tree_at(
            lambda model: model.training_pairs,
            completed_model(member),
            {"alpine": {"species": (("c", "d"),)}, "valley": {"species": (("v-a", "v-b"),)}},
        )
        members.append(replace(member, fold=fold, model=model))
    regional = predict_out_of_fold(
        Ensemble(tuple(members)),
        [region, valley],
        {"alpine": {"species": [("c", "d")]}, "valley": {"species": [("v-c", "v-d")]}},
    )
    np.testing.assert_array_equal(regional.eligible_counts, [0, 2])
    assert tuple(regional.predictions) == ("alpine", "valley")
    assert regional.predictions["alpine"]["species"].region_name == "alpine"
    assert regional.predictions["valley"]["species"].region_name == "valley"
    assert np.isnan(regional.predictions["alpine"]["species"].values[0])
    np.testing.assert_allclose(regional.predictions["valley"]["species"].values, [2.5])
    broken = replace(
        ensemble.members[0],
        model=eqx.tree_at(
            lambda model: model.distance_model,
            completed_model(ensemble.members[0]),
            ScalarEmbedding(jnp.asarray(np.nan)),
        ),
    )
    partial = predict_out_of_fold(
        Ensemble((broken, ensemble.members[1])), [region], {region.name: {"species": [("c", "d")]}}
    )
    np.testing.assert_array_equal(partial.eligible_counts, [1])
    assert partial.values[0] == 4
    assert (
        "nonfinite"
        in partial.prediction_failures[broken.identity.member_id][region.name]["species"].message
    )
    assert partial.exclusion_reasons[broken.identity.member_id] == ("prediction_failed",)


def test_only_eligible_pairs_are_inverse_transformed():
    from ilg_toolkit import TrainingConfig, fit_ensemble

    with jax.enable_x64():
        ids = ("a", "b", "c", "d", "e")
        region = RegionBatch(
            "alpine",
            cx.field(np.arange(5).reshape(1, 5, 1), "row", "column", "feature"),
            {"species": SamplingUnits(ids, np.array([[0, i] for i in range(5)]))},
        )
        target = TargetSpec("divergence", units="index", transform="sqrt")
        training_pairs = (("a", "b"), ("a", "c"), ("b", "c"))
        # Independent reference: sqrt(target) = 4 - 0.5 * distance.
        observed = PairwiseObservations.from_pairs(
            training_pairs, [12.25, 4, 12.25], target=target, sampling_unit_ids=ids
        )
        fold = PopulationFold(
            "signed",
            {region.name: {"species": ("d", "e")}},
            {
                region.name: {
                    "species": ObservationPartition(
                        region.name, "species", training_pairs, "training"
                    )
                }
            },
        )
        ensemble = fit_ensemble(
            [region],
            {region.name: {"species": observed}},
            folds=[fold],
            config=TrainingConfig(objective="mlpe", steps=0),
            model_factory=lambda key: ScalarEmbedding(jnp.asarray(1.0)),
        )
        assert ensemble.members[0].status == "completed"
        with pytest.raises(ValueError, match="sqrt inverse"):
            ensemble.predict(region)["species"]
        marginal = completed_model(ensemble.members[0]).predict(
            region, pairs={"species": [("d", "e")]}
        )["species"]
        assert marginal.pairs == (("d", "e"),)
        assert marginal.target == target
        assert marginal.region_name == region.name
        assert marginal.scale == "original"
        selected = predict_out_of_fold(ensemble, [region], {region.name: {"species": [("d", "e")]}})
    assert selected.coverage == 1
    np.testing.assert_allclose(selected.values, [12.25], atol=1e-12)
    np.testing.assert_array_equal(selected.values, marginal.values)
    assert selected.prediction_failures == {}


def test_reused_calibration_support_cannot_contribute_and_target_scales_must_match():
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime, EvaluationSupport

    with jax.enable_x64():
        (region, observations, ensemble) = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]["species"]
        calibration = completed_model(member).calibration
        assert isinstance(calibration, UnpooledMLPECalibrationModule)
        reused = calibration.groups[region.name]["species"].calibration_pairs[0]
        values = {
            tuple(sorted(pair)): value
            for (pair, value) in zip(
                observations.observed_pairs, observations.observed_values, strict=True
            )
        }
        support_observed = PairwiseObservations.from_pairs(
            [reused], [values[reused]], target=observations.target
        )
        support = EvaluationSupport(
            support_observed, ObservationPartition(region.name, "species", (reused,), "support")
        )
        rejected = predict_out_of_fold(
            ensemble,
            [region],
            {region.name: {"species": [held]}},
            support={region.name: {"species": support}},
            regime=EvaluationRegime(
                prediction_mode="support", support_endpoint_policy="allow_declared_support"
            ),
        )
        assert rejected.coverage == 0
        assert (
            "reuses a calibration pair"
            in rejected.prediction_failures[member.identity.member_id][region.name][
                "species"
            ].message
        )
        prediction = predict_out_of_fold(ensemble, [region], {region.name: {"species": [held]}})
    incompatible = PairwiseObservations.from_pairs(
        [held], [3], target=replace(observations.target, units="percent")
    )
    with pytest.raises(ValueError, match="same declared target scale"):
        score_out_of_fold(prediction, {"alpine": {"species": incompatible}})
    with pytest.raises(ValueError, match="query roles"):
        evaluate_ensemble(
            ensemble,
            [region],
            {region.name: {"species": observations}},
            partitions={region.name: {"species": member.fold.training[region.name]["species"]}},
        )


def test_direct_pair_predictions_share_oof_policy_and_skip_unrequested_inverse_overflow():
    from dataclasses import replace

    from ilg_toolkit import Prediction

    (region, ensemble, _) = known_ensemble()
    region = replace(
        region,
        features=cx.field(np.array([0, 100, 1, 2]).reshape(1, 4, 1), "row", "column", "feature"),
    )
    target = TargetSpec("divergence", units="index", transform="log1p")
    members = tuple(
        replace(
            member,
            model=eqx.tree_at(lambda model: model.target, completed_model(member), target),
        )
        for member in ensemble.members
    )
    ensemble = Ensemble(members)
    expected = [np.expm1(1), np.expm1(4)]
    for member, value in zip(members, expected, strict=True):
        model = completed_model(member)
        with pytest.raises(FloatingPointError, match="inverse transformation"):
            model.predict(region)["species"]
        selected = model.predict(region, pairs={"species": [("d", "c")]})["species"]
        assert isinstance(selected, Prediction)
        assert selected.pairs == (("d", "c"),)
        assert selected.target == target
        assert selected.region_name == region.name
        assert selected.scale == "original"
        np.testing.assert_allclose(selected.values, [value], rtol=1e-07)
        matrix = selected.as_matrix()
        assert matrix.shape == (4, 4)
        assert np.isnan(matrix[0, 1])
        np.testing.assert_allclose(matrix[3, 2], value, rtol=1e-07)
    out_of_fold = predict_out_of_fold(
        ensemble, [region], {region.name: {"species": [("a", "b"), ("c", "d")]}}
    )
    np.testing.assert_array_equal(out_of_fold.eligible_counts, [0, 2])
    np.testing.assert_allclose(out_of_fold.values[1], np.mean(expected), rtol=1e-07)
    assert out_of_fold.prediction_failures == {}


def test_pair_queries_require_unique_distinct_labels_with_query_locations():
    (region, ensemble, _) = known_ensemble()
    model = completed_model(ensemble.members[0])
    for pairs in ([], [("a", "a")], [("a", "b"), ("b", "a")], [("a",)], ["ab"], [("a", "unknown")]):
        with pytest.raises(ValueError):
            model.predict(region, pairs={"species": pairs})["species"]
