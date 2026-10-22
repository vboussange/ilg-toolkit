"""Scientific behavior of evaluation."""

from typing import cast

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    Ensemble,
    MLPECalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    PopulationFold,
    RegionBatch,
    TargetSpec,
    evaluate_ensemble,
    predict_out_of_fold,
    score_out_of_fold,
)


def test_label_free_eligible_predictions_average_first_and_score_each_pair_once(known_ensemble):
    region, ensemble, target = known_ensemble()
    prediction = predict_out_of_fold(ensemble, region, [("d", "c"), ("a", "b"), ("c", "d")])
    assert tuple(prediction.predictions) == ("alpine",)
    points = prediction.predictions["alpine"]
    assert points.sampling_unit_ids == region.sampling_unit_ids
    assert points.pairs == (("a", "b"), ("c", "d"))
    assert points.region_name == "alpine"
    assert np.isnan(points.as_matrix()[0, 1])
    assert points.as_matrix()[2, 3] == 2.5
    for values in (prediction.values, prediction.member_values, prediction.member_spread):
        assert values.dtype == jnp.float64
    assert prediction.keys == (("alpine", ("a", "b")), ("alpine", ("c", "d")))
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
    evaluation = score_out_of_fold(prediction, observations)
    assert evaluation.observed_values.dtype == jnp.float64
    assert evaluation.n_pairs == 1
    assert evaluation.n_query_pairs == 2
    assert evaluation.mse == 0.25
    assert evaluation.mae == evaluation.rmse == 0.5
    assert evaluation.predictions is prediction
    wrapped = evaluate_ensemble(ensemble, region, observations)
    np.testing.assert_array_equal(wrapped.predictions.values, prediction.values)
    assert wrapped.mse == 0.25


def test_actual_mlpe_fit_keeps_prior_only_heldout_ids_eligible_and_query_perturbations_label_free(
    completed_model, embedding_problem, model_factory
):

    from ilg_toolkit import TrainingConfig, fit_ensemble, generate_population_folds

    with jax.enable_x64():
        region, observations = embedding_problem()
        folds = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)
        ensemble = fit_ensemble(
            region,
            observations,
            folds=folds,
            initialization_seeds=(13, 29),
            config=TrainingConfig(objective="mlpe", epochs=2),
            model_factory=model_factory,
        )
        query = folds[0].query[region.name]
        first = evaluate_ensemble(ensemble, region, observations, partitions=query)
        changed = PairwiseObservations.from_pairs(
            observations.observed_pairs,
            observations.observed_values + 100,
            target=observations.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        perturbed = evaluate_ensemble(ensemble, region, changed, partitions=query)
    assert first.n_pairs == first.n_query_pairs == 1
    assert first.coverage == 1
    np.testing.assert_array_equal(first.predictions.eligible_counts, [2])
    np.testing.assert_array_equal(first.predictions.values, perturbed.predictions.values)
    assert first.mse != perturbed.mse
    for member in ensemble.members:
        calibration = completed_model(member).calibration
        head = cast(MLPECalibrationModule, calibration).regions[region.name]
        assert set(folds[0].held_out_units[region.name]).issubset(head.population_ids)
        assert not any(
            set(pair) & set(folds[0].held_out_units[region.name]) for pair in head.calibration_pairs
        )
        assert (
            first.predictions.access[member.identity.member_id][region.name].calibration_pairs
            == head.calibration_pairs
        )


def test_validation_selection_and_own_query_target_access_preclude_scoring(
    completed_model, known_ensemble
):
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    region, ensemble, target = known_ensemble()
    contaminated = []
    for member in ensemble.members:
        model = eqx.tree_at(
            lambda model: model.validation_pairs,
            completed_model(member),
            {region.name: (("d", "c"),)},
        )
        contaminated.append(replace(member, model=model))
    contaminated = Ensemble(tuple(contaminated))
    observed = PairwiseObservations.from_pairs([("c", "d")], [3], target=target)
    result = evaluate_ensemble(
        contaminated,
        region,
        observed,
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


def test_at_least_one_requires_the_same_endpoint_to_be_heldout_and_still_unseen(
    completed_model, known_ensemble
):
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    region, ensemble, _ = known_ensemble()
    # c is nominally held out but selected via (a,c). b was never accessed,
    # yet is nominally a training endpoint: separate nominal/unseen counts
    # would wrongly admit (b,c).
    members = []
    for member in ensemble.members:
        model = eqx.tree_at(
            lambda model: (model.training_pairs, model.validation_pairs),
            completed_model(member),
            ({region.name: ()}, {region.name: (("a", "c"),)}),
        )
        members.append(replace(member, model=model))
    result = predict_out_of_fold(
        Ensemble(tuple(members)),
        region,
        [("b", "c"), ("c", "d")],
        regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
    )
    np.testing.assert_array_equal(result.eligible_counts, [0, 2])
    assert all(
        reason[0] == "insufficient_held_out_unseen_endpoints"
        for reason in result.exclusion_reasons.values()
    )


def test_failed_pending_and_unknown_access_members_are_explicit_in_coverage(
    completed_model, known_ensemble
):
    from dataclasses import replace

    from ilg_toolkit import MemberFailure

    region, ensemble, target = known_ensemble()
    first, second = ensemble.members
    failed = replace(
        second,
        status="failed",
        model=None,
        failure=MemberFailure("fit", "RuntimeError", "failed fit"),
    )
    result = predict_out_of_fold(Ensemble((first, failed)), region, [("c", "d")])
    np.testing.assert_array_equal(result.eligible_counts, [1])
    assert result.exclusion_reasons[failed.identity.member_id] == ("member_failed",)
    pending = replace(failed, status="pending", failure=None)
    unavailable = predict_out_of_fold(Ensemble((pending,)), region, [("c", "d")], target=target)
    assert unavailable.coverage == 0
    assert unavailable.member_statuses[pending.identity.member_id] == "pending"
    unknown = replace(
        first, model=eqx.tree_at(lambda model: model.training_pairs, completed_model(first), {})
    )
    unknown_result = predict_out_of_fold(Ensemble((unknown,)), region, [("c", "d")])
    assert unknown_result.coverage == 0
    assert unknown_result.exclusion_reasons[unknown.identity.member_id] == (
        "unknown_encoder_access",
    )
    declared = replace(
        unknown,
        model=eqx.tree_at(
            lambda model: model.training_pairs, completed_model(unknown), {region.name: ()}
        ),
    )
    declared_result = predict_out_of_fold(Ensemble((declared,)), region, [("c", "d")])
    assert declared_result.coverage == 1


def test_actual_recalibration_changes_eligibility_and_query_calibration_always_excludes(
    calibrated, completed_model, fitted_member_problem
):
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime

    with jax.enable_x64():
        region, observations, ensemble = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]
        old = next(label for label in region.sampling_unit_ids if label not in held)
        cross = ObservationPartition(region.name, ((held[0], old),), "calibration")
        training = member.fold.training[region.name]
        calibrated = completed_model(member).calibrate(
            region, observations, partitions=(training, cross)
        )
        changed = Ensemble((replace(member, model=calibrated),))
        strict = predict_out_of_fold(changed, region, [held])
        partial = predict_out_of_fold(
            changed, region, [held], regime=EvaluationRegime(endpoint_regime="at_least_one_unseen")
        )
        own = ObservationPartition(region.name, ((held[0], held[1]),), "calibration")
        contaminated = completed_model(member).calibrate(
            region, observations, partitions=(training, own)
        )
        forbidden = predict_out_of_fold(
            Ensemble((replace(member, model=contaminated),)),
            region,
            [held],
            regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
        )
    assert eqx.tree_equal(calibrated.distance_model, completed_model(member).distance_model)
    assert strict.coverage == 0
    assert partial.coverage == 1
    assert forbidden.coverage == 0
    assert forbidden.exclusion_reasons[member.identity.member_id] == ("query_target_accessed",)


def test_calibrated_nominal_holdout_cannot_be_replaced_by_an_unused_nominal_training_endpoint(
    calibrated, completed_model, embedding_problem, model_factory
):
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime, TrainingConfig, fit_ensemble

    with jax.enable_x64():
        region, full_observations = embedding_problem()
        held = ("unit-0", "unit-5")
        unused = "unit-1"
        training_pairs = (("unit-2", "unit-3"), ("unit-2", "unit-4"), ("unit-3", "unit-4"))
        cross = ("unit-0", "unit-2")
        query = ("unit-0", unused)
        selected = training_pairs + (cross, query)
        value_by_pair = {
            tuple(sorted(pair)): value
            for pair, value in zip(
                full_observations.observed_pairs, full_observations.observed_values, strict=True
            )
        }
        observed = PairwiseObservations.from_pairs(
            selected,
            [value_by_pair[tuple(sorted(pair))] for pair in selected],
            target=full_observations.target,
            sampling_unit_ids=region.sampling_unit_ids,
        )
        fold = PopulationFold(
            "incomplete",
            {region.name: held},
            {region.name: ObservationPartition(region.name, training_pairs, "training")},
            {region.name: ObservationPartition(region.name, (query,), "query")},
            query_regime="at_least_one_unseen",
        )
        ensemble = fit_ensemble(
            region,
            observed,
            folds=[fold],
            model_factory=model_factory,
            config=TrainingConfig(objective="mlpe", epochs=0),
        )
        member = ensemble.members[0]
        assert member.status == "completed"
        calibrated = completed_model(member).calibrate(
            region,
            observed,
            partitions=(
                fold.training[region.name],
                ObservationPartition(region.name, (cross,), "calibration"),
            ),
        )
        result = predict_out_of_fold(
            Ensemble((replace(member, model=calibrated),)),
            region,
            [query],
            regime=EvaluationRegime(endpoint_regime="at_least_one_unseen"),
        )
    access = result.access[member.identity.member_id][region.name]
    assert not any(unused in pair for pair in access.training_pairs + access.calibration_pairs)
    assert any(held[0] in pair for pair in access.calibration_pairs)
    assert result.coverage == 0


def test_known_effects_and_support_require_explicit_modes_and_report_separate_variance(
    completed_model, fitted_member_problem
):

    from ilg_toolkit import EvaluationRegime, EvaluationSupport

    with jax.enable_x64():
        region, observations, ensemble = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]
        old = next(label for label in region.sampling_unit_ids if label not in held)
        known_pair = (old, held[0])
        known = predict_out_of_fold(
            ensemble,
            region,
            [known_pair],
            regime=EvaluationRegime(
                endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
            ),
        )
        expected_known = completed_model(member).predict_known_effects(
            region, [tuple(sorted(known_pair))]
        )
        np.testing.assert_allclose(known.values, expected_known.prediction.values, atol=1e-12)
        assert known.member_model_variances is not None
        assert known.member_model_variances.dtype == jnp.float64
        np.testing.assert_allclose(
            known.member_model_variances[0], expected_known.model_variance, atol=1e-12
        )
        assert known.variance_scale == "model"
        assert (
            known.conditioning_provenance[member.identity.member_id][region.name].mode
            == "known_effects"
        )
        support_pair = (held[0], old)
        lookup = {
            tuple(sorted(pair)): value
            for pair, value in zip(
                observations.observed_pairs, observations.observed_values, strict=True
            )
        }
        support_observed = PairwiseObservations.from_pairs(
            [support_pair], [lookup[tuple(sorted(support_pair))]], target=observations.target
        )
        support = EvaluationSupport(
            support_observed, ObservationPartition(region.name, (support_pair,), "support")
        )
        with pytest.raises(ValueError):
            predict_out_of_fold(ensemble, region, [held], support=support)
        strict = predict_out_of_fold(
            ensemble,
            region,
            [held],
            support=support,
            regime=EvaluationRegime(prediction_mode="support"),
        )
        assert strict.coverage == 0
        permitted = EvaluationRegime(
            prediction_mode="support", support_endpoint_policy="allow_declared_support"
        )
        conditional = predict_out_of_fold(
            ensemble, region, [held], support=support, regime=permitted
        )
        expected = completed_model(member).predict_with_support(
            region, [held], support_observed, support_partition=support.partition
        )
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
            conditional.conditioning_provenance[member.identity.member_id][
                region.name
            ].support_pairs
            == support.partition.pairs
        )
        with pytest.raises(ValueError):
            predict_out_of_fold(ensemble, region, [support_pair], support=support, regime=permitted)
        with pytest.raises(ValueError):
            predict_out_of_fold(ensemble, region, [held], regime=permitted)


def test_access_identity_includes_region_and_prediction_failures_remain_visible(
    FeatureEmbedding, completed_model, known_ensemble
):
    from dataclasses import replace

    region, ensemble, _ = known_ensemble()
    valley = replace(region, name="valley")
    members = []
    for member in ensemble.members:
        fold = replace(
            member.fold,
            held_out_units={"alpine": ("c", "d"), "valley": ("c", "d")},
            training={
                name: ObservationPartition(name, (("a", "b"),), "training")
                for name in ("alpine", "valley")
            },
            query={
                name: ObservationPartition(name, (("c", "d"),), "query")
                for name in ("alpine", "valley")
            },
        )
        model = eqx.tree_at(
            lambda model: model.training_pairs,
            completed_model(member),
            {"alpine": (("c", "d"),), "valley": (("a", "b"),)},
        )
        members.append(replace(member, fold=fold, model=model))
    regional = predict_out_of_fold(
        Ensemble(tuple(members)), [region, valley], {"alpine": [("c", "d")], "valley": [("c", "d")]}
    )
    np.testing.assert_array_equal(regional.eligible_counts, [0, 2])
    assert tuple(regional.predictions) == ("alpine", "valley")
    assert regional.predictions["alpine"].region_name == "alpine"
    assert regional.predictions["valley"].region_name == "valley"
    assert np.isnan(regional.predictions["alpine"].values[0])
    np.testing.assert_allclose(regional.predictions["valley"].values, [2.5])
    broken = replace(
        ensemble.members[0],
        model=eqx.tree_at(
            lambda model: model.distance_model,
            completed_model(ensemble.members[0]),
            FeatureEmbedding(jnp.asarray(np.nan)),
        ),
    )
    partial = predict_out_of_fold(Ensemble((broken, ensemble.members[1])), region, [("c", "d")])
    np.testing.assert_array_equal(partial.eligible_counts, [1])
    assert partial.values[0] == 4
    assert partial.exclusion_reasons[broken.identity.member_id] == ("prediction_failed",)


def test_only_eligible_pairs_are_inverse_transformed(FeatureEmbedding, completed_model):

    from ilg_toolkit import TrainingConfig, fit_ensemble

    with jax.enable_x64():
        ids = ("a", "b", "c", "d", "e")
        region = RegionBatch(
            "alpine",
            cx.field(np.arange(5).reshape(1, 5, 1), "row", "column", "feature"),
            ids,
            np.array([[0, i] for i in range(5)]),
        )
        target = TargetSpec("divergence", units="index", transform="sqrt")
        training_pairs = (("a", "b"), ("a", "c"), ("b", "c"))
        # sqrt(target)=4-.5*distance, all observed distances in {1,4}.
        observed = PairwiseObservations.from_pairs(
            training_pairs, [12.25, 4, 12.25], target=target, sampling_unit_ids=ids
        )
        fold = PopulationFold(
            "signed",
            {region.name: ("d", "e")},
            {region.name: ObservationPartition(region.name, training_pairs, "training")},
        )
        ensemble = fit_ensemble(
            region,
            observed,
            folds=[fold],
            config=TrainingConfig(objective="mlpe", epochs=0),
            model_factory=lambda key: FeatureEmbedding(jnp.asarray(1.0)),
        )
        assert ensemble.members[0].status == "completed"
        with pytest.raises(ValueError):
            ensemble.predict(region)  # unrelated (a,e) has distance 16 and fitted mean -4
        marginal = completed_model(ensemble.members[0]).predict(region, [("d", "e")])
        assert marginal.pairs == (("d", "e"),)
        assert marginal.target == target
        assert marginal.region_name == region.name
        assert marginal.scale == "original"
        selected = predict_out_of_fold(ensemble, region, [("d", "e")])
    assert selected.coverage == 1
    np.testing.assert_allclose(selected.values, [12.25], atol=1e-12)
    np.testing.assert_array_equal(selected.values, marginal.values)
    assert selected.prediction_failures == {}


def test_reused_calibration_support_cannot_contribute_and_target_scales_must_match(
    completed_model, fitted_member_problem
):
    from dataclasses import replace

    from ilg_toolkit import EvaluationRegime, EvaluationSupport

    with jax.enable_x64():
        region, observations, ensemble = fitted_member_problem()
        member = ensemble.members[0]
        held = member.fold.held_out_units[region.name]
        calibration = completed_model(member).calibration
        reused = cast(MLPECalibrationModule, calibration).regions[region.name].calibration_pairs[0]
        values = {
            tuple(sorted(pair)): value
            for pair, value in zip(
                observations.observed_pairs, observations.observed_values, strict=True
            )
        }
        support_observed = PairwiseObservations.from_pairs(
            [reused], [values[reused]], target=observations.target
        )
        support = EvaluationSupport(
            support_observed, ObservationPartition(region.name, (reused,), "support")
        )
        rejected = predict_out_of_fold(
            ensemble,
            region,
            [held],
            support=support,
            regime=EvaluationRegime(
                prediction_mode="support", support_endpoint_policy="allow_declared_support"
            ),
        )
        assert rejected.coverage == 0
        prediction = predict_out_of_fold(ensemble, region, [held])
    incompatible = PairwiseObservations.from_pairs(
        [held], [3], target=replace(observations.target, units="percent")
    )
    with pytest.raises(ValueError):
        score_out_of_fold(prediction, incompatible)
    with pytest.raises(ValueError):
        evaluate_ensemble(
            ensemble, region, observations, partitions=member.fold.training[region.name]
        )


def test_direct_pair_predictions_share_oof_policy_and_skip_unrequested_inverse_overflow(
    completed_model, known_ensemble
):
    from dataclasses import replace

    region, ensemble, _ = known_ensemble()
    region = replace(
        region,
        features=cx.field(np.array([0, 100, 1, 2]).reshape(1, 4, 1), "row", "column", "feature"),
    )
    target = TargetSpec("divergence", units="index", transform="log1p")
    members = tuple(
        replace(
            member, model=eqx.tree_at(lambda model: model.target, completed_model(member), target)
        )
        for member in ensemble.members
    )
    ensemble = Ensemble(members)
    expected = [np.expm1(1), np.expm1(4)]
    for member, value in zip(members, expected, strict=True):
        model = completed_model(member)
        with pytest.raises(FloatingPointError):
            model.predict(region)
        selected = model.predict(region, [("d", "c")])
        assert selected.pairs == (("d", "c"),)
        assert selected.target == target
        assert selected.region_name == region.name
        assert selected.scale == "original"
        np.testing.assert_allclose(selected.values, [value], rtol=1e-7)
        matrix = selected.as_matrix()
        assert matrix.shape == (4, 4)
        assert np.isnan(matrix[0, 1])  # Its inverse transform would overflow.
        np.testing.assert_allclose(matrix[3, 2], value, rtol=1e-7)
    out_of_fold = predict_out_of_fold(ensemble, region, [("a", "b"), ("c", "d")])
    np.testing.assert_array_equal(out_of_fold.eligible_counts, [0, 2])
    np.testing.assert_allclose(out_of_fold.values[1], np.mean(expected), rtol=1e-7)
    assert out_of_fold.prediction_failures == {}


def test_pair_queries_require_unique_distinct_labels_with_query_locations(
    completed_model, known_ensemble
):

    region, ensemble, _ = known_ensemble()
    model = completed_model(ensemble.members[0])
    for pairs in ([], [("a", "a")], [("a", "b"), ("b", "a")], [("a",)], ["ab"], [("a", "unknown")]):
        with pytest.raises(ValueError):
            model.predict(region, pairs)
