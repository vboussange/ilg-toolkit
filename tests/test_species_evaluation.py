"""Independent group coverage, provenance and held-out study-design references."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from ensemble_helpers import labelled_ensemble
from test_species_conditioning import conditioned_species as conditioned_species

from ilg_toolkit import (
    Ensemble,
    ObservationPartition,
    PairwiseObservations,
    predict_out_of_fold,
    score_out_of_fold,
)


def test_group_and_species_metrics_score_unique_queries_once_and_keep_missing_coverage():
    region, ensemble, target = labelled_ensemble()
    valley = replace(
        region,
        name="valley",
        sampling_units={
            "fox": replace(region.sampling_units["fox"], ids=("v-a", "v-b", "v-c", "v-d"))
        },
    )
    members = []
    for index, member in enumerate(ensemble.members):
        assert member.model is not None
        fold = replace(
            member.fold,
            held_out_units={**member.fold.held_out_units, "valley": {"fox": ("v-c", "v-d")}},
            training={
                **member.fold.training,
                "valley": {
                    "fox": ObservationPartition("valley", "fox", (("v-a", "v-b"),), "training")
                },
            },
        )
        training = {
            "alpine": {
                "fox": (("a", "b"),),
                "lynx": (("c", "d"),) if index == 0 else (("a", "b"),),
            },
            "valley": {"fox": (("v-a", "v-b"),)},
        }
        validation = {"alpine": {"fox": (("c", "d"),)}} if index == 0 else {}
        model = eqx.tree_at(
            lambda m: (m.training_pairs, m.validation_pairs), member.model, (training, validation)
        )
        if index == 1:
            model = eqx.tree_at(lambda m: m.distance_model.weights, model, jnp.array([2.0, np.nan]))
        members.append(replace(member, model=model, fold=fold))
    queries = {
        "alpine": {"fox": [("d", "c"), ("a", "b"), ("c", "d")], "lynx": [("c", "d")]},
        "valley": {"fox": [("v-a", "v-c"), ("v-c", "v-d")]},
    }
    prediction = predict_out_of_fold(Ensemble(tuple(members)), [valley, region], queries)
    assert prediction.keys == (
        ("alpine", "fox", ("a", "b")),
        ("alpine", "fox", ("c", "d")),
        ("alpine", "lynx", ("c", "d")),
        ("valley", "fox", ("v-a", "v-c")),
        ("valley", "fox", ("v-c", "v-d")),
    )
    np.testing.assert_array_equal(prediction.eligible_counts, [0, 1, 0, 0, 2])
    np.testing.assert_allclose(prediction.values, [np.nan, 4, np.nan, np.nan, 2.5])
    np.testing.assert_allclose(prediction.member_spread, [np.nan, 0, np.nan, np.nan, 1.5])
    failed = prediction.prediction_failures[members[1].identity.member_id]["alpine"]
    assert set(failed) == {"lynx"}
    assert prediction.access[members[0].identity.member_id]["alpine"]["fox"].species_name == "fox"
    observed = {
        "alpine": {
            "fox": PairwiseObservations.from_pairs(
                [("a", "b"), ("c", "d")], [99, 5], target=target
            ),
            "lynx": PairwiseObservations.from_pairs([("c", "d")], [900], target=target),
        },
        "valley": {
            "fox": PairwiseObservations.from_pairs(
                [("v-a", "v-c"), ("v-c", "v-d")], [800, 4.5], target=target
            )
        },
    }
    scored = score_out_of_fold(prediction, observed)
    assert scored.n_pairs == 2 and scored.n_query_pairs == 5 and scored.coverage == 0.4
    assert scored.mse == 2.5 and scored.mae == 1.5
    assert scored.by_group["alpine"]["fox"].mse == 1
    assert scored.by_group["valley"]["fox"].mse == 4
    assert scored.by_species["fox"].n_pairs == 2
    assert scored.by_species["fox"].n_query_pairs == 4
    assert scored.by_species["fox"].mse == 2.5
    assert scored.by_species["lynx"].coverage == 0
    assert scored.by_species["lynx"].n_query_pairs == 1
    assert scored.by_species["lynx"].mse is None
    assert scored.by_group["alpine"]["lynx"].rmse is None
    assert scored.by_group["alpine"]["lynx"].mae is None


def test_disjoint_development_and_common_outer_test_keep_measurements_out_of_prediction():
    from itertools import combinations

    import coordax as cx
    import jax
    from ensemble_helpers import SpeciesEmbedding

    from ilg_toolkit import (
        EvaluationRegime,
        PopulationFold,
        RegionBatch,
        SamplingUnits,
        TargetSpec,
        TrainingConfig,
        UnpooledMLPECalibrationModule,
        fit_ensemble,
    )

    rng = np.random.default_rng(72)
    features = rng.normal(size=(1, 9, 2))
    species = ("fox", "lynx")
    # These spatial assignments are coordinated across species by the caller.
    units = {
        name: SamplingUnits(tuple(str(i) for i in range(n)), np.array([[0, i] for i in range(n)]))
        for name, n in [("fox", 8), ("lynx", 9)]
    }
    region = RegionBatch("study", cx.field(features, "row", "column", "feature"), units)
    target = TargetSpec("divergence", units="index")
    validation_ids = ("3", "4")
    train_sets = (("0", "1", "2"), ("0", "1", "5"))
    development_pairs = tuple(
        sorted(
            set(combinations(train_sets[0], 2))
            | set(combinations(train_sets[1], 2))
            | {validation_ids}
        )
    )

    def measured(name, pairs, shift=0):
        index = species.index(name)
        points = features[0] * np.array([0.5 + index, 1.7])
        values = [
            2
            + index
            + 0.25 * np.sum((points[int(a)] - points[int(b)]) ** 2)
            + 0.03 * (int(a) + int(b))
            + shift
            for a, b in pairs
        ]
        return PairwiseObservations.from_pairs(pairs, values, target=target)

    development = {region.name: {name: measured(name, development_pairs) for name in species}}
    folds = []
    for index, train_ids in enumerate(train_sets):
        folds.append(
            PopulationFold(
                f"coordinated-{index}",
                {
                    region.name: {
                        name: tuple(label for label in u.ids if label not in train_ids)
                        for name, u in units.items()
                    }
                },
                {
                    region.name: {
                        name: ObservationPartition(
                            region.name, name, tuple(combinations(train_ids, 2)), "training"
                        )
                        for name in species
                    }
                },
                validation={
                    region.name: {
                        name: ObservationPartition(
                            region.name, name, (validation_ids,), "validation"
                        )
                        for name in species
                    }
                },
            )
        )

    def factory(key):
        return SpeciesEmbedding(species=species, weights=0.5 + jax.random.uniform(key, (2, 2)))

    ensemble = fit_ensemble(
        [region],
        development,
        folds=folds,
        initialization_seeds=(17,),
        config=TrainingConfig(objective="mlpe", steps=2),
        model_factory=factory,
    )
    assert all(member.status == "completed" for member in ensemble.members)
    for member, train_ids in zip(ensemble.members, train_sets, strict=True):
        assert member.model is not None and member.training_state is not None
        assert isinstance(member.model.calibration, UnpooledMLPECalibrationModule)
        assert member.training_state.selection == "validation"
        losses = [(record.validation_loss, record.step) for record in member.training_state.history]
        assert member.training_state.selected_step == min(losses)[1]
        for name in species:
            assert member.model.training_pairs[region.name][name] == tuple(
                combinations(train_ids, 2)
            )
            assert member.model.validation_pairs[region.name][name] == (validation_ids,)
            head = member.model.calibration.for_group(region.name, name)
            assert head.calibration_pairs == tuple(combinations(train_ids, 2))
            assert head.calibration_roles == ("training",) * 3
    # Test measurements have not even been constructed yet. Cross-partition edges
    # are excluded from this common outer-test estimand.
    outer = {
        name: tuple(combinations(tuple(label for label in u.ids if int(label) >= 6), 2))
        for name, u in units.items()
    }
    deployed = ensemble.predict(region, species=species, pairs=outer)
    fixed = predict_out_of_fold(ensemble, [region], {region.name: outer})
    np.testing.assert_array_equal(fixed.eligible_counts, np.full(4, 2))
    for name in species:
        reference = np.stack(
            [
                member.model.predict(region, species=name, pairs={name: outer[name]})[name].values
                for member in ensemble.members
                if member.model is not None
            ]
        )
        np.testing.assert_allclose(
            deployed[name].prediction.values, reference.mean(axis=0), rtol=1e-12
        )
        np.testing.assert_allclose(
            fixed.predictions[region.name][name].values, reference.mean(axis=0), rtol=1e-12
        )
    observations = {region.name: {name: measured(name, pairs) for name, pairs in outer.items()}}
    baseline = score_out_of_fold(fixed, observations)
    changed = score_out_of_fold(
        fixed,
        {region.name: {name: measured(name, pairs, shift=100) for name, pairs in outer.items()}},
    )
    assert baseline.mse != changed.mse
    assert changed.predictions is fixed
    # OOF can instead use just one member. Population2 is training in member0,
    # but an untouched nominal development holdout in member1.
    cross = {region.name: {name: [("2", "6"), ("0", "6"), validation_ids] for name in species}}
    strict = predict_out_of_fold(ensemble, [region], cross)
    permissive = predict_out_of_fold(
        ensemble, [region], cross, regime=EvaluationRegime(endpoint_regime="at_least_one_unseen")
    )
    np.testing.assert_array_equal(strict.eligible_counts, [0, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(permissive.eligible_counts, [2, 2, 0, 2, 2, 0])
    assert strict.coverage < 1 and fixed.coverage == 1


@pytest.mark.parametrize("overflow", ["point", "variance"])
def test_finite_conditional_outputs_that_overflow_storage_remain_explicit_group_failures(overflow):
    import jax

    from ilg_toolkit import (
        EvaluationRegime,
        MLPECalibration,
        MLPEConfig,
        UnpooledMLPECalibrationModule,
    )

    with jax.enable_x64(False):
        region, ensemble, target = labelled_ensemble()
        member = ensemble.members[0]
        assert member.model is not None
        with jax.enable_x64():
            heads = {}
            for name in ("fox", "lynx"):
                heads[name] = MLPECalibration(
                    region_name=region.name,
                    species_name=name,
                    target=target,
                    population_ids=("a", "b", "c", "d"),
                    distance_center=0.0,
                    distance_scale=1.0,
                    intercept=1e100 if name == "fox" and overflow == "point" else 7.0,
                    slope=0.0,
                    unit_variance=1.0,
                    residual_variance=1e100 if name == "fox" and overflow == "variance" else 1.0,
                    effect_mean=jnp.zeros(4),
                    effect_covariance=jnp.eye(4),
                    effect_precision_cholesky=jnp.eye(4),
                    calibration_pairs=(("a", "b"),),
                    calibration_roles=("training",),
                    ml_log_likelihood=0.0,
                    config=MLPEConfig(),
                    optimizer_iterations=0,
                    optimizer_message="independent synthetic posterior",
                )
            model = eqx.tree_at(
                lambda m: m.calibration,
                member.model,
                UnpooledMLPECalibrationModule(target, {region.name: heads}),
            )
        direct = model.predict_known_effects(region, species="fox", pairs={"fox": (("c", "d"),)})[
            "fox"
        ]
        assert (
            np.isfinite(direct.prediction.values).all() and np.isfinite(direct.model_variance).all()
        )
        prediction = predict_out_of_fold(
            Ensemble((replace(member, model=model),)),
            [region],
            {region.name: {name: [("c", "d")] for name in ("fox", "lynx")}},
            regime=EvaluationRegime(prediction_mode="known_effects"),
        )
        np.testing.assert_array_equal(prediction.eligible_counts, [0, 1])
        assert np.isnan(prediction.member_values[0, 0])
        assert prediction.member_model_variances is not None
        assert np.isnan(prediction.member_model_variances[0, 0])
        assert (
            prediction.member_values.dtype == prediction.member_model_variances.dtype == jnp.float32
        )
        assert prediction.values[1] == 7
        assert prediction.exclusion_reasons[member.identity.member_id] == (
            "prediction_failed",
            None,
        )
        failure = prediction.prediction_failures[member.identity.member_id][region.name]["fox"]
        assert failure.error_type == "FloatingPointError"
        assert "storage" in failure.message
        assert not jax.config.read("jax_enable_x64")


def test_known_and_support_oof_keep_reused_labels_and_uncertainty_separate(conditioned_species):
    from ilg_toolkit import (
        EnsembleMember,
        EvaluationRegime,
        EvaluationSupport,
        MemberIdentity,
        PopulationFold,
    )

    model, region, _, _, support, partitions = conditioned_species
    held = ("population-4", "population-5")
    training = {
        region.name: {
            name: ObservationPartition(region.name, name, pairs, "training")
            for name, pairs in model.training_pairs[region.name].items()
        }
    }
    fold = PopulationFold("f", {region.name: {name: held for name in model.species}}, training)
    ensemble = Ensemble((EnsembleMember(MemberIdentity("f", 1), fold, "completed", model),))
    queries = {region.name: {name: [held] for name in model.species}}
    known = predict_out_of_fold(
        ensemble, [region], queries, regime=EvaluationRegime(prediction_mode="known_effects")
    )
    declared = {
        region.name: {
            name: EvaluationSupport(support[name], partitions[name]) for name in model.species
        }
    }
    strict = predict_out_of_fold(
        ensemble,
        [region],
        queries,
        support=declared,
        regime=EvaluationRegime(prediction_mode="support"),
    )
    np.testing.assert_array_equal(strict.eligible_counts, [0, 0])
    regime = EvaluationRegime(
        prediction_mode="support", support_endpoint_policy="allow_declared_support"
    )
    conditional = predict_out_of_fold(ensemble, [region], queries, support=declared, regime=regime)
    np.testing.assert_array_equal(conditional.eligible_counts, [1, 1])
    assert (
        known.member_model_variances is not None and conditional.member_model_variances is not None
    )
    for column, name in enumerate(model.species):
        reference = model.predict_with_support(
            region,
            {name: support[name]},
            species=name,
            pairs={name: [held]},
            support_partition={name: partitions[name]},
        )[name]
        np.testing.assert_allclose(
            conditional.values[column], reference.prediction.values[0], rtol=1e-12
        )
        np.testing.assert_allclose(
            conditional.member_model_variances[0, column], reference.model_variance[0], rtol=1e-12
        )
        assert (
            conditional.conditioning_provenance["f:seed=1"][region.name][name]
            == reference.provenance
        )
        assert conditional.access["f:seed=1"][region.name][name].species_name == name
    changed_support = PairwiseObservations.from_pairs(
        support["fox"].observed_pairs, support["fox"].observed_values + 10, target=model.target
    )
    changed_declaration = {
        region.name: {
            **declared[region.name],
            "fox": EvaluationSupport(changed_support, partitions["fox"]),
        }
    }
    changed = predict_out_of_fold(
        ensemble, [region], queries, support=changed_declaration, regime=regime
    )
    assert changed.values[0] != conditional.values[0]
    assert changed.values[1] == conditional.values[1]
    assert conditional.variance_scale == "model"
    np.testing.assert_array_equal(conditional.member_spread, [0, 0])


def test_conditional_pair_role_associations_are_validated_together(
    conditioned_species, monkeypatch
):
    from ilg_toolkit import (
        EnsembleMember,
        EvaluationRegime,
        LandscapeGeneticsModel,
        MemberIdentity,
        PopulationFold,
    )

    model, region, observations, _, _, _ = conditioned_species
    train = ObservationPartition(
        region.name, "fox", model.training_pairs[region.name]["fox"], "training"
    )
    extra = ObservationPartition(
        region.name, "fox", (("population-0", "population-4"),), "calibration"
    )
    model = model.calibrate(
        region, {"fox": observations["fox"]}, partitions={"fox": (train, extra)}
    )
    held = ("population-4", "population-5")
    fold = PopulationFold(
        "f",
        {region.name: {name: held for name in model.species}},
        {
            region.name: {
                name: ObservationPartition(region.name, name, pairs, "training")
                for name, pairs in model.training_pairs[region.name].items()
            }
        },
    )
    original = LandscapeGeneticsModel.predict_known_effects

    def shuffled_access(self, region, *, species=None, pairs=None):
        output = original(self, region, species=species, pairs=pairs)
        if "fox" in output:
            result = output["fox"]
            # Preserve the pair set AND the role tuple, but assign roles to the
            # wrong pairs. Independent pair/role comparisons would accept this.
            provenance = replace(
                result.provenance,
                calibration_pairs=result.provenance.calibration_pairs[1:]
                + result.provenance.calibration_pairs[:1],
            )
            output["fox"] = eqx.tree_at(lambda value: value.provenance, result, provenance)
        return output

    monkeypatch.setattr(LandscapeGeneticsModel, "predict_known_effects", shuffled_access)
    result = predict_out_of_fold(
        Ensemble((EnsembleMember(MemberIdentity("f", 1), fold, "completed", model),)),
        [region],
        {region.name: {name: [held] for name in model.species}},
        regime=EvaluationRegime(
            endpoint_regime="at_least_one_unseen", prediction_mode="known_effects"
        ),
    )
    np.testing.assert_array_equal(result.eligible_counts, [0, 1])
    assert "provenance" in result.prediction_failures["f:seed=1"][region.name]["fox"].message


@pytest.mark.parametrize("wrong", ["species", "region", "target", "order"])
def test_marginal_output_identity_and_order_failures_remain_group_local(monkeypatch, wrong):
    from ilg_toolkit import LandscapeGeneticsModel

    region, ensemble, _ = labelled_ensemble()
    original = LandscapeGeneticsModel.predict

    def wrong_identity(self, region, *, species=None, pairs=None):
        output = original(self, region, species=species, pairs=pairs)
        if "fox" in output:
            point = output["fox"]
            if wrong == "species":
                point = eqx.tree_at(lambda p: p.species_name, point, "lynx")
            elif wrong == "region":
                point = eqx.tree_at(lambda p: p.region_name, point, "elsewhere")
            elif wrong == "target":
                point = eqx.tree_at(
                    lambda p: p.target, point, replace(point.target, units="percent")
                )
            else:
                point = eqx.tree_at(lambda p: p.pairs, point, tuple((b, a) for a, b in point.pairs))
            output["fox"] = point
        return output

    monkeypatch.setattr(LandscapeGeneticsModel, "predict", wrong_identity)
    result = predict_out_of_fold(
        ensemble, [region], {region.name: {name: [("c", "d")] for name in ("fox", "lynx")}}
    )
    np.testing.assert_array_equal(result.eligible_counts, [0, 2])
    for member in ensemble.members:
        assert set(result.prediction_failures[member.identity.member_id][region.name]) == {"fox"}


def test_fitted_mlpe_marginal_prediction_preserves_stored_precision_under_float32_policy():
    import jax
    from test_mlpe_training import WeightedEmbedding, embedding_problem

    from ilg_toolkit import TrainingConfig, fit_ensemble, generate_population_folds

    region, observed = embedding_problem()
    observations = {region.name: {"species": observed}}
    folds = generate_population_folds([region], observations, n_folds=1, holdout_size=2, seed=47)
    ensemble = fit_ensemble(
        [region],
        observations,
        folds=folds,
        config=TrainingConfig(objective="mlpe", steps=0),
        model_factory=lambda key: WeightedEmbedding(jnp.zeros(2, dtype=jnp.float32)),
    )
    member = ensemble.members[0]
    assert member.model is not None
    pairs = folds[0].query[region.name]["species"].pairs
    expected = member.model.predict(region, pairs={"species": pairs})["species"]
    with jax.enable_x64(False):
        marginal = member.model.predict(region, pairs={"species": pairs})["species"]
        assert marginal.values.dtype == expected.values.dtype == jnp.float64
        np.testing.assert_allclose(marginal.values, expected.values, rtol=1e-6)
        assert marginal.as_matrix().dtype == jnp.float64
        predicted = predict_out_of_fold(ensemble, [region], {region.name: {"species": pairs}})
        assert predicted.member_values.dtype == jnp.float32
        assert predicted.coverage == 1 and predicted.prediction_failures == {}
        np.testing.assert_allclose(predicted.values, expected.values, rtol=1e-6)
        assert not jax.config.read("jax_enable_x64")


def test_oof_group_collections_reject_conflicting_sampling_unit_region_assignments():
    region, ensemble, _ = labelled_ensemble()
    moved = replace(region, name="valley")
    queries = {name: {"fox": [("c", "d")]} for name in ("alpine", "valley")}
    with pytest.raises(ValueError, match="region assignment"):
        predict_out_of_fold(ensemble, [region, moved], queries)


def test_complete_training_manifest_can_leave_one_species_group_without_target_access():
    region, ensemble, _ = labelled_ensemble()
    member = ensemble.members[0]
    assert member.model is not None
    model = eqx.tree_at(
        lambda m: m.training_pairs, member.model, {region.name: {"fox": (("c", "d"),)}}
    )
    declared = Ensemble((replace(member, model=model),))
    queries = {region.name: {name: [("c", "d")] for name in ("fox", "lynx")}}
    result = predict_out_of_fold(declared, [region], queries)
    np.testing.assert_array_equal(result.eligible_counts, [0, 1])
    assert result.access[member.identity.member_id][region.name]["lynx"].training_pairs == ()
    unknown = eqx.tree_at(lambda m: m.training_pairs, model, {})
    excluded = predict_out_of_fold(Ensemble((replace(member, model=unknown),)), [region], queries)
    assert excluded.exclusion_reasons[member.identity.member_id] == ("unknown_encoder_access",) * 2
