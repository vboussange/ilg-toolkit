"""Native artifact workflows retain OOF access and exact member continuation."""

import numpy as np
import pytest
from artifact_helpers import NativeEmbedding, checkpoint_problem
from test_checkpoint import assert_same_state
from test_persistence import problem

from ilg_toolkit import (
    Ensemble,
    EnsembleMember,
    MemberIdentity,
    ObservationPartition,
    PairwiseObservations,
    PopulationFold,
    TrainingConfig,
    fit,
    fit_ensemble_member,
    generate_population_folds,
    load_checkpoint,
    load_model,
    predict_out_of_fold,
    save_checkpoint,
    save_model,
)


def test_native_artifact_preserves_out_of_fold_access(tmp_path):
    region, observations = problem()
    labels = set(region.sampling_units["fox"].ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    training = ObservationPartition(region.name, "fox", pairs[:-1], role="training")
    validation = ObservationPartition(region.name, "fox", pairs[-1:], role="validation")
    model = fit(
        [region],
        {region.name: {"fox": observations}},
        distance_model=NativeEmbedding(species=("fox",)),
        config=TrainingConfig(steps=0, objective="mlpe"),
        partition={region.name: {"fox": training}},
        validation=([region], {region.name: {"fox": observations}}),
        validation_partition={region.name: {"fox": validation}},
    ).model
    path = tmp_path / "custom-mlpe.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=NativeEmbedding(species=("fox",), seed=71))
    assert restored.training_pairs == model.training_pairs
    assert restored.validation_pairs == model.validation_pairs
    np.testing.assert_array_equal(
        restored.predict(region)["fox"].values, model.predict(region)["fox"].values
    )
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict_known_effects(region, pairs={"fox": query})["fox"]
    after = restored.predict_known_effects(region, pairs={"fox": query})["fox"]
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    declared = ObservationPartition(region.name, "fox", support.observed_pairs, role="support")
    before = model.predict_with_support(
        region, {"fox": support}, pairs={"fox": query}, support_partition={"fox": declared}
    )["fox"]
    after = restored.predict_with_support(
        region, {"fox": support}, pairs={"fox": query}, support_partition={"fox": declared}
    )["fox"]
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    fold = PopulationFold(
        "held-out",
        {region.name: {"fox": region.sampling_units["fox"].ids[-2:]}},
        {region.name: {"fox": training}},
        validation={region.name: {"fox": validation}},
    )
    evaluation_pairs = (
        validation.pairs[0],
        ("population-2", "population-4"),
        ("population-4", "population-5"),
    )

    def evaluate(calibrated):
        member = EnsembleMember(MemberIdentity(fold.fold_id, 3), fold, "completed", calibrated)
        return predict_out_of_fold(
            Ensemble((member,)), [region], {region.name: {"fox": evaluation_pairs}}
        )

    before_evaluation, after_evaluation = evaluate(model), evaluate(restored)
    np.testing.assert_array_equal(after_evaluation.eligible_counts, [0, 0, 1])
    np.testing.assert_array_equal(after_evaluation.values, before_evaluation.values)
    assert after_evaluation.access == before_evaluation.access
    assert after_evaluation.exclusion_reasons == before_evaluation.exclusion_reasons


def test_individual_ensemble_member_disk_resume_uses_a_custom_encoder(tmp_path):
    region, observations = checkpoint_problem()
    nested = {region.name: observations}
    fold = generate_population_folds([region], nested, n_folds=1, holdout_size=2, seed=47)[0]
    config = TrainingConfig(
        objective="direct_log1p", steps=3, evaluation_interval=2, learning_rate=0.002
    )
    calls = []

    def factory(key):
        calls.append(np.asarray(key).copy())
        return NativeEmbedding(seed=int(np.asarray(key)[1]))

    uninterrupted = fit_ensemble_member(
        [region], nested, config=config, fold=fold, initialization_seed=13, model_factory=factory
    )
    path = tmp_path / "fold-0-seed-13.eqx"
    seen = []

    def save_and_interrupt(identity, state):
        seen.append(state.step)
        assert identity.initialization_seed == 13
        assert [record.step for record in state.history] == [0]
        save_checkpoint(path, state)
        raise RuntimeError("saved before scheduled evaluation")

    with pytest.raises(RuntimeError, match="saved before scheduled evaluation"):
        fit_ensemble_member(
            [region],
            nested,
            config=config,
            on_step=save_and_interrupt,
            fold=fold,
            initialization_seed=13,
            model_factory=factory,
        )
    assert seen == [1]
    state = load_checkpoint(path, distance_model=NativeEmbedding(seed=91))
    assert state.step == 1 and state.selected_step == 0
    resumed = fit_ensemble_member(
        [region],
        nested,
        config=config,
        state=state,
        fold=fold,
        initialization_seed=13,
        on_step=lambda identity, state: seen.append(state.step),
    )
    assert len(calls) == 2
    assert seen == [1, 2, 3]
    assert resumed.status == uninterrupted.status == "completed"
    assert resumed.identity == uninterrupted.identity
    assert resumed.training_state is not None and uninterrupted.training_state is not None
    assert_same_state(resumed.training_state, uninterrupted.training_state)
    assert resumed.model is not None and uninterrupted.model is not None
    for species in ("fox", "lynx"):
        np.testing.assert_array_equal(
            resumed.model.predict(region, species=[species])[species].values,
            uninterrupted.model.predict(region, species=[species])[species].values,
        )
