"""Artifact workflow consumers retained for the native ensemble migration (#62)."""

from dataclasses import replace
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from test_checkpoint import assert_same_state, checkpoint_problem
from test_persistence import problem

from ilg_toolkit import (
    PairwiseObservations,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    fit,
    load_checkpoint,
    load_model,
    save_checkpoint,
    save_model,
)


def test_native_artifact_preserves_out_of_fold_access(tmp_path):
    from dataclasses import replace

    from ilg_toolkit import ObservationPartition
    from ilg_toolkit.models import EmbeddingDistanceModel

    @final
    class ScaledEmbedding(EmbeddingDistanceModel):
        log_weights: jax.Array
        patch_size: int = eqx.field(static=True, default=1)

        def __init__(self, log_weights: jax.Array):
            self.log_weights = log_weights

        def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
            return features * jnp.exp(self.log_weights)

    region, observations = problem()
    other = replace(region, name="second-calibration")
    other_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2.0, target=observations.target
    )
    labels = set(region.sampling_unit_ids[:4])
    pairs = tuple(pair for pair in observations.observed_pairs if set(pair) <= labels)
    training = ObservationPartition(region.name, pairs[:-1], role="training")
    validation = ObservationPartition(region.name, pairs[-1:], role="validation")
    model = fit(
        (region, other),
        (observations, other_observations),
        distance_model=ScaledEmbedding(jnp.asarray([0.2, -0.7])),
        config=TrainingConfig(epochs=0, objective="mlpe"),
        partition=(training, replace(training, region_name=other.name)),
        validation=(region, observations),
        validation_partition=validation,
    ).model
    path = tmp_path / "custom-mlpe.ilg"
    save_model(path, model)
    restored = load_model(path, distance_model=ScaledEmbedding(jnp.asarray([4.0, 3.0])))
    assert type(restored.distance_model) is ScaledEmbedding
    for prepared in (region, other):
        np.testing.assert_array_equal(
            restored.predict(prepared).values, model.predict(prepared).values
        )
        assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
        assert isinstance(restored.calibration, UnpooledMLPECalibrationModule)
        assert eqx.tree_equal(
            restored.calibration.regions[prepared.name], model.calibration.regions[prepared.name]
        )
    assert restored.training_pairs == model.training_pairs
    assert restored.validation_pairs == model.validation_pairs
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    before = model.predict_known_effects(region, query)
    after = restored.predict_known_effects(region, query)
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0")], [1.5], target=model.target
    )
    declared = ObservationPartition(region.name, support.observed_pairs, role="support")
    before = model.predict_with_support(region, query, support, support_partition=declared)
    after = restored.predict_with_support(region, query, support, support_partition=declared)
    np.testing.assert_array_equal(after.prediction.values, before.prediction.values)
    np.testing.assert_array_equal(after.model_variance, before.model_variance)
    assert after.provenance == before.provenance
    assert isinstance(restored.calibration, UnpooledMLPECalibrationModule)
    assert restored.calibration.regions[region.name].converged is False

    from ilg_toolkit import (
        Ensemble,
        EnsembleMember,
        MemberIdentity,
        PopulationFold,
        predict_out_of_fold,
    )

    fold = PopulationFold(
        "held-out",
        {region.name: region.sampling_unit_ids[-2:]},
        {region.name: training},
        validation={region.name: validation},
    )
    evaluation_pairs = (
        validation.pairs[0],
        ("population-2", "population-4"),
        ("population-4", "population-5"),
    )

    def evaluate(calibrated):
        member = EnsembleMember(MemberIdentity(fold.fold_id, 3), fold, "completed", calibrated)
        return predict_out_of_fold(Ensemble((member,)), region, evaluation_pairs)

    before_evaluation, after_evaluation = evaluate(model), evaluate(restored)
    np.testing.assert_array_equal(after_evaluation.eligible_counts, [0, 0, 1])
    np.testing.assert_array_equal(after_evaluation.values, before_evaluation.values)
    assert after_evaluation.access == before_evaluation.access
    assert after_evaluation.exclusion_reasons == before_evaluation.exclusion_reasons


def test_individual_ensemble_member_disk_resume_uses_a_custom_encoder(tmp_path):
    from test_mlpe_training import WeightedEmbedding

    from ilg_toolkit import fit_ensemble_member, generate_population_folds

    region, observations = checkpoint_problem()
    fold = generate_population_folds(region, observations, n_folds=1, holdout_size=2, seed=47)[0]
    config = TrainingConfig(objective="direct_log1p", epochs=2, learning_rate=0.002)

    def factory(key):
        return WeightedEmbedding(jax.random.normal(key, (2,)) * 0.1)

    uninterrupted = fit_ensemble_member(
        region,
        observations,
        config=config,
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    path = tmp_path / "fold-0-seed-13.eqx"
    partial = fit_ensemble_member(
        region,
        observations,
        config=replace(config, epochs=1),
        on_epoch=lambda identity, state: save_checkpoint(path, state),
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    assert partial.status == uninterrupted.status == "completed"
    state = load_checkpoint(path, distance_model=factory(jax.random.PRNGKey(13)))
    resumed = fit_ensemble_member(
        region,
        observations,
        config=config,
        state=state,
        fold=fold,
        initialization_seed=13,
        model_factory=factory,
    )
    assert resumed.status == "completed"
    assert resumed.identity == uninterrupted.identity
    assert resumed.training_state is not None and uninterrupted.training_state is not None
    assert_same_state(resumed.training_state, uninterrupted.training_state)
    assert resumed.model is not None and uninterrupted.model is not None
    np.testing.assert_array_equal(
        resumed.model.predict(region).values, uninterrupted.model.predict(region).values
    )
