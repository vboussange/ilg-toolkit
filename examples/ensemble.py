"""Run a complete synthetic deployment/OOF workflow without private data.

From the repository root: ``JAX_PLATFORMS=cpu python examples/ensemble.py``.
The explicit x64 scope supports joint MLPE fitting; predictions are fixed before
query targets are supplied to scoring. Printed identities/counts expose exactly
which pairs have coverage and what each uncertainty diagnostic means.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import coordax as cx
import equinox as eqx
import jax
import numpy as np

from ilg_toolkit import (
    Ensemble,
    EvaluationRegime,
    EvaluationSupport,
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    fit_ensemble,
    fit_ensemble_member,
    load_checkpoint,
    predict_out_of_fold,
    save_checkpoint,
    score_out_of_fold,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def model_factory(key):
    # Each member receives jax.random.PRNGKey(its declared initialization seed).
    return UNetEmbeddingDistance(
        2, species=("fox",), patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=key
    )


class Interrupted(Exception):
    """Simulate interruption after a caller-owned checkpoint is safely written."""


def continue_members(directory, region, observations, folds, config):
    """Interrupt each original two-step budget after step 1, then restore it."""
    directory = Path(directory)
    members = []
    for fold in folds:
        for seed in (13, 29):
            checkpoint = directory / f"{fold.fold_id}-seed-{seed}.ilg"

            def save_and_interrupt(identity, current, checkpoint=checkpoint):
                if current.step == 1:
                    assert [record.step for record in current.history] == [0]
                    save_checkpoint(checkpoint, current)
                    raise Interrupted(identity.member_id)

            reference = fit_ensemble_member(
                [region],
                observations,
                fold=fold,
                initialization_seed=seed,
                config=config,
                model_factory=model_factory,
            )
            try:
                fit_ensemble_member(
                    [region],
                    observations,
                    fold=fold,
                    initialization_seed=seed,
                    config=config,
                    model_factory=model_factory,
                    on_step=save_and_interrupt,
                )
            except Interrupted:
                pass
            else:
                raise AssertionError("Expected interruption after the first update")
            state = load_checkpoint(
                checkpoint,
                distance_model=model_factory(jax.random.PRNGKey(seed)),
            )
            assert state.step == 1 and state.config.steps == 2
            continued = fit_ensemble_member(
                [region],
                observations,
                fold=fold,
                initialization_seed=seed,
                state=state,
            )
            assert reference.training_state is not None and continued.training_state is not None
            assert eqx.tree_equal(reference.training_state, continued.training_state)
            members.append(continued)
    return Ensemble(tuple(members))


with jax.enable_x64():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(4, 4, 2)).astype(np.float32)
    positions = np.array([[0, 0], [0, 3], [1, 1], [2, 2], [3, 0], [3, 3]])
    labels = tuple(f"population-{i}" for i in range(len(positions)))
    region = RegionBatch(
        "synthetic",
        cx.field(
            features,
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation", "habitat"])),
        ),
        {"fox": SamplingUnits(labels, positions)},
    )
    embeddings = features[tuple(positions.T)] * np.array([0.3, 1.5])
    distances = ((embeddings[:, None] - embeddings[None, :]) ** 2).sum(-1)
    left, right = np.triu_indices(len(labels), 1)
    targets = 1 + 0.1 * distances[left, right] + rng.normal(0, 0.02, len(left))
    observations = PairwiseObservations.from_pairs(
        [(labels[i], labels[j]) for i, j in zip(left, right, strict=True)],
        targets,
        target=TargetSpec("synthetic divergence", units="index", transform="log1p"),
        sampling_unit_ids=labels,
    )
    grouped = {region.name: {"fox": observations}}
    ensemble = fit_ensemble(
        [region],
        grouped,
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(13, 29),
        model_factory=model_factory,
        config=TrainingConfig(objective="mlpe", steps=1, learning_rate=0.001),
    )
    if ensemble.failures:
        raise RuntimeError(ensemble.failures)
    deployment = ensemble.predict(region)["fox"]
    print("Independent members:", deployment.member_ids)
    print("Deployment averages every member AFTER inverse transformation; units: index")
    print("Deployment pairs:", deployment.prediction.pairs)
    print(deployment.prediction.values)
    print("Descriptive member SD (separate from MLPE predictive variance):")
    print(deployment.member_spread)

    # Request identities only. No query measurements enter this prediction call.
    oof = predict_out_of_fold(
        ensemble, [region], {region.name: {"fox": observations.observed_pairs}}
    )
    print("OOF region/pair, contributors, mean (NaN means uncovered):")
    for identity, count, value in zip(oof.keys, oof.eligible_counts, oof.values, strict=True):
        print(identity, int(count), float(value))
    print("OOF coverage status:", oof.coverage_status)
    print("Fit/prediction failures:", ensemble.failures, oof.prediction_failures)

    # The prediction is fixed; only now supply measured query targets for errors.
    evaluation = score_out_of_fold(oof, grouped)
    print("OOF covered unique pairs:", evaluation.n_pairs, "of", evaluation.n_query_pairs)
    print("OOF coverage, RMSE (index):", evaluation.coverage, evaluation.rmse)
    print("Each covered pair is scored once after averaging its eligible member points.")
    print("Training pairs stay present with zero coverage and do not contribute errors.")
    assert oof.coverage_status == "partial"
    assert evaluation.n_pairs == 1 and evaluation.n_query_pairs == 15

    # Changing scoring targets changes errors, never this already fixed prediction.
    shifted = PairwiseObservations.from_pairs(
        observations.observed_pairs,
        observations.observed_values + 1,
        target=observations.target,
        sampling_unit_ids=labels,
    )
    shifted_score = score_out_of_fold(oof, {region.name: {"fox": shifted}})
    assert shifted_score.predictions is oof
    print("Same fixed prediction, shifted scoring-target RMSE:", shifted_score.rmse)

    member = ensemble.members[0]
    assert member.model is not None
    training_pair = member.fold.training[region.name]["fox"].pairs[0]
    known = member.model.predict_known_effects(region, pairs={"fox": [training_pair]})["fox"]
    print("Known-effect deployment on a calibration pair:", known.prediction.values)
    excluded = predict_out_of_fold(ensemble, [region], {region.name: {"fox": [training_pair]}})
    excluded_score = score_out_of_fold(excluded, grouped)
    print("This pair is excluded from OOF because its own target was accessed:")
    print(excluded.exclusion_reasons)
    print("Zero-coverage points/metrics:", excluded.values, excluded_score.rmse)
    assert excluded.coverage_status == "none" and excluded_score.rmse is None

    # Supply one new genetic measurement explicitly; the query's own target stays separate.
    held_out = member.fold.held_out_units[region.name]["fox"]
    training_endpoint = next(label for label in labels if label not in held_out)
    left_endpoint, right_endpoint = sorted((held_out[0], training_endpoint))
    support_pair = (left_endpoint, right_endpoint)
    measured = dict(zip(observations.observed_pairs, observations.observed_values, strict=True))
    support_observations = PairwiseObservations.from_pairs(
        [support_pair],
        [measured[support_pair]],
        target=observations.target,
        sampling_unit_ids=labels,
    )
    support = EvaluationSupport(
        support_observations,
        ObservationPartition(region.name, "fox", (support_pair,), role="support"),
    )
    strict = predict_out_of_fold(
        ensemble,
        [region],
        {region.name: {"fox": member.fold.query[region.name]["fox"].pairs}},
        support={region.name: {"fox": support}},
        regime=EvaluationRegime(prediction_mode="support"),
    )
    print("Strict support counts its endpoints as seen; coverage:", strict.coverage)
    conditional = predict_out_of_fold(
        ensemble,
        [region],
        {region.name: {"fox": member.fold.query[region.name]["fox"].pairs}},
        support={region.name: {"fox": support}},
        regime=EvaluationRegime(
            prediction_mode="support",
            support_endpoint_policy="allow_declared_support",
        ),
    )
    # Query targets are read only at this separate scoring step.
    conditional_score = score_out_of_fold(conditional, grouped)
    print("Declared-support conditional OOF identities:", conditional.keys)
    print("Declared-support conditional OOF mean:", conditional.values)
    print("Eligibility was checked before disjoint support; endpoints are now conditioned.")
    assert strict.coverage == 0 and conditional.coverage == 1
    print("Conditional OOF coverage, RMSE:", conditional.coverage, conditional_score.rmse)
    print("Per-member predictive variance stays on the log1p model scale:")
    print(conditional.member_model_variances)

    # Save after a real update between evaluations under the original budget.
    # A shorter completed budget would have an extra final evaluation.
    folds = tuple({member.fold.fold_id: member.fold for member in ensemble.members}.values())
    with TemporaryDirectory() as directory:
        continued = continue_members(
            directory,
            region,
            grouped,
            folds,
            TrainingConfig(objective="mlpe", steps=2, evaluation_interval=2),
        )
        if continued.failures:
            raise RuntimeError(continued.failures)
        print("Exactly continued members:", continued.expected_member_ids)
