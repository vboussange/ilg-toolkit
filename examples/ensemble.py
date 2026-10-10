"""Independent synthetic fits: deployment, known effects, support and OOF coverage."""

import coordax as cx
import jax
import numpy as np

from ilg_toolkit import (
    EvaluationRegime,
    EvaluationSupport,
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    evaluate_ensemble,
    fit_ensemble,
    predict_out_of_fold,
    score_out_of_fold,
)
from ilg_toolkit.models import UNetEmbeddingDistance


def model_factory(key):
    # Every invocation initializes a new model from its member-specific key.
    return UNetEmbeddingDistance(
        2, patch_size=1, base_channels=2, embedding_dim=2, dropout=0, key=key
    )


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
        labels,
        positions,
    )
    embeddings = features[tuple(positions.T)] * np.array([0.3, 1.5])
    scores = ((embeddings[:, None] - embeddings[None, :]) ** 2).sum(-1)
    left, right = np.triu_indices(len(labels), 1)
    targets = 1 + 0.1 * scores[left, right] + rng.normal(0, 0.02, len(left))
    observations = PairwiseObservations.from_pairs(
        [(labels[i], labels[j]) for i, j in zip(left, right, strict=True)],
        targets,
        target=TargetSpec("synthetic divergence", units="index", transform="log1p"),
        sampling_unit_ids=labels,
    )
    ensemble = fit_ensemble(
        region,
        observations,
        n_folds=1,
        holdout_size=2,
        fold_seed=47,
        initialization_seeds=(13, 29),
        model_factory=model_factory,
        config=TrainingConfig(objective="mlpe", epochs=1, learning_rate=0.001),
    )
    if ensemble.failures:
        raise RuntimeError(ensemble.failures)
    deployment = ensemble.predict(region)
    print("Independent members:", deployment.member_ids)
    print("Deployment averages every member AFTER inverse transformation; units: index")
    print(deployment.values)
    print("Descriptive member SD (separate from MLPE predictive variance):")
    print(deployment.member_spread)

    # All measured pairs are requested. Only untouched held-out pairs get OOF coverage.
    evaluation = evaluate_ensemble(ensemble, region, observations)
    print("Fit/prediction failures:", ensemble.failures, evaluation.predictions.prediction_failures)
    print("OOF covered unique pairs:", evaluation.n_pairs, "of", evaluation.n_query_pairs)
    print("OOF coverage, RMSE:", evaluation.coverage, evaluation.rmse)

    member = ensemble.members[0]
    assert member.model is not None
    training_pair = member.fold.training[region.name].pairs[0]
    known = member.model.predict_known_effects(region, [training_pair])
    print("Known-effect deployment on a calibration pair:", known.values)
    print("This pair is excluded from OOF evaluation because its target was accessed.")

    # Supply one new genetic measurement explicitly; the query's own target stays separate.
    held_out = member.fold.held_out_units[region.name]
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
        ObservationPartition(region.name, (support_pair,), role="support"),
    )
    conditional = predict_out_of_fold(
        ensemble,
        region,
        member.fold.query[region.name].pairs,
        support=support,
        regime=EvaluationRegime(
            prediction_mode="support",
            support_endpoint_policy="allow_declared_support",
        ),
    )
    # Query targets are read only at this separate scoring step.
    conditional_score = score_out_of_fold(conditional, observations)
    print("Declared-support conditional OOF mean:", conditional.values)
    print("Conditional OOF coverage, RMSE:", conditional.coverage, conditional_score.rmse)
    print("Per-member predictive variance stays on the log1p model scale:")
    print(conditional.member_model_variances)
