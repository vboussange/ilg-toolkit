"""Frozen model recalibration and regional transfer at the public seam."""

from dataclasses import replace
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import EmbeddingDistanceModel


@final
class ScalarEmbedding(EmbeddingDistanceModel):
    weight: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def __init__(self, weight: jax.Array):
        self.weight = weight

    def embedding_grid(self, features, *, inference=True, key=None, patch_batch_size=None):
        return features[..., :1] * self.weight


def problem(
    name: str = "original",
) -> tuple[RegionBatch, PairwiseObservations, ObservationPartition, ObservationPartition]:
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    ids = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        features,
        ids,
        np.array(list(np.ndindex(2, 3))),
        feature_names=("east_covariate", "canopy_covariate"),
    )
    scores = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    rng = np.random.default_rng(17)
    observed = 2 + 0.7 * scores + effects[:, None] + effects[None, :]
    noise = rng.normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(
        ids, observed, target=TargetSpec("synthetic divergence", units="index")
    )
    training = ObservationPartition(name, observations.observed_pairs[:10], role="training")
    validation = ObservationPartition(name, observations.observed_pairs[10:], role="validation")
    return region, observations, training, validation


def test_explicit_development_recalibration_keeps_encoder_and_prior_history_frozen():
    from ilg_toolkit import recalibrate

    region, observations, training, validation = problem()
    result = fit(
        region,
        observations,
        model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    )
    history = result.history
    from ilg_toolkit import MLPEConfig

    calibrated = recalibrate(result.model, region, observations, config=MLPEConfig(jitter=1e-7))
    assert calibrated.objective == "mlpe"
    assert calibrated.encoder is result.model.encoder
    assert calibrated.calibrations[region.name].calibration_pairs == training.pairs
    assert set(calibrated.calibrations[region.name].calibration_roles) == {"training"}
    developed = recalibrate(calibrated, region, observations, partitions=(training, validation))
    assert developed.encoder is calibrated.encoder
    assert developed.calibrations[region.name].config is calibrated.calibrations[region.name].config
    assert result.history is history
    assert result.model.objective == "direct_log1p"
    assert not result.model.calibrations
    assert len(developed.calibrations[region.name].calibration_pairs) == 15
    assert set(developed.calibrations[region.name].calibration_roles) == {"training", "validation"}
    assert not np.allclose(calibrated.predict(region).values, developed.predict(region).values)
    np.testing.assert_array_equal(
        developed.landscape_scores(region), calibrated.landscape_scores(region)
    )


def test_transfer_scores_are_label_free_and_each_region_requires_its_own_head():
    import pytest

    from ilg_toolkit import recalibrate

    region, observations, training, _ = problem()
    original = fit(
        region,
        observations,
        model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
        partition=training,
    ).model
    calibrated = recalibrate(original, region, observations)
    new_region = replace(region, name="unseen-catchment")
    scores = calibrated.landscape_scores(new_region)
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError, match="unseen-catchment.*no MLPE calibration"):
        calibrated.predict(new_region)
    with pytest.raises(ValueError, match="declare a calibration partition"):
        recalibrate(calibrated, new_region, observations)
    new_observations = PairwiseObservations.from_matrix(
        observations.sampling_unit_ids, observations.values * 2, target=observations.target
    )
    development = ObservationPartition(
        new_region.name, observations.observed_pairs, role="calibration"
    )
    transferred = calibrated.recalibrate(new_region, new_observations, partitions=development)
    assert transferred.encoder is calibrated.encoder
    assert transferred.calibrations[region.name] is calibrated.calibrations[region.name]
    np.testing.assert_array_equal(
        transferred.predict(region).values, calibrated.predict(region).values
    )
    assert np.isfinite(transferred.predict(new_region).values).all()
    assert not np.allclose(
        transferred.predict(region).values, transferred.predict(new_region).values
    )
    assert new_region.name not in calibrated.calibrations


def test_default_recalibration_ignores_validation_targets_and_records_encoder_access():
    from ilg_toolkit import recalibrate

    region, observations, training, validation = problem()
    model = fit(
        region,
        observations,
        model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
        partition=training,
        validation=(region, observations),
        validation_partition=validation,
    ).model
    assert model.training_pairs[region.name] == training.pairs
    assert model.validation_pairs[region.name] == validation.pairs
    altered = np.array(observations.values)
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    for a, b in validation.pairs:
        altered[lookup[a], lookup[b]] = altered[lookup[b], lookup[a]] = 100
    altered = PairwiseObservations.from_matrix(
        region.sampling_unit_ids, altered, target=observations.target
    )
    original = recalibrate(model, region, observations)
    changed = recalibrate(model, region, altered)
    np.testing.assert_array_equal(original.predict(region).values, changed.predict(region).values)


def test_recalibration_rejects_incompatible_features_roles_targets_and_individual_effects():
    import pytest

    from ilg_toolkit import recalibrate

    region, observations, training, _ = problem()
    model = fit(
        region,
        observations,
        model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(epochs=0),
        partition=training,
    ).model
    assert region.feature_names is not None
    with pytest.raises(ValueError, match="feature contract"):
        recalibrate(
            model,
            replace(region, feature_names=tuple(reversed(region.feature_names))),
            observations,
        )
    incompatible = PairwiseObservations.from_matrix(
        region.sampling_unit_ids,
        observations.values,
        target=TargetSpec("another target", units="fraction"),
    )
    with pytest.raises(ValueError, match="target meaning"):
        recalibrate(model, region, incompatible)
    for role in ("query", "support"):
        with pytest.raises(ValueError, match="query or support"):
            recalibrate(
                model,
                region,
                observations,
                partitions=ObservationPartition(region.name, training.pairs, role=role),
            )
    with pytest.raises(ValueError, match="overlap"):
        recalibrate(model, region, observations, partitions=(training, training))
    individuals = replace(region, sampling_unit_kinds=("individual",) + ("population",) * 5)
    with pytest.raises(ValueError, match="population sampling units"):
        recalibrate(model, individuals, observations)
    calibrated = recalibrate(model, region, observations)
    assert np.isfinite(calibrated.landscape_scores(individuals)).all()
    with pytest.raises(ValueError, match="population sampling units"):
        calibrated.predict(individuals)
