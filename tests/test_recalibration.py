"""Frozen model recalibration and regional transfer at the public seam."""

from dataclasses import replace
from typing import final

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import (
    DirectCalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    SamplingUnits,
    TargetSpec,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    fit,
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
        return features[..., :1] * self.weight


def problem(
    name: str = "original",
) -> tuple[RegionBatch, PairwiseObservations, ObservationPartition, ObservationPartition]:
    coordinates = np.array([0, 0.2, 0.8, 1.1, 1.9, 2.8])
    features = np.stack((coordinates, coordinates * 0.1 + 1), axis=-1).reshape(2, 3, 2)
    ids = tuple(f"population-{index}" for index in range(6))
    region = RegionBatch(
        name,
        cx.field(features, "row", "column", "feature"),
        {"species": SamplingUnits(ids, np.array(list(np.ndindex(2, 3))))},
        feature_names=("east_covariate", "canopy_covariate"),
    )
    distances = (coordinates[:, None] - coordinates[None, :]) ** 2
    effects = np.array([0.2, -0.15, 0.1, 0.3, -0.2, 0.05])
    rng = np.random.default_rng(17)
    observed = 2 + 0.7 * distances + effects[:, None] + effects[None, :]
    noise = rng.normal(scale=0.12, size=(6, 6))
    observed += (noise + noise.T) / 2
    np.fill_diagonal(observed, 0)
    observations = PairwiseObservations.from_matrix(
        ids, observed, target=TargetSpec("synthetic divergence", units="index")
    )
    training = ObservationPartition(
        name, "species", observations.observed_pairs[:10], role="training"
    )
    validation = ObservationPartition(
        name, "species", observations.observed_pairs[10:], role="validation"
    )
    return region, observations, training, validation


def test_explicit_development_recalibration_keeps_encoder_and_prior_history_frozen():
    from ilg_toolkit import calibrate

    region, observations, training, validation = problem()
    result = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
        validation=([region], {region.name: {"species": observations}}),
        validation_partition={region.name: {"species": validation}},
    )
    history = result.history
    from ilg_toolkit import MLPEConfig

    calibrated = calibrate(
        result.model, region, {"species": observations}, config=MLPEConfig(jitter=1e-7)
    )
    assert isinstance(calibrated.calibration, UnpooledMLPECalibrationModule)
    assert calibrated.training_objective == "direct_log1p"
    assert eqx.tree_equal(calibrated.distance_model, result.model.distance_model)
    assert (
        calibrated.calibration.for_group(region.name, "species").calibration_pairs == training.pairs
    )
    assert set(calibrated.calibration.for_group(region.name, "species").calibration_roles) == {
        "training"
    }
    developed = calibrate(
        calibrated,
        region,
        {"species": observations},
        partitions={"species": (training, validation)},
    )
    assert eqx.tree_equal(developed.distance_model, calibrated.distance_model)
    assert isinstance(developed.calibration, UnpooledMLPECalibrationModule)
    assert (
        developed.calibration.for_group(region.name, "species").config
        is calibrated.calibration.for_group(region.name, "species").config
    )
    assert result.history is history
    assert result.model.training_objective == "direct_log1p"
    assert isinstance(result.model.calibration, DirectCalibrationModule)
    assert len(developed.calibration.for_group(region.name, "species").calibration_pairs) == 15
    assert set(developed.calibration.for_group(region.name, "species").calibration_roles) == {
        "training",
        "validation",
    }
    assert not np.allclose(
        calibrated.predict(region)["species"].values, developed.predict(region)["species"].values
    )
    np.testing.assert_array_equal(
        developed.ecological_distances(region)["species"],
        calibrated.ecological_distances(region)["species"],
    )


def test_grouped_recalibration_freezes_model_and_preserves_unrelated_species():
    from test_mlpe_training import WeightedEmbedding
    from test_species_mlpe import irregular_problem

    from ilg_toolkit import calibrate

    regions, observed, _ = irregular_problem()
    state = fit(
        regions,
        observed,
        distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
        config=TrainingConfig(objective="mlpe", steps=0),
    )
    before = state.model
    assert isinstance(before.calibration, UnpooledMLPECalibrationModule)
    changed = {
        region.name: {
            "fox": PairwiseObservations.from_pairs(
                observed[region.name]["fox"].observed_pairs,
                2 * observed[region.name]["fox"].observed_values + 1,
                target=before.target,
            )
        }
        for region in regions
    }
    recalibrated = calibrate(before, {region.name: region for region in regions}, changed)
    assert isinstance(recalibrated.calibration, UnpooledMLPECalibrationModule)
    assert eqx.tree_equal(recalibrated.distance_model, before.distance_model)
    assert recalibrated.training_pairs == before.training_pairs
    assert recalibrated.validation_pairs == before.validation_pairs
    assert eqx.tree_equal(
        recalibrated.calibration.for_group("alpine", "lynx"),
        before.calibration.for_group("alpine", "lynx"),
    )
    for region in regions:
        assert not np.allclose(
            recalibrated.predict(region, species="fox")["fox"].values,
            before.predict(region, species="fox")["fox"].values,
        )
    alpine = next(region for region in regions if region.name == "alpine")
    np.testing.assert_array_equal(
        recalibrated.predict(alpine, species="lynx")["lynx"].values,
        before.predict(alpine, species="lynx")["lynx"].values,
    )


def test_recalibration_rejects_duplicate_species_units_across_supplied_or_retained_regions():
    import pytest

    from ilg_toolkit import LandscapeGeneticsModel, calibrate

    region, observed, training, _ = problem()
    model = fit(
        [region],
        {region.name: {"species": observed}},
        distance_model=ScalarEmbedding(jnp.array(1.0)),
        config=TrainingConfig(objective="mlpe", steps=0),
        partition={region.name: {"species": training}},
    ).model
    duplicate = replace(region, name="duplicate")
    declared = ObservationPartition("duplicate", "species", observed.observed_pairs, "calibration")
    with pytest.raises(ValueError, match="sampling unit.*species.*region"):
        model.calibrate(duplicate, {"species": observed}, partitions={"species": declared})
    fresh = LandscapeGeneticsModel(
        model.distance_model,
        DirectCalibrationModule(model.target),
        2,
        region.feature_names,
    )
    with pytest.raises(ValueError, match="sampling unit.*species.*region"):
        calibrate(
            fresh,
            [region, duplicate],
            {region.name: {"species": observed}, "duplicate": {"species": observed}},
            partitions={region.name: {"species": training}, "duplicate": {"species": declared}},
        )
    with pytest.raises(ValueError, match="region.*keys"):
        calibrate(model, {"wrong": region}, {region.name: {"species": observed}})


def test_composed_calibration_owns_inference_and_preserves_training_objective():
    import pytest

    from ilg_toolkit import LandscapeGeneticsModel, calibrate

    region, observations, training, _ = problem()
    state = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
    )
    assert isinstance(state.model, LandscapeGeneticsModel)
    assert isinstance(state.model.calibration, DirectCalibrationModule)
    assert state.model.target is state.model.calibration.target
    converted = calibrate(state.model, region, {"species": observations})
    assert isinstance(converted.calibration, UnpooledMLPECalibrationModule)
    assert converted.training_objective == state.config.objective == "direct_log1p"
    assert eqx.tree_equal(converted.distance_model, state.model.distance_model)
    unseen = replace(region, name="unseen")
    assert np.isfinite(state.model.predict(unseen)["species"].values).all()
    with pytest.raises(ValueError, match="unseen.*no MLPE calibration"):
        converted.predict(unseen)


def test_recalibrated_model_artifact_preserves_training_objective(tmp_path):
    from ilg_toolkit import calibrate, load_model, save_model

    region, observations, training, _ = problem()
    model = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
    ).model
    converted = calibrate(model, region, {"species": observations})
    path = tmp_path / "converted.ilg"
    save_model(path, converted)
    restored = load_model(path, distance_model=ScalarEmbedding(jnp.asarray(0.0)))
    assert restored.training_objective == "direct_log1p"
    assert restored.target == observations.target
    np.testing.assert_array_equal(
        restored.predict(region)["species"].values, converted.predict(region)["species"].values
    )


def test_mlpe_installation_requires_matching_region_species_and_target_contracts():
    import pytest

    from ilg_toolkit import LandscapeGeneticsModel, MLPECalibration, MLPEError

    region, observations, training, _ = problem()
    model = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="mlpe", steps=0),
        partition={region.name: {"species": training}},
    ).model
    assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
    head = model.calibration.for_group(region.name, "species")
    assert isinstance(head, MLPECalibration)
    assert head.target == model.target == observations.target
    for groups in ({"wrong-region": {"species": head}}, {region.name: {"other": head}}):
        with pytest.raises(ValueError, match="group identities and target"):
            UnpooledMLPECalibrationModule(model.target, groups)
    with pytest.raises(ValueError, match="group identities and target"):
        UnpooledMLPECalibrationModule(
            TargetSpec("another target"), {region.name: {"species": head}}
        )
    with pytest.raises(MLPEError, match="dissimilarity TargetSpec"):
        replace(head, target=TargetSpec("relatedness", kind="relatedness"))
    other_species = UnpooledMLPECalibrationModule(
        model.target, {region.name: {"other": replace(head, species_name="other")}}
    )
    with pytest.raises(ValueError, match="calibration.*unknown species"):
        LandscapeGeneticsModel(model.distance_model, other_species, 2, region.feature_names)


def test_transfer_distances_are_label_free_and_each_group_requires_its_own_head():
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = problem()
    original = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
    ).model
    calibrated = calibrate(original, region, {"species": observations})
    # New geographic groups have independently identified biological units.
    units = replace(region.sampling_units["species"], ids=tuple(f"new-{i}" for i in range(6)))
    new_region = replace(region, name="unseen-catchment", sampling_units={"species": units})
    new_observations = PairwiseObservations.from_matrix(
        units.ids, observations.values * 2, target=observations.target
    )
    assert np.isfinite(calibrated.ecological_distances(new_region)["species"]).all()
    with pytest.raises(ValueError, match="unseen-catchment.*no MLPE calibration"):
        calibrated.predict(new_region)
    with pytest.raises(ValueError, match="declare a calibration partition"):
        calibrate(calibrated, new_region, {"species": new_observations})
    development = ObservationPartition(
        new_region.name, "species", new_observations.observed_pairs, role="calibration"
    )
    transferred = calibrated.calibrate(
        new_region, {"species": new_observations}, partitions={"species": development}
    )
    assert eqx.tree_equal(transferred.distance_model, calibrated.distance_model)
    assert isinstance(calibrated.calibration, UnpooledMLPECalibrationModule)
    assert isinstance(transferred.calibration, UnpooledMLPECalibrationModule)
    assert eqx.tree_equal(
        transferred.calibration.for_group(region.name, "species"),
        calibrated.calibration.for_group(region.name, "species"),
    )
    np.testing.assert_array_equal(
        transferred.predict(region)["species"].values, calibrated.predict(region)["species"].values
    )
    assert np.isfinite(transferred.predict(new_region)["species"].values).all()
    assert not np.allclose(
        transferred.predict(region)["species"].values,
        transferred.predict(new_region)["species"].values,
    )
    assert new_region.name not in calibrated.calibration.groups


def test_default_recalibration_ignores_validation_targets_and_records_distance_model_access():
    from ilg_toolkit import calibrate

    region, observations, training, validation = problem()
    model = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
        validation=([region], {region.name: {"species": observations}}),
        validation_partition={region.name: {"species": validation}},
    ).model
    assert model.training_pairs[region.name]["species"] == training.pairs
    assert model.validation_pairs[region.name]["species"] == validation.pairs
    altered = np.array(observations.values)
    ids = region.sampling_units["species"].ids
    lookup = {label: index for index, label in enumerate(ids)}
    for a, b in validation.pairs:
        altered[lookup[a], lookup[b]] = altered[lookup[b], lookup[a]] = 100
    altered = PairwiseObservations.from_matrix(ids, altered, target=observations.target)
    original = calibrate(model, region, {"species": observations})
    changed = calibrate(model, region, {"species": altered})
    np.testing.assert_array_equal(
        original.predict(region)["species"].values, changed.predict(region)["species"].values
    )


def test_recalibration_rejects_incompatible_features_roles_targets_and_individual_effects():
    import pytest

    from ilg_toolkit import calibrate

    region, observations, training, _ = problem()
    model = fit(
        [region],
        {region.name: {"species": observations}},
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition={region.name: {"species": training}},
    ).model
    assert region.feature_names is not None
    with pytest.raises(ValueError, match="feature contract"):
        calibrate(
            model,
            replace(
                region,
                features=cx.field(region.feature_array, "row", "column", "feature"),
                feature_names=tuple(reversed(region.feature_names)),
            ),
            {"species": observations},
        )
    incompatible = PairwiseObservations.from_matrix(
        region.sampling_units["species"].ids,
        observations.values,
        target=TargetSpec("another target", units="fraction"),
    )
    with pytest.raises(ValueError, match="target meaning"):
        calibrate(model, region, {"species": incompatible})
    for role in ("query", "support"):
        with pytest.raises(ValueError, match="query or support"):
            calibrate(
                model,
                region,
                {"species": observations},
                partitions={
                    "species": ObservationPartition(
                        region.name, "species", training.pairs, role=role
                    )
                },
            )
    with pytest.raises(ValueError, match="overlap"):
        calibrate(
            model, region, {"species": observations}, partitions={"species": (training, training)}
        )
    individuals = replace(
        region,
        sampling_units={
            "species": replace(
                region.sampling_units["species"], kinds=("individual",) + ("population",) * 5
            )
        },
    )
    with pytest.raises(ValueError, match="population sampling units"):
        calibrate(model, individuals, {"species": observations})
    calibrated = calibrate(model, region, {"species": observations})
    assert np.isfinite(calibrated.ecological_distances(individuals)["species"]).all()
    with pytest.raises(ValueError, match="population sampling units"):
        calibrated.predict(individuals)
