"""Scientific behavior of model."""

from dataclasses import replace
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from ilg_toolkit import (
    DirectCalibrationModule,
    LandscapeGeneticsModel,
    MLPECalibrationModule,
    ObservationPartition,
    PairwiseObservations,
    TargetSpec,
    TrainingConfig,
    fit,
)
from ilg_toolkit.models import LogLinearConductance


def test_genetic_queries_use_saved_species_order_and_original_scale(observation_problem):
    region, _, _ = observation_problem()
    distance_model = LogLinearConductance(2, num_species=2, patch_size=1)
    distance_model = eqx.tree_at(
        lambda model: model.intercepts, distance_model, jnp.log(jnp.array([1.99, 0.99]))
    )
    model = LandscapeGeneticsModel(
        distance_model,
        DirectCalibrationModule(TargetSpec("divergence", transform="log1p")),
        2,
        feature_names=region.feature_names,
        species=("stoat", "hare"),
    )
    region = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=jnp.concatenate((region.grid_positions, region.grid_positions)),
        sampling_unit_kinds=region.sampling_unit_kinds * 2,
        species=("hare",) * 3 + ("stoat",) * 3,
    )
    results = model.predict(region, species=("hare", "stoat"))
    np.testing.assert_allclose(results["hare"].values, np.expm1([1, 2, 1]), rtol=1e-6)
    np.testing.assert_allclose(results["stoat"].values, np.expm1([0.5, 1, 0.5]), rtol=1e-6)
    np.testing.assert_array_equal(
        model.predict(region, species="hare").values, results["hare"].values
    )
    partial = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids[:3],
        grid_positions=region.grid_positions[:3],
        sampling_unit_kinds=region.sampling_unit_kinds[:3],
        species=("hare",) * 3,
    )
    np.testing.assert_array_equal(
        model.predict(partial, species="hare").values, results["hare"].values
    )
    with pytest.raises(ValueError):
        model.predict(region)


def test_species_calibration_and_support_preserve_separate_population_effects(
    ScalarEmbedding, calibration_problem
):
    region, observations, training, _ = calibration_problem()
    training = ObservationPartition(
        region.name,
        tuple(
            pair
            for pair in observations.observed_pairs
            if set(pair).issubset(region.sampling_unit_ids[:4])
        ),
    )
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    names = ("stoat", "hare")
    references = tuple(
        fit(
            region,
            replace(observations, values=observations.values * factor),
            distance_model=ScalarEmbedding(jnp.asarray(factor)),
            config=TrainingConfig(objective="mlpe", steps=0),
            partition=training,
        ).model
        for factor in (1.0, 2.0)
    )
    model = LandscapeGeneticsModel(
        ScalarEmbedding(jnp.array([1.0, 2.0]), num_species=2),
        MLPECalibrationModule(
            observations.target,
            {
                (region.name, index): cast(MLPECalibrationModule, reference.calibration).for_region(
                    region.name
                )
                for index, reference in enumerate(references)
            },
        ),
        region.feature_array.shape[-1],
        feature_names=region.feature_names,
        species=names,
    )
    kinds = region.sampling_unit_kinds
    assert kinds is not None
    combined = replace(
        region,
        sampling_unit_ids=region.sampling_unit_ids * 2,
        grid_positions=jnp.concatenate((region.grid_positions, region.grid_positions)),
        sampling_unit_kinds=kinds * 2,
        species=tuple(name for name in names for _ in region.sampling_unit_ids),
    )
    support = tuple(
        PairwiseObservations.from_pairs(
            [("population-4", "population-0")],
            [2.0 * (index + 1)],
            target=observations.target,
            species=name,
        )
        for index, name in enumerate(names)
    )
    for mode in ("marginal", "known_effects", "support"):
        results = model.predict(
            combined,
            species=tuple(reversed(names)),
            pairs=pairs,
            known_effects=mode == "known_effects",
            support=tuple(reversed(support)) if mode == "support" else None,
        )
        for index, name in enumerate(names):
            expected = references[index].predict(
                region,
                pairs=pairs,
                known_effects=mode == "known_effects",
                support=replace(support[index], species=None) if mode == "support" else None,
            )
            np.testing.assert_allclose(results[name].values, expected.values, atol=1e-10)
            for extra, values in expected.extras.items():
                np.testing.assert_allclose(results[name].extras[extra], values, atol=1e-10)
    with pytest.raises(ValueError):
        model.predict(combined, species="hare", pairs=pairs, support=support[0])
    with pytest.raises(ValueError):
        model.predict(combined, species=names, pairs=pairs, support=(support[0], support[0]))


def test_model_conditions_real_ecological_distances_with_calibration_fixed(
    ScalarEmbedding, calibration_problem, joint_gaussian_reference
):
    import itertools

    import jax.numpy as jnp

    from ilg_toolkit import (
        PairwiseObservations,
        TrainingConfig,
        fit,
    )

    region, observations, _, _ = calibration_problem("conditioned-landscape")
    calibration_pairs = tuple(itertools.combinations(region.sampling_unit_ids[:4], 2))
    training = ObservationPartition(region.name, calibration_pairs, role="training")
    direct = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(weight=jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition=training,
    ).model
    model = fit(
        region,
        observations,
        distance_model=direct.distance_model,
        partition=training,
        config=TrainingConfig(objective="mlpe", steps=0),
    ).selected_model
    head = cast(MLPECalibrationModule, model.calibration).regions[region.name]
    pairs = (("population-4", "population-2"), ("population-5", "population-3"))
    known = model.predict(region, pairs=pairs, known_effects=True)
    distances = model.distance_model.predict(region).as_matrix()
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}

    def selected_distances(selected):
        return np.array([distances[lookup[a], lookup[b]] for a, b in selected])

    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs,
        selected_distances(calibration_pairs),
        observations.aligned_values(region)[
            [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
        ],
        pairs,
        selected_distances(pairs),
    )
    np.testing.assert_allclose(known.extras["model_values"], expected_mean, atol=1e-10)
    np.testing.assert_allclose(known.extras["model_variance"], expected_variance, atol=1e-10)
    support = PairwiseObservations.from_pairs(
        [("population-4", "population-0"), ("population-4", "population-1")],
        [2.0, 2.5],
        target=head.target,
    )
    result = model.predict(region, pairs=pairs, support=support)
    expected_mean, expected_variance = joint_gaussian_reference(
        head,
        calibration_pairs + support.observed_pairs,
        np.r_[selected_distances(calibration_pairs), selected_distances(support.observed_pairs)],
        np.r_[
            observations.aligned_values(region)[
                [lookup[a] for a, _ in calibration_pairs], [lookup[b] for _, b in calibration_pairs]
            ],
            support.observed_values,
        ],
        pairs,
        selected_distances(pairs),
    )
    np.testing.assert_allclose(result.extras["model_values"], expected_mean, atol=1e-10)
    np.testing.assert_allclose(result.extras["model_variance"], expected_variance, atol=1e-10)
    np.testing.assert_allclose(result.as_matrix()[[4, 5], [2, 3]], expected_mean, atol=1e-10)
    np.testing.assert_allclose(
        result.as_matrix("effect_variance")[[4, 5], [2, 3]],
        expected_variance - head.residual_variance - head.config.jitter,
        atol=1e-10,
    )
    assert eqx.tree_equal(model.distance_model, direct.distance_model)
    reused = PairwiseObservations.from_pairs(
        [tuple(reversed(calibration_pairs[0]))], [1.0], target=head.target
    )
    with pytest.raises(ValueError):
        model.predict(region, pairs=pairs, support=reused)
    with pytest.raises(ValueError):
        model.predict(
            region,
            pairs=[(support.observed_pairs[0][1], support.observed_pairs[0][0])],
            support=support,
        )
    with pytest.raises(ValueError):
        model.predict(region, support=support)
    with pytest.raises(ValueError):
        model.predict(region, pairs=pairs, known_effects=True, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs=pairs, support=support)
    with pytest.raises(ValueError):
        direct.predict(region, pairs=pairs, known_effects=True)
    with pytest.raises(ValueError):
        model.predict(region, pairs=[("missing-location", "population-0")], known_effects=True)


def test_explicit_transform_returns_original_units_and_separate_ecological_distances(
    ScalarEmbedding, observation_problem
):
    region, _, _ = observation_problem()
    target = TargetSpec("explicitly transformed divergence", units="index", transform="log1p")
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [np.e - 1], target=target
    )
    result = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
    )
    prediction = result.model.predict(region)
    assert result.history[0].training_loss < 1e-12
    np.testing.assert_allclose(prediction.as_matrix()[0, 1], np.e - 1, rtol=1e-6)
    np.testing.assert_allclose(result.model.distance_model.predict(region).as_matrix()[0, 1], 1)


def test_recalibrated_model_retains_target_scale_after_save_and_load(
    ScalarEmbedding, calibration_problem, tmp_path
):
    import pytest

    from ilg_toolkit import (
        load_model,
        save_model,
    )

    region, observations, training, _ = calibration_problem()
    state = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="direct_log1p", steps=0),
        partition=training,
    )
    assert state.model.target is state.model.calibration.target
    converted = fit(
        region,
        observations,
        distance_model=state.model.distance_model,
        partition=training,
        config=TrainingConfig(objective="mlpe", steps=0),
    ).selected_model
    assert converted.training_objective == "mlpe"
    assert eqx.tree_equal(converted.distance_model, state.model.distance_model)
    unseen = replace(region, name="unseen")
    assert np.isfinite(state.model.predict(unseen).values).all()
    with pytest.raises(ValueError):
        converted.predict(unseen)
    path = tmp_path / "converted.ilg"
    save_model(path, converted)
    restored = load_model(path, distance_model=ScalarEmbedding(jnp.asarray(0.0)))
    assert restored.training_objective == "mlpe"
    assert restored.target == observations.target
    np.testing.assert_array_equal(restored.predict(region).values, converted.predict(region).values)
