"""Independent Gaussian references for explicitly selected species conditioning."""

from dataclasses import replace
from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from test_mlpe_conditioning import joint_gaussian_reference
from test_mlpe_training import WeightedEmbedding
from test_recalibration import problem

from ilg_toolkit import (
    ObservationPartition,
    PairwiseObservations,
    TrainingConfig,
    UnpooledMLPECalibrationModule,
    fit,
)


@pytest.fixture(scope="module")
def conditioned_species():
    prepared, observed, _, _ = problem("same-labels")
    units = prepared.sampling_units["species"]
    region = replace(
        prepared,
        sampling_units={
            "fox": units,
            "lynx": replace(units, grid_positions=np.asarray(units.grid_positions)[::-1]),
        },
    )
    observations = {
        "fox": observed,
        "lynx": PairwiseObservations.from_pairs(
            observed.observed_pairs,
            1 + 2 * observed.observed_values,
            target=observed.target,
        ),
    }
    calibration_pairs = tuple(combinations(units.ids[:4], 2))
    model = fit(
        [region],
        {region.name: observations},
        distance_model=WeightedEmbedding(jnp.zeros(2), species=("fox", "lynx")),
        config=TrainingConfig(objective="mlpe", steps=0),
        partition={
            region.name: {
                name: ObservationPartition(
                    region.name,
                    name,
                    calibration_pairs,
                )
                for name in observations
            }
        },
    ).model
    query = (("population-4", "population-2"), ("population-5", "population-3"))
    support_pairs = (("population-4", "population-0"), ("population-4", "population-1"))
    support = {
        name: PairwiseObservations.from_pairs(support_pairs, values, target=model.target)
        for name, values in (("fox", [2.0, 2.5]), ("lynx", [7.0, 5.0]))
    }
    partitions = {
        name: ObservationPartition(region.name, name, support_pairs, "support") for name in support
    }
    return model, region, observations, query, support, partitions


def test_model_conditioning_keeps_identical_population_labels_separate_by_species(
    conditioned_species,
):
    model, region, observations, query, support, partitions = conditioned_species
    names = ("fox", "lynx")
    pairs = {name: query for name in names}
    known = model.predict_known_effects(region, species=names, pairs=pairs)
    predictions = model.predict_with_support(
        region,
        support,
        species=names,
        pairs=pairs,
        support_partition=partitions,
    )
    assert isinstance(model.calibration, UnpooledMLPECalibrationModule)
    for name in names:
        units = region.sampling_units[name]
        positions = np.asarray(units.grid_positions)
        points = np.asarray(region.feature_array)[positions[:, 0], positions[:, 1]]
        lookup = {label: i for i, label in enumerate(units.ids)}

        def distances(selected, points=points, lookup=lookup):
            return np.array(
                [np.sum((points[lookup[a]] - points[lookup[b]]) ** 2) for a, b in selected]
            )

        head = model.calibration.for_group(region.name, name)
        observed = observations[name]
        values = {
            pair: value
            for pair, value in zip(observed.observed_pairs, observed.observed_values, strict=True)
        }
        targets = np.array([values[pair] for pair in head.calibration_pairs])
        expected_mean, expected_variance = joint_gaussian_reference(
            head,
            head.calibration_pairs,
            distances(head.calibration_pairs),
            targets,
            query,
            distances(query),
        )
        np.testing.assert_allclose(known[name].model_values, expected_mean, atol=1e-10)
        np.testing.assert_allclose(known[name].model_variance, expected_variance, atol=1e-10)
        expected_mean, expected_variance = joint_gaussian_reference(
            head,
            head.calibration_pairs + support[name].observed_pairs,
            np.r_[distances(head.calibration_pairs), distances(support[name].observed_pairs)],
            np.r_[targets, support[name].observed_values],
            query,
            distances(query),
        )
        result = predictions[name]
        np.testing.assert_allclose(result.model_values, expected_mean, atol=1e-10)
        np.testing.assert_allclose(result.model_variance, expected_variance, atol=1e-10)
        assert result.prediction.pairs == query
        assert result.prediction.sampling_unit_ids == units.ids
        assert result.prediction.species_name == result.provenance.species_name == name
        assert result.provenance.region_name == region.name
        assert result.provenance.calibration_pairs == head.calibration_pairs
        assert result.provenance.support_pairs == partitions[name].pairs
        assert result.provenance.support_roles == ("support", "support")
    altered = dict(support)
    altered["fox"] = PairwiseObservations.from_pairs(
        support["fox"].observed_pairs,
        [90.0, 100.0],
        target=model.target,
    )
    changed = model.predict_with_support(
        region,
        altered,
        species=names,
        pairs=pairs,
        support_partition=partitions,
    )
    assert not np.allclose(changed["fox"].model_values, predictions["fox"].model_values)
    np.testing.assert_array_equal(changed["lynx"].model_values, predictions["lynx"].model_values)
    np.testing.assert_array_equal(
        changed["lynx"].model_variance, predictions["lynx"].model_variance
    )
    repeated_known = model.predict_known_effects(region, species=names, pairs=pairs)
    assert eqx.tree_equal(repeated_known, known)


@pytest.mark.parametrize("role", ["training", "validation", "calibration", "query"])
def test_model_support_requires_explicit_support_role(conditioned_species, role):
    model, region, _, query, support, partitions = conditioned_species
    with pytest.raises(ValueError, match="support-role"):
        model.predict_with_support(
            region,
            {"fox": support["fox"]},
            species="fox",
            pairs={"fox": query},
            support_partition={"fox": replace(partitions["fox"], role=role)},
        )


def test_model_support_validates_selected_groups_and_rejects_target_leakage(conditioned_species):
    from operator import methodcaller

    model, region, _, query, support, partitions = conditioned_species
    with pytest.raises(ValueError, match="explicit species"):
        model.predict_known_effects(region, pairs={"fox": query})
    with pytest.raises(ValueError, match="exactly the selected species"):
        model.predict_with_support(
            region,
            support,
            species="fox",
            pairs={"fox": query},
            support_partition=partitions,
        )
    with pytest.raises(ValueError, match="species_name"):
        model.predict_with_support(
            region,
            {"fox": support["fox"]},
            species="fox",
            pairs={"fox": query},
            support_partition={"fox": partitions["lynx"]},
        )
    with pytest.raises(ValueError, match="Support/query overlap"):
        model.predict_with_support(
            region,
            {"fox": support["fox"]},
            species="fox",
            pairs={"fox": tuple((b, a) for a, b in support["fox"].observed_pairs)},
            support_partition={"fox": partitions["fox"]},
        )
    with pytest.raises(TypeError):
        methodcaller(
            "predict_known_effects",
            region,
            species="fox",
            pairs={"fox": query},
            query_targets=[1, 2],
        )(model)
