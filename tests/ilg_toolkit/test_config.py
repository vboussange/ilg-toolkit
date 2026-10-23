"""Scientific behavior of config."""

import numpy as np
import pytest

from ilg_toolkit import PairwiseObservations, TrainingConfig, fit


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, "2"])
@pytest.mark.parametrize(
    "name", ["region_batch_size", "species_batch_size", "species_microbatch_size"]
)
def test_region_batch_size_rejects_invalid_counts(value, name):
    with pytest.raises(ValueError):
        TrainingConfig(objective="direct_log1p", **{name: value})


def test_fresh_fit_requires_declared_encoder_and_objective_while_resume_reuses_them(
    observation_problem,
):
    region, target, encoder = observation_problem()
    observations = PairwiseObservations.from_pairs([("unit/7", "unit/20")], [0.5], target=target)
    omitted = {}
    with pytest.raises(TypeError):
        TrainingConfig(**omitted)
    config = TrainingConfig(objective="direct_log1p", steps=0)
    with pytest.raises(ValueError):
        fit(region, observations, config=config)
    with pytest.raises(ValueError):
        fit(region, observations, distance_model=encoder)
    initial = fit(region, observations, distance_model=encoder, config=config)

    continued = fit(region, observations, state=initial)
    assert continued.history == initial.history

    assert continued.config == config
    np.testing.assert_array_equal(
        continued.model.predict(region).values, initial.model.predict(region).values
    )
    with pytest.raises(ValueError):
        fit(region, observations, state=initial, distance_model=encoder)


def test_fit_validates_species_order_and_derives_a_single_name(
    observation_problem, ScalarEmbedding
):
    from dataclasses import replace

    import jax.numpy as jnp

    region, target, _ = observation_problem()
    region = replace(region, species=("a",) * 3)
    observations = PairwiseObservations.from_pairs(
        [("unit/7", "unit/20")], [0.5], target=target, species="a"
    )
    config = TrainingConfig(objective="direct_log1p", steps=0)
    for order in ("ab", {"a", "b"}, ("a",), ("a", "a"), ("", "a")):
        with pytest.raises(ValueError):
            fit(
                region,
                observations,
                distance_model=ScalarEmbedding(jnp.array(1.0), num_species=2),
                species=order,  # type: ignore[arg-type]
                config=config,
            )
    result = fit(
        region, observations, distance_model=ScalarEmbedding(jnp.array(1.0)), config=config
    )
    assert result.model.species == ("a",)
    assert result.history[0].training_by_species is None
    np.testing.assert_array_equal(
        result.model.predict(region).values, result.model.predict(region, species="a").values
    )
