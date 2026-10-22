"""Scientific behavior of calibration modules."""

from dataclasses import replace
from typing import cast

import jax.numpy as jnp

from ilg_toolkit import MLPECalibrationModule, TargetSpec, TrainingConfig, fit


def test_regional_mlpe_installation_requires_matching_region_and_target_contracts(
    ScalarEmbedding, calibration_problem
):
    import pytest

    from ilg_toolkit import MLPEError, RegionalMLPECalibration

    region, observations, training, _ = calibration_problem()
    model = fit(
        region,
        observations,
        distance_model=ScalarEmbedding(jnp.asarray(1.0)),
        config=TrainingConfig(objective="mlpe", steps=0),
        partition=training,
    ).model
    regional = cast(MLPECalibrationModule, model.calibration).for_region(region.name)
    assert cast(RegionalMLPECalibration, regional).target == model.target == observations.target
    with pytest.raises(ValueError):
        MLPECalibrationModule(
            model.target, {"wrong-region": cast(RegionalMLPECalibration, regional)}
        )
    with pytest.raises(ValueError):
        MLPECalibrationModule(
            TargetSpec("another target"), {region.name: cast(RegionalMLPECalibration, regional)}
        )
    with pytest.raises(MLPEError):
        replace(
            cast(RegionalMLPECalibration, regional),
            target=TargetSpec("relatedness", kind="relatedness"),
        )
