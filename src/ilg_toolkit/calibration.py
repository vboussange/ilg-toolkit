"""Target-scale relationships and frozen regional genetic calibration."""

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax

from .data import TargetSpec
from .mlpe import RegionalMLPECalibration


class AbstractCalibrationModule(eqx.Module):
    """Own the target contract and map selected ecological distances to genetic values."""

    target: eqx.AbstractVar[TargetSpec]

    @abstractmethod
    def predict(self, distances, pairs, *, region_name: str) -> jax.Array:
        """Return original-scale values for selected distances and labelled pairs."""


@final
class DirectCalibrationModule(AbstractCalibrationModule):
    """Apply the declared inverse target transform without fitting another stage."""

    target: TargetSpec

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec):
            raise ValueError("Direct calibration requires a TargetSpec")

    def predict(self, distances, pairs, *, region_name: str) -> jax.Array:
        return self.target.inverse(distances)


@final
class MLPECalibrationModule(AbstractCalibrationModule):
    """Select a fitted regional MLPE relationship; uncalibrated regions are refused.

    ``regions`` maps region identities to standalone regional calibrations. Every
    installed relationship must agree with this module's authoritative target.
    """

    target: TargetSpec
    regions: dict[str, RegionalMLPECalibration] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec) or self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        regions = dict(self.regions)
        if any(not isinstance(value, RegionalMLPECalibration) for value in regions.values()):
            raise ValueError("MLPE regions require fitted RegionalMLPECalibration values")
        if any(
            name != value.region_name or value.target != self.target
            for name, value in regions.items()
        ):
            raise ValueError(
                "Regional calibration names and target contracts must match the module"
            )
        self.regions = regions

    def for_region(self, region_name: str) -> RegionalMLPECalibration:
        """Return a supported region's calibration, requiring explicit regional fitting."""
        value = self.regions.get(region_name)
        if value is None:
            raise ValueError(f"Region {region_name!r} has no MLPE calibration; fit it with steps=0")
        return value

    def predict(self, distances, pairs, *, region_name: str) -> jax.Array:
        return self.for_region(region_name).predict(distances, pairs).values
