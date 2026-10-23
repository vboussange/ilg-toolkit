"""Genetic calibration from ecological distances to original-scale predictions."""

from abc import abstractmethod
from typing import final

import equinox as eqx
import jax

from .data import TargetSpec
from .mlpe import RegionalMLPECalibration

type GroupKey = str | tuple[str, int]


class AbstractCalibrationModule(eqx.Module):
    """Abstract genetic calibration mapping ecological distances to genetic predictions."""

    target: eqx.AbstractVar[TargetSpec]

    @abstractmethod
    def predict(
        self, distances, pairs, *, region_name: str, species_index: int | None = None
    ) -> jax.Array:
        """Genetic predictions for ecological distances aligned with labeled query pairs."""


@final
class DirectCalibrationModule(AbstractCalibrationModule):
    """Direct genetic calibration applying the inverse target transform."""

    target: TargetSpec

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec):
            raise ValueError("Direct calibration requires a TargetSpec")

    def predict(
        self, distances, pairs, *, region_name: str, species_index: int | None = None
    ) -> jax.Array:
        return self.target.inverse(distances)


@final
class MLPECalibrationModule(AbstractCalibrationModule):
    """Unpooled maximum-likelihood population effects (MLPE) genetic calibration.

    ``regions`` maps region names, or region–species indices for multispecies
    models, to fitted calibrations with the module's target contract.
    """

    target: TargetSpec
    regions: dict[GroupKey, RegionalMLPECalibration] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec) or self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        regions = dict(self.regions)
        if any(not isinstance(value, RegionalMLPECalibration) for value in regions.values()):
            raise ValueError("MLPE regions require fitted RegionalMLPECalibration values")
        for key, value in regions.items():
            if isinstance(key, tuple) and (len(key) != 2 or type(key[1]) is not int or key[1] < 0):
                raise ValueError("MLPE calibration keys require a region and species index")
            if (
                key[0] if isinstance(key, tuple) else key
            ) != value.region_name or value.target != self.target:
                raise ValueError(
                    "Regional calibration names and target contracts must match the module"
                )
        self.regions = regions

    def for_region(
        self, region_name: str, species_index: int | None = None
    ) -> RegionalMLPECalibration:
        """Fitted MLPE calibration for one region–species group."""
        key = region_name if species_index is None else (region_name, species_index)
        value = self.regions.get(key)
        if value is None:
            raise ValueError(
                f"Region–species group {key!r} has no MLPE calibration; fit with steps=0"
            )
        return value

    def predict(
        self, distances, pairs, *, region_name: str, species_index: int | None = None
    ) -> jax.Array:
        return self.for_region(region_name, species_index).predict(distances, pairs).values
