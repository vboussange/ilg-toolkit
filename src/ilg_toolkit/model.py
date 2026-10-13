"""Compose a native ecological distance model with genetic calibration."""

from typing import final

import equinox as eqx
import jax.numpy as jnp

from .calibration import AbstractCalibrationModule
from .data import RegionBatch, TargetSpec
from .models import AbstractDistanceModel, ConductanceModel
from .prediction import Prediction


@final
class LandscapeGeneticsModel(eqx.Module):
    """A fixed-registry distance model with target and group access provenance."""

    distance_model: AbstractDistanceModel
    calibration: AbstractCalibrationModule
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    training_objective: str | None = eqx.field(default=None, kw_only=True)
    training_pairs: dict[str, dict[str, tuple[tuple[str, str], ...]]] = eqx.field(
        default_factory=dict
    )
    validation_pairs: dict[str, dict[str, tuple[tuple[str, str], ...]]] = eqx.field(
        default_factory=dict
    )

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement the native AbstractDistanceModel")
        if not isinstance(self.calibration, AbstractCalibrationModule):
            raise TypeError("calibration must implement AbstractCalibrationModule")
        if self.training_objective not in {None, "direct_log1p", "mlpe"}:
            raise ValueError("Unknown training objective provenance")
        for field in ("training_pairs", "validation_pairs"):
            access = {
                region: {
                    species: tuple(tuple(sorted(pair)) for pair in pairs)
                    for species, pairs in groups.items()
                }
                for region, groups in getattr(self, field).items()
            }
            if any(set(groups) - set(self.species) for groups in access.values()):
                raise ValueError("Recorded provenance includes unknown species")
            setattr(self, field, access)

    @property
    def species(self) -> tuple[str, ...]:
        return self.distance_model.species

    @property
    def target(self) -> TargetSpec:
        return self.calibration.target

    def _validate_region(self, region: RegionBatch):
        if region.feature_array.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError("Query feature contract must match training meanings and order")

    def predict_raw_distances(self, region: RegionBatch, *, species=None, pairs=None):
        """Species-keyed ecological pair values, without query genetic targets."""
        self._validate_region(region)
        return self.distance_model.predict_raw_distances(region, species=species, pairs=pairs)

    def ecological_distances(self, region: RegionBatch, *, species=None):
        """Explicit species-keyed complete ecological matrices."""
        return {
            name: raw.as_matrix()
            for name, raw in self.predict_raw_distances(region, species=species).items()
        }

    def predict(self, region: RegionBatch, *, species=None, pairs=None):
        """Return species-keyed original genetic-scale predictions."""
        raw = self.predict_raw_distances(region, species=species, pairs=pairs)
        return {
            name: Prediction(
                jnp.asarray(
                    self.calibration.predict(
                        result.values, result.pairs, region_name=region.name, species_name=name
                    )
                ),
                result.pairs,
                region.name,
                name,
                self.target,
                result.sampling_unit_ids,
            )
            for name, result in raw.items()
        }

    def conductance_surface(self, region: RegionBatch, *, species=None):
        """Return selected conductance surfaces from a conductance-family model."""
        if not isinstance(self.distance_model, ConductanceModel):
            raise TypeError("This distance model does not produce a conductance surface")
        self._validate_region(region)
        return self.distance_model.conductance_surface(region, species=species)

    def calibrate(self, region: RegionBatch, observations, *, partitions=None, config=None):
        from .calibration import calibrate

        return calibrate(self, region, observations, partitions=partitions, config=config)
