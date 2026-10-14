"""Compose a native ecological distance model with genetic calibration."""

from collections.abc import Mapping
from typing import final

import equinox as eqx
import jax.numpy as jnp

from .calibration import AbstractCalibrationModule, UnpooledMLPECalibrationModule
from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPEError
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
        if isinstance(self.calibration, UnpooledMLPECalibrationModule) and any(
            set(groups) - set(self.species) for groups in self.calibration.groups.values()
        ):
            raise ValueError("MLPE calibration contains unknown species outside the model registry")
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
        if isinstance(self.calibration, UnpooledMLPECalibrationModule):
            for name in self.distance_model._select_species(region, species):
                self._mlpe_head(region, name)
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

    def _mlpe_head(self, region: RegionBatch, species: str):
        self._validate_region(region)
        if not isinstance(self.calibration, UnpooledMLPECalibrationModule):
            raise ValueError("This operation requires an MLPE model")
        head = self.calibration.for_group(region.name, species)
        kinds = region.sampling_units[species].kinds
        assert kinds is not None
        if any(kind != "population" for kind in kinds):
            raise ValueError("Population MLPE prediction requires population sampling units")
        return head

    def predict_known_effects(self, region: RegionBatch, *, species=None, pairs=None):
        """Use each group's saved population effects, returning species-keyed results.

        Unseen populations receive independent prior effects. The returned model-
        scale variances include effect-posterior, residual and configured jitter
        uncertainty, with calibration provenance retained. Query targets are never
        accepted; distance parameters and saved calibrations remain unchanged.
        Means are inverse-transformed to target units; variances stay on the model
        scale and exclude uncertainty in distance parameters, fixed coefficients
        and variance parameters.
        """
        selected = self.distance_model._select_species(region, species)
        heads = {name: self._mlpe_head(region, name) for name in selected}
        raw = self.predict_raw_distances(region, species=selected, pairs=pairs)
        return {
            name: eqx.tree_at(
                lambda value: value.prediction.sampling_unit_ids,
                heads[name].predict_known_effects(result.values, result.pairs),
                result.sampling_unit_ids,
            )
            for name, result in raw.items()
        }

    def predict_with_support(
        self,
        region: RegionBatch,
        support_observations,
        *,
        support_partition,
        species=None,
        pairs=None,
    ):
        """Condition each selected group's effects on explicitly declared support.

        Support observations, support partitions and query pairs are species-keyed
        mappings. Support must have the calibrated target and role ``support``;
        it cannot reuse calibration or query pairs, including reversed pairs.
        No query targets are accepted. The entire distance model and saved group
        calibrations remain frozen; returned uncertainty and provenance follow
        ``predict_known_effects`` with the additional support access recorded.
        """
        selected = self.distance_model._select_species(region, species)
        heads = {name: self._mlpe_head(region, name) for name in selected}
        for mapping, label in (
            (support_observations, "observations"),
            (support_partition, "partitions"),
        ):
            if not isinstance(mapping, Mapping) or set(mapping) != set(selected):
                raise ValueError(f"Support {label} must match exactly the selected species")
        if pairs is not None and (not isinstance(pairs, Mapping) or set(pairs) != set(selected)):
            raise ValueError("pairs must be a mapping keyed by exactly the selected species")
        combined, queries = {}, {}
        for name in selected:
            observations = support_observations[name]
            partition = support_partition[name]
            if not isinstance(observations, PairwiseObservations):
                raise ValueError("Support groups require PairwiseObservations")
            if not isinstance(partition, ObservationPartition) or partition.role != "support":
                raise MLPEError("Conditioning requires an explicit support-role partition")
            observations.aligned_pairs(region, name, partition)
            ids = region.sampling_units[name].ids
            supplied = (
                tuple((left, right) for i, left in enumerate(ids) for right in ids[i + 1 :])
                if pairs is None
                else tuple(pairs[name])
            )
            query = ObservationPartition(region.name, name, supplied, role="query")
            if set(query.pairs) & set(partition.pairs):
                raise MLPEError(
                    "Support/query overlap: an unordered support pair cannot be a query"
                )
            queries[name] = tuple(tuple(pair) for pair in supplied)
            combined[name] = observations.observed_pairs + queries[name]
        raw = self.predict_raw_distances(region, species=selected, pairs=combined)
        results = {}
        for name, result in raw.items():
            observations = support_observations[name]
            count = len(observations.observed_pairs)
            conditioned = heads[name].condition_on_support(
                result.values[:count], observations, partition=support_partition[name]
            )
            prediction = conditioned.predict(result.values[count:], queries[name])
            results[name] = eqx.tree_at(
                lambda value: value.prediction.sampling_unit_ids,
                prediction,
                result.sampling_unit_ids,
            )
        return results

    def calibrate(self, region: RegionBatch, observations, *, partitions=None, config=None):
        """Recalibrate supplied species in this region with the distance model frozen."""
        from .calibration import calibrate

        return calibrate(self, region, observations, partitions=partitions, config=config)
