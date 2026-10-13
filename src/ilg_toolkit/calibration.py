"""Target-scale relationships and frozen regional genetic calibration."""

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPEConfig, RegionalMLPECalibration, calibrate_mlpe

if TYPE_CHECKING:
    from .model import LandscapeGeneticsModel


class AbstractCalibrationModule(eqx.Module):
    """Own the target contract and map selected ecological distances to genetic values."""

    target: eqx.AbstractVar[TargetSpec]

    @abstractmethod
    def predict(self, distances, pairs, *, region_name: str, species_name: str) -> jax.Array:
        """Return original-scale values for selected distances and labelled pairs."""


@final
class DirectCalibrationModule(AbstractCalibrationModule):
    """Apply the declared inverse target transform without fitting another stage."""

    target: TargetSpec

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec):
            raise ValueError("Direct calibration requires a TargetSpec")

    def predict(self, distances, pairs, *, region_name: str, species_name: str) -> jax.Array:
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
            raise ValueError(
                f"Region {region_name!r} has no MLPE calibration; calibrate it explicitly"
            )
        return value

    def predict(self, distances, pairs, *, region_name: str) -> jax.Array:
        return self.for_region(region_name).predict_marginal(distances, pairs).prediction.values


def calibrate(
    model: "LandscapeGeneticsModel",
    region: RegionBatch,
    observations: PairwiseObservations,
    *,
    partitions: ObservationPartition | Sequence[ObservationPartition] | None = None,
    config: MLPEConfig | None = None,
) -> "LandscapeGeneticsModel":
    """Fit or replace a regional MLPE relationship with explicitly declared data use.

    Without partitions only recorded distance-model training observations are selected.
    Additional development observations require explicit training, validation,
    or calibration partitions. New regions therefore require a declared
    partition. Query and support roles cannot calibrate a model.

    Returns a model with an MLPE calibration module and the frozen distance model,
    retaining other MLPE regions and training/selection provenance. Converting a
    direct module replaces it completely: only explicitly calibrated regions can
    then predict genetic values. Prior training objectives and histories survive.
    """
    model._validate_region(region)
    if observations.target != model.target:
        raise ValueError("Calibration target meaning, transform, and units must match the model")
    observations.aligned_pairs(region)
    if partitions is None:
        pairs = model.training_pairs.get(region.name)
        if not pairs:
            raise ValueError(
                "No recorded training observations for this region; declare a calibration partition"
            )
        partitions = (ObservationPartition(region.name, pairs, role="training"),)
    elif isinstance(partitions, ObservationPartition):
        partitions = (partitions,)
    elif isinstance(partitions, Sequence) and not isinstance(partitions, (str, bytes)):
        partitions = tuple(partitions)
    else:
        raise ValueError(
            "Calibration partitions must explicitly declare observed pairs and data roles"
        )
    if not partitions:
        raise ValueError("Calibration requires a nonempty declared partition selection")
    if region.name not in model.training_pairs and model.feature_names is None:
        raise ValueError("Transferring calibration to a new region requires declared feature_names")
    roles = {}
    sampling_kinds = region.sampling_unit_kinds
    assert sampling_kinds is not None  # RegionBatch normalizes optional constructor input.
    kinds = dict(zip(region.sampling_unit_ids, sampling_kinds, strict=True))
    for partition in partitions:
        if not isinstance(partition, ObservationPartition):
            raise ValueError("Calibration selections must be ObservationPartition inputs")
        if partition.role not in {"training", "validation", "calibration"}:
            raise ValueError("Calibration cannot consume query or support observations")
        observations.aligned_pairs(region, partition)
        for pair in partition.pairs:
            if pair in roles:
                raise ValueError(
                    "Calibration partitions overlap; declare each pair's data role once"
                )
            if any(kinds[label] != "population" for label in pair):
                raise ValueError("Population MLPE calibration requires population sampling units")
            roles[pair] = partition.role
    pairs = tuple(pair for pair in observations.observed_pairs if tuple(sorted(pair)) in roles)
    lookup = {label: index for index, label in enumerate(observations.sampling_unit_ids)}
    left = jnp.asarray([lookup[a] for a, _ in pairs])
    right = jnp.asarray([lookup[b] for _, b in pairs])
    selected_observations = PairwiseObservations.from_pairs(
        pairs,
        observations.values[left, right],
        target=observations.target,
        sampling_unit_ids=observations.sampling_unit_ids,
    )
    observed_distances = model._distances_for_pairs(region, selected_observations.observed_pairs)
    selected = ObservationPartition(region.name, tuple(roles), role="calibration")
    regions = (
        dict(model.calibration.regions)
        if isinstance(model.calibration, MLPECalibrationModule)
        else {}
    )
    if config is None and region.name in regions:
        config = regions[region.name].config
    head = calibrate_mlpe(
        observed_distances,
        selected_observations,
        region_name=region.name,
        partition=selected,
        config=config,
    )
    head = eqx.tree_at(
        lambda value: value.calibration_roles,
        head,
        tuple(roles[tuple(sorted(pair))] for pair in head.calibration_pairs),
    )
    regions[region.name] = head
    return eqx.tree_at(
        lambda value: value.calibration, model, MLPECalibrationModule(model.target, regions)
    )
