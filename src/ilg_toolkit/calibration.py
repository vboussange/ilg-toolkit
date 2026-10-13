"""Target-scale relationships and frozen regional genetic calibration."""

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, final

import equinox as eqx
import jax
import jax.numpy as jnp

from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .mlpe import MLPECalibration, MLPEConfig, calibrate_mlpe

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
class UnpooledMLPECalibrationModule(AbstractCalibrationModule):
    """Select independent fitted relationships by exact region and species identity."""

    target: TargetSpec
    groups: dict[str, dict[str, MLPECalibration]] = eqx.field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.target, TargetSpec) or self.target.kind != "dissimilarity":
            raise ValueError("Population MLPE requires a dissimilarity target")
        groups = {region: dict(species) for region, species in self.groups.items()}
        for region, species in groups.items():
            for name, value in species.items():
                if not isinstance(value, MLPECalibration):
                    raise ValueError("MLPE groups require fitted MLPECalibration values")
                if (region, name) != (
                    value.region_name,
                    value.species_name,
                ) or value.target != self.target:
                    raise ValueError(
                        "Calibration group identities and target must match the module"
                    )
        self.groups = groups

    def for_group(self, region_name: str, species_name: str) -> MLPECalibration:
        value = self.groups.get(region_name, {}).get(species_name)
        if value is None:
            raise ValueError(
                f"Group ({region_name!r}, {species_name!r}) has no MLPE calibration; "
                "calibrate it explicitly"
            )
        return value

    def predict(self, distances, pairs, *, region_name: str, species_name: str) -> jax.Array:
        return (
            self.for_group(region_name, species_name)
            .predict_marginal(distances, pairs)
            .prediction.values
        )


def calibrate(
    model: "LandscapeGeneticsModel",
    region: RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch],
    observations,
    *,
    partitions=None,
    config: MLPEConfig | None = None,
) -> "LandscapeGeneticsModel":
    """Replace supplied species' MLPE calibrations with the distance model frozen.

    For one RegionBatch, observations and partitions are keyed by species. A
    region collection uses region-then-species mappings for both. Without an
    explicit partition, only recorded distance-model training pairs are used.
    Additional training, validation or calibration pairs require declared roles;
    query and support observations cannot become calibration data.

    Other group calibrations and all distance-model training/selection provenance
    are retained. Converting direct calibration installs MLPE only for supplied
    groups, so other groups require explicit calibration before genetic prediction.
    """
    if isinstance(region, RegionBatch):
        _validate_unit_regions(model, (region,))
        return _calibrate_region(model, region, observations, partitions, config)
    if isinstance(region, Mapping):
        if any(
            not isinstance(value, RegionBatch) or key != value.name for key, value in region.items()
        ):
            raise ValueError("Calibration region mapping keys must match RegionBatch names")
        region = tuple(region.values())
    if (
        not isinstance(region, Sequence)
        or isinstance(region, (str, bytes))
        or not region
        or any(not isinstance(item, RegionBatch) for item in region)
    ):
        raise ValueError("Calibration requires a RegionBatch or nonempty region collection")
    names = tuple(item.name for item in region)
    if len(set(names)) != len(names):
        raise ValueError("Calibration region names must be unique")
    if not isinstance(observations, Mapping) or set(observations) != set(names):
        raise ValueError("Grouped calibration observations must match the supplied regions")
    if partitions is not None and (
        not isinstance(partitions, Mapping) or set(partitions) != set(names)
    ):
        raise ValueError("Grouped calibration partitions must match the supplied regions")
    _validate_unit_regions(model, region)
    for prepared in sorted(region, key=lambda item: item.name):
        model = _calibrate_region(
            model,
            prepared,
            observations[prepared.name],
            None if partitions is None else partitions[prepared.name],
            config,
        )
    return model


def _validate_unit_regions(model, regions):
    """Keep biological identities within one geographic group per species."""
    assignments = {}

    def record(region, species, labels):
        for label in labels:
            prior = assignments.setdefault((species, label), region)
            if prior != region:
                raise ValueError(
                    f"Biological sampling unit {label!r} for species {species!r} "
                    f"is assigned to both regions {prior!r} and {region!r}"
                )

    for provenance in (model.training_pairs, model.validation_pairs):
        for region, groups in provenance.items():
            for species, pairs in groups.items():
                record(region, species, (label for pair in pairs for label in pair))
    if isinstance(model.calibration, UnpooledMLPECalibrationModule):
        for region, groups in model.calibration.groups.items():
            for species, head in groups.items():
                record(region, species, head.population_ids)
    for region in regions:
        for species, units in region.sampling_units.items():
            record(region.name, species, units.ids)


def _calibrate_region(model, region, observations, partitions, config):
    model._validate_region(region)
    if not isinstance(observations, Mapping) or not observations:
        raise ValueError("Calibration observations require a nonempty species-keyed mapping")
    species = model.distance_model._select_species(region, tuple(observations))
    if partitions is not None and (
        not isinstance(partitions, Mapping) or set(partitions) != set(species)
    ):
        raise ValueError("Calibration partitions must match exactly the supplied species")
    if region.name not in model.training_pairs and model.feature_names is None:
        raise ValueError("Transferring calibration to a new region requires declared feature_names")
    groups = (
        {name: dict(heads) for name, heads in model.calibration.groups.items()}
        if isinstance(model.calibration, UnpooledMLPECalibrationModule)
        else {}
    )
    heads = groups.setdefault(region.name, {})
    for name in species:
        prior = heads.get(name)
        group_config = prior.config if config is None and prior is not None else config
        heads[name] = _calibrate_group(
            model,
            region,
            name,
            observations[name],
            None if partitions is None else partitions[name],
            group_config,
        )
    return eqx.tree_at(
        lambda value: value.calibration, model, UnpooledMLPECalibrationModule(model.target, groups)
    )


def _calibrate_group(model, region, species, observations, partitions, config):
    if not isinstance(observations, PairwiseObservations):
        raise ValueError("Calibration groups require PairwiseObservations")
    if observations.target != model.target:
        raise ValueError("Calibration target meaning, transform, and units must match the model")
    observations.aligned_pairs(region, species)
    if partitions is None:
        pairs = model.training_pairs.get(region.name, {}).get(species)
        if not pairs:
            raise ValueError(
                f"No recorded training observations for group ({region.name!r}, {species!r}); "
                "declare a calibration partition"
            )
        partitions = (ObservationPartition(region.name, species, pairs, role="training"),)
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
    roles = {}
    units = region.sampling_units[species]
    sampling_kinds = units.kinds
    assert sampling_kinds is not None
    kinds = dict(zip(units.ids, sampling_kinds, strict=True))
    for partition in partitions:
        if not isinstance(partition, ObservationPartition):
            raise ValueError("Calibration selections must be ObservationPartition inputs")
        if partition.role not in {"training", "validation", "calibration"}:
            raise ValueError("Calibration cannot consume query or support observations")
        observations.aligned_pairs(region, species, partition)
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
    observed_distances = model.predict_raw_distances(
        region, species=species, pairs={species: selected_observations.observed_pairs}
    )[species].values
    selected = ObservationPartition(region.name, species, tuple(roles), role="calibration")
    head = calibrate_mlpe(
        observed_distances,
        selected_observations,
        region_name=region.name,
        species_name=species,
        partition=selected,
        config=config,
    )
    return eqx.tree_at(
        lambda value: value.calibration_roles,
        head,
        tuple(roles[tuple(sorted(pair))] for pair in head.calibration_pairs),
    )
