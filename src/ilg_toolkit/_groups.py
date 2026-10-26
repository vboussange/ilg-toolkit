"""Aligned region–species observations shared by training and calibration."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .data import PairwiseObservations, RegionBatch
from .losses import GroupKey
from .models.base import region_for_species

RegionCollection = RegionBatch | Sequence[RegionBatch] | Mapping[str, RegionBatch]
RegionObservations = PairwiseObservations | Sequence[PairwiseObservations]
ObservationCollection = (
    RegionObservations | Sequence[RegionObservations] | Mapping[str, RegionObservations]
)


class _RegionalInput(NamedTuple):
    region: RegionBatch
    observations: PairwiseObservations


class _GroupInput(NamedTuple):
    region: RegionBatch
    observations: PairwiseObservations
    species_index: int
    identity: GroupKey
    weight: float


def _align_collection(values: object, names: Sequence[str], description: str) -> tuple[object, ...]:
    """A labeled collection aligned with its region batches."""
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError(f"{description} mapping keys must match region names exactly")
        return tuple(values[name] for name in names)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if len(values) != len(names):
            raise ValueError(f"{description} sequence must align with all supplied regions")
        return tuple(values)
    if len(names) == 1:
        return (values,)
    raise ValueError(f"{description} must align with the supplied region collection")


def _normalize_inputs(
    regions: RegionCollection,
    observations: ObservationCollection,
) -> tuple[_RegionalInput, ...]:
    """Aligned collections with region identities sorted for reproducible random sampling."""
    if isinstance(regions, RegionBatch):
        ordered = [regions]
    elif isinstance(regions, Mapping):
        ordered = list(regions.values())
        if any(
            not isinstance(value, RegionBatch) or key != value.name
            for key, value in regions.items()
        ):
            raise ValueError("Region mapping keys must match each region batch's name")
    elif isinstance(regions, Sequence) and not isinstance(regions, (str, bytes)):
        ordered = list(regions)
    else:
        raise ValueError("Provide a region batch or a nonempty region sequence/mapping")
    if not ordered or any(not isinstance(region, RegionBatch) for region in ordered):
        raise ValueError("Regions must be a nonempty collection of RegionBatch inputs")
    names = [region.name for region in ordered]
    if len(set(names)) != len(names):
        raise ValueError("Region names must be unique; duplicate names cannot share a fit")

    observed = (
        (observations,)
        if isinstance(regions, RegionBatch) and not isinstance(observations, Mapping)
        else _align_collection(observations, names, "Observations")
    )
    inputs = []
    for region, observation in zip(ordered, observed, strict=True):
        group_observations = (
            (observation,) if isinstance(observation, PairwiseObservations) else observation
        )
        if not isinstance(group_observations, Sequence) or not group_observations:
            raise ValueError("Each region requires observations for at least one species")
        for observed_group in group_observations:
            if not isinstance(observed_group, PairwiseObservations):
                raise ValueError("Observations contain unsupported input types")
            inputs.append(_RegionalInput(region, observed_group))
    return tuple(sorted(inputs, key=lambda item: item.region.name))


def _index_inputs(
    inputs: Sequence[_RegionalInput], order: tuple[str, ...] | None, num_species: int
) -> tuple[_GroupInput, ...]:
    groups = []
    identities = set()
    for region, observations in inputs:
        name = observations.species
        if name is None and order is not None and len(order) == 1:
            name = order[0]
        if order is None:
            if name is not None or region.species is not None:
                raise ValueError("Named observations require a species order")
            index = 0
        else:
            if name not in order or (num_species > 1 and region.species is None):
                raise ValueError("Observations and sampling units must identify a modeled species")
            index = order.index(name)
            if region.species is not None and not set(region.species).issubset(order):
                raise ValueError("Sampling-unit species must belong to the model species order")
        identity = (region.name, index)
        if identity in identities:
            raise ValueError("Provide one observations object per region–species group")
        identities.add(identity)
        selected = region_for_species(region, name)
        groups.append(_GroupInput(selected, observations, index, identity, 0.0))
    coverage = Counter(group.species_index for group in groups)
    return tuple(
        group._replace(weight=1 / (len(coverage) * coverage[group.species_index]))
        for group in sorted(groups, key=lambda group: (group.region.name, group.species_index))
    )
