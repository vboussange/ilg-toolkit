"""Region–species groups shared by training and calibration fitting."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .data import _ObservationCollection, _RegionCollection, PairwiseObservations, RegionBatch


# Region name and species index in the distance model's parameter order.
type _RegionSpeciesKey = tuple[str, int]


class _RegionWithGeneticObservations(NamedTuple):
    """One region batch paired with one species' genetic observations."""

    region: RegionBatch
    observations: PairwiseObservations


class _RegionSpeciesTrainingInput(NamedTuple):
    """One region–species group's region batch, observations, species index and loss weight."""

    region: RegionBatch
    observations: PairwiseObservations
    species_index: int
    key: _RegionSpeciesKey
    loss_weight: float


def _align_observations_to_regions(
    values: object, names: Sequence[str], description: str
) -> tuple[object, ...]:
    """Returns a collection aligned with the region names, one entry per region."""
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


def _normalize_training_inputs(
    regions: _RegionCollection,
    observations: _ObservationCollection,
) -> tuple[_RegionWithGeneticObservations, ...]:
    """Returns region batches paired with their observations, sorted by region name."""
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
        raise ValueError("Region names must be unique")

    observed = (
        (observations,)
        if isinstance(regions, RegionBatch) and not isinstance(observations, Mapping)
        else _align_observations_to_regions(observations, names, "Observations")
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
            inputs.append(_RegionWithGeneticObservations(region, observed_group))
    return tuple(sorted(inputs, key=lambda item: item.region.name))


def _validate_shared_training_contract(inputs: Sequence[_RegionWithGeneticObservations]) -> None:
    """Raises a ``ValueError`` unless all inputs share one target contract and feature names."""
    first = inputs[0]
    for region, observations in inputs:
        if observations.target != first.observations.target:
            raise ValueError("Regional observations must share one target contract and scale")
        if region.feature_names != first.region.feature_names:
            raise ValueError("Regional feature names must match in meaning and order")


def _prepare_region_species_inputs(
    inputs: Sequence[_RegionWithGeneticObservations],
    species: tuple[str, ...] | None,
) -> tuple[_RegionSpeciesTrainingInput, ...]:
    """Returns weighted region–species groups, with equal species and region influence."""
    from .models.base import _region_for_species

    groups = []
    keys = set()
    for region, observations in inputs:
        name = observations.species
        if name is None and species is not None and len(species) == 1:
            name = species[0]
        if species is None:
            if name is not None or region.species is not None:
                raise ValueError(
                    "Named observations or sampling units require a distance model with species"
                )
            index = 0
        else:
            if name not in species or (len(species) > 1 and region.species is None):
                raise ValueError(
                    f"Observations and sampling units must identify one of the distance "
                    f"model's species {species!r}"
                )
            index = species.index(name)
            if region.species is not None and not set(region.species).issubset(species):
                raise ValueError(
                    f"Sampling-unit species must belong to the distance model's species {species!r}"
                )
        key = (region.name, index)
        if key in keys:
            raise ValueError("Provide one observations object per region–species group")
        keys.add(key)
        selected = _region_for_species(region, name)
        groups.append(_RegionSpeciesTrainingInput(selected, observations, index, key, 0.0))
    coverage = Counter(group.species_index for group in groups)
    return tuple(
        group._replace(loss_weight=1 / (len(coverage) * coverage[group.species_index]))
        for group in sorted(groups, key=lambda group: (group.region.name, group.species_index))
    )


class _PreparedTrainingInputs(NamedTuple):
    """Aligned inputs, weighted region–species groups, and every distinct region batch."""

    inputs: tuple[_RegionWithGeneticObservations, ...]
    training: tuple[_RegionSpeciesTrainingInput, ...]
    validation: tuple[_RegionSpeciesTrainingInput, ...]
    regions: tuple[RegionBatch, ...]


def _prepare_training_inputs(
    region: _RegionCollection,
    observations: _ObservationCollection,
    species: tuple[str, ...] | None,
    *,
    validation: tuple[_RegionCollection, _ObservationCollection] | None = None,
) -> _PreparedTrainingInputs:
    """Returns the training and validation region–species groups and their region batches.

    ``species`` are the distance model's species names. Validation inputs must
    share the contract of the training inputs.
    """
    inputs = _normalize_training_inputs(region, observations)
    validation_inputs = () if validation is None else _normalize_training_inputs(*validation)
    _validate_shared_training_contract((*inputs, *validation_inputs))
    return _PreparedTrainingInputs(
        inputs,
        _prepare_region_species_inputs(inputs, species),
        _prepare_region_species_inputs(validation_inputs, species),
        tuple({id(item.region): item.region for item in (*inputs, *validation_inputs)}.values()),
    )
