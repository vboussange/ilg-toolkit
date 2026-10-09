"""Explicit regional genetic recalibration without encoder optimization."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .data import ObservationPartition, PairwiseObservations, PreparedRegion
from .mlpe import MLPEConfig, calibrate_mlpe
from .predictor import Predictor


def recalibrate(
    predictor: Predictor,
    region: PreparedRegion,
    observations: PairwiseObservations,
    *,
    partitions: ObservationPartition | Sequence[ObservationPartition] | None = None,
    config: MLPEConfig | None = None,
) -> Predictor:
    """Fit or replace one regional MLPE head with explicitly declared data use.

    Without partitions only recorded encoder-training observations are selected.
    Additional development observations require explicit training, validation,
    or calibration partitions. New regions therefore require a declared
    partition. Query and support roles cannot calibrate a predictor.

    Returns an MLPE-mode predictor with the same encoder, existing regional
    heads, and training/selection provenance. Prior fit histories and the input
    predictor remain unchanged. This also provides an explicit conversion from
    a direct predictor; ordinary direct prediction needs no MLPE head.
    """
    predictor._validate_region(region)
    if observations.target != predictor.target:
        raise ValueError(
            "Calibration target meaning, transform, and units must match the predictor"
        )
    observations.aligned_pairs(region)
    if partitions is None:
        pairs = predictor.training_pairs.get(region.name)
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
    if region.name not in predictor.training_pairs and predictor.feature_names is None:
        raise ValueError("Transferring calibration to a new region requires declared feature_names")
    roles = {}
    kinds = dict(zip(region.sampling_unit_ids, region.sampling_unit_kinds, strict=True))
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
    scores = predictor.landscape_scores(region)
    lookup = {label: index for index, label in enumerate(region.sampling_unit_ids)}
    observed_scores = np.asarray(
        [scores[lookup[a], lookup[b]] for a, b in observations.observed_pairs], dtype=np.float64
    )
    selected = ObservationPartition(region.name, tuple(roles), role="calibration")
    if config is None and region.name in predictor.calibrations:
        config = predictor.calibrations[region.name].config
    head = calibrate_mlpe(
        observed_scores, observations, region_name=region.name, partition=selected, config=config
    )
    head = replace(
        head, calibration_roles=tuple(roles[tuple(sorted(pair))] for pair in head.calibration_pairs)
    )
    heads = dict(predictor.calibrations)
    heads[region.name] = head
    return replace(predictor, objective="mlpe", calibrations=heads)
