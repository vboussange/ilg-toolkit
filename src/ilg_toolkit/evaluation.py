"""Label-free out-of-fold prediction, followed by unique-pair target scoring."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .data import ObservationPartition, PairwiseObservations, RegionBatch, TargetSpec
from .ensemble import Ensemble, MemberFailure, _summarize_members
from .training import _normalize_inputs


@dataclass(frozen=True)
class EvaluationRegime:
    """Endpoint and target-access rules, separate from deployment prediction.

    ``unseen`` counts support endpoints as seen. ``allow_declared_support``
    explicitly evaluates conditional prediction after the endpoint holdout was
    established from encoder training/selection and calibration. Neither policy
    permits using the query pair's own target.
    """

    endpoint_regime: str = "both_unseen"
    prediction_mode: str = "marginal"
    support_endpoint_policy: str = "unseen"

    def __post_init__(self):
        if self.endpoint_regime not in {"both_unseen", "at_least_one_unseen"}:
            raise ValueError("endpoint_regime must be both_unseen or at_least_one_unseen")
        if self.prediction_mode not in {"marginal", "known_effects", "support"}:
            raise ValueError("prediction_mode must be marginal, known_effects or support")
        if self.support_endpoint_policy not in {"unseen", "allow_declared_support"}:
            raise ValueError("support_endpoint_policy must be unseen or allow_declared_support")
        if self.prediction_mode != "support" and self.support_endpoint_policy != "unseen":
            raise ValueError("A permissive support policy requires explicit support prediction")


@dataclass(frozen=True)
class EvaluationSupport:
    """Exactly declared regional support targets, separate from query targets."""

    observations: PairwiseObservations
    partition: ObservationPartition

    def __post_init__(self):
        if (
            not isinstance(self.observations, PairwiseObservations)
            or not isinstance(self.partition, ObservationPartition)
            or self.partition.role != "support"
        ):
            raise ValueError(
                "Evaluation support requires observations and a support-role partition"
            )
        observed = {tuple(sorted(pair)) for pair in self.observations.observed_pairs}
        if observed != set(self.partition.pairs):
            raise ValueError("Support observations must contain exactly the declared support pairs")


@dataclass(frozen=True)
class EvaluationAccess:
    """Snapshot of actual regional target access used by one member."""

    region_name: str
    training_pairs: tuple[tuple[str, str], ...]
    validation_pairs: tuple[tuple[str, str], ...]
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    support_pairs: tuple[tuple[str, str], ...]
    prediction_mode: str
    encoder_access_declared: bool


@dataclass(frozen=True)
class OOFPrediction:
    """One row per canonical region/pair, with explicit eligibility and coverage.

    Member SD is descriptive in original target units. Conditional member
    variances are retained separately on the transformed model scale; they do
    not include encoder, fixed-coefficient or variance-parameter uncertainty.
    """

    keys: tuple[tuple[str, tuple[str, str]], ...]
    values: np.ndarray
    target: TargetSpec
    regime: EvaluationRegime
    member_ids: tuple[str, ...]
    member_values: np.ndarray
    member_spread: np.ndarray
    eligible_counts: np.ndarray
    covered_mask: np.ndarray
    exclusion_reasons: dict[str, tuple[str | None, ...]]
    member_statuses: dict[str, str]
    member_failures: dict[str, MemberFailure]
    prediction_failures: dict[str, dict[str, MemberFailure]]
    access: dict[str, dict[str, EvaluationAccess]]
    conditioning_provenance: dict = field(default_factory=dict)
    member_model_variances: np.ndarray | None = None
    scale: str = "original"
    variance_scale: str = "model"
    excluded_uncertainty: tuple[str, ...] = (
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    )

    @property
    def eligible_member_ids(self):
        """Member identities that successfully contributed to each unique pair."""
        return tuple(
            tuple(
                self.member_ids[row]
                for row in np.flatnonzero(np.isfinite(self.member_values[:, column]))
            )
            for column in range(len(self.keys))
        )

    @property
    def coverage(self):
        return float(np.mean(self.covered_mask))

    @property
    def coverage_status(self):
        return (
            "complete"
            if self.covered_mask.all()
            else "partial"
            if self.covered_mask.any()
            else "none"
        )


@dataclass(frozen=True)
class OOFEvaluation:
    """Metrics after eligible means; each covered region/pair counts once."""

    predictions: OOFPrediction
    observed_values: np.ndarray
    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None


def _canonical_pairs(pairs, universe=None):
    try:
        pairs = tuple(tuple(pair) for pair in pairs)
        if any(
            len(pair) != 2
            or any(not isinstance(label, str) or not label for label in pair)
            or pair[0] == pair[1]
            for pair in pairs
        ):
            raise ValueError("Invalid labelled query pairs")
        canonical = tuple(sorted({tuple(sorted(pair)) for pair in pairs}))
    except (TypeError, ValueError) as error:
        raise ValueError("Pairs require distinct nonempty sampling-unit labels") from error
    if universe is not None and any(not set(pair).issubset(universe) for pair in canonical):
        raise ValueError("Query pair endpoints need locations in the prepared region")
    return canonical


def _query_inputs(region, pairs):
    if isinstance(region, RegionBatch):
        regions = {region.name: region}
    elif isinstance(region, Mapping):
        regions = dict(region)
        if any(
            not isinstance(value, RegionBatch) or name != value.name
            for name, value in regions.items()
        ):
            raise ValueError("Prepared region mapping keys must match region names")
    elif isinstance(region, Sequence) and not isinstance(region, (str, bytes)):
        if any(not isinstance(value, RegionBatch) for value in region):
            raise ValueError("Provide prepared query regions")
        regions = {value.name: value for value in region}
        if len(regions) != len(region):
            raise ValueError("Query region names must be unique")
    else:
        raise ValueError("Provide prepared query regions")
    if not regions:
        raise ValueError("Provide at least one prepared query region")
    if not isinstance(pairs, Mapping):
        if len(regions) != 1:
            raise ValueError("Regional query pairs must be mapped by region name")
        pairs = {next(iter(regions)): pairs}
    if set(pairs) != set(regions):
        raise ValueError("Query pair mapping keys must match the prepared regions")
    queries = {}
    for name in sorted(regions):
        selected = pairs[name]
        if isinstance(selected, ObservationPartition):
            if selected.region_name != name or selected.role != "query":
                raise ValueError("Evaluation partitions require matching query roles and regions")
            selected = selected.pairs
        queries[name] = _canonical_pairs(selected, set(regions[name].sampling_unit_ids))
        if not queries[name]:
            raise ValueError("Each query region requires at least one pair")
    return regions, queries


def _support_inputs(support, regions, queries, target, regime):
    if regime.prediction_mode != "support":
        if support is not None:
            raise ValueError("Support targets require explicit support prediction mode")
        return {}
    if support is None:
        raise ValueError("Support prediction requires declared support observations and partitions")
    if isinstance(support, EvaluationSupport) and len(regions) == 1:
        support = {next(iter(regions)): support}
    if not isinstance(support, Mapping) or set(support) != set(regions):
        raise ValueError("Support mapping must declare every query region exactly")
    support = dict(support)
    for name, declared in support.items():
        if not isinstance(declared, EvaluationSupport) or declared.partition.region_name != name:
            raise ValueError("Support requires matching EvaluationSupport region declarations")
        if declared.observations.target != target:
            raise ValueError("Support target scale must match the ensemble target")
        declared.observations.aligned_values(regions[name])
        if set(declared.partition.pairs) & set(queries[name]):
            raise ValueError("Support/query overlap: query targets cannot be used as support")
    return support


def _access(model, name, support, regime):
    head = model.calibrations.get(name)
    calibration = (
        () if head is None else tuple(tuple(sorted(pair)) for pair in head.calibration_pairs)
    )
    return EvaluationAccess(
        name,
        _canonical_pairs(model.training_pairs.get(name, ())),
        _canonical_pairs(model.validation_pairs.get(name, ())),
        calibration,
        () if head is None else head.calibration_roles,
        () if support is None else support.partition.pairs,
        regime.prediction_mode,
        bool(model.training_pairs),
    )


def _exclusion(pair, held_out, access, regime):
    if not access.encoder_access_declared:
        return "unknown_encoder_access"
    consumed = (
        set(access.training_pairs) | set(access.validation_pairs) | set(access.calibration_pairs)
    )
    if pair in consumed or pair in set(access.support_pairs):
        return "query_target_accessed"
    if not held_out:
        return "region_not_held_out"
    seen = {label for observed in consumed for label in observed}
    if regime.support_endpoint_policy == "unseen":
        seen.update(label for observed in access.support_pairs for label in observed)
    untouched_holdouts = set(held_out) - seen
    threshold = 2 if regime.endpoint_regime == "both_unseen" else 1
    if len(set(pair) & untouched_holdouts) < threshold:
        return "insufficient_held_out_unseen_endpoints"
    return None


def _member_predict(model, region, pairs, regime, support, access):
    provenance = None
    variances = None
    if regime.prediction_mode == "marginal":
        values = model.predict_pairs(region, pairs).values
    else:
        if regime.prediction_mode == "known_effects":
            result = model.predict_known_effects(region, pairs)
        else:
            result = model.predict_with_support(
                region, pairs, support.observations, support_partition=support.partition
            )
        provenance = result.provenance
        if (
            result.scale != "original"
            or result.target != model.target
            or result.region_name != region.name
            or tuple(tuple(sorted(pair)) for pair in result.pairs) != pairs
            or result.variance_scale != "model"
            or provenance.region_name != region.name
            or provenance.mode != regime.prediction_mode
            or set(provenance.calibration_pairs) != set(access.calibration_pairs)
            or set(provenance.support_pairs) != set(access.support_pairs)
            or provenance.calibration_roles != access.calibration_roles
            or provenance.support_roles != ("support",) * len(access.support_pairs)
        ):
            raise ValueError(
                "Conditional prediction target-access provenance differs from its declaration"
            )
        values, variances = result.values, result.model_variance
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(pairs),) or not np.isfinite(values).all():
        raise FloatingPointError(
            "Member prediction must return aligned finite original-scale values"
        )
    if variances is not None and (
        np.shape(variances) != values.shape or not np.isfinite(variances).all()
    ):
        raise FloatingPointError("Conditional member model variance must be aligned and finite")
    return values, variances, provenance


def predict_out_of_fold(
    ensemble: Ensemble,
    region,
    pairs,
    *,
    regime: EvaluationRegime | None = None,
    support: EvaluationSupport | Mapping[str, EvaluationSupport] | None = None,
    target: TargetSpec | None = None,
) -> OOFPrediction:
    """Predict labelled pairs with only eligible members; no query targets accepted.

    Counts use endpoints that are BOTH nominally held out and actually unseen.
    Encoder training, validation selection, calibration and declared support all
    contribute target access. Failed/pending members and prediction errors remain
    visible. Supply target metadata explicitly if every member is unavailable.
    """
    if not isinstance(ensemble, Ensemble):
        raise ValueError("Provide an Ensemble with its complete requested composition")
    regime = EvaluationRegime() if regime is None else regime
    if not isinstance(regime, EvaluationRegime):
        raise ValueError("regime must be an EvaluationRegime")
    completed = [member for member in ensemble.members if member.status == "completed"]
    if completed:
        inferred = completed[0].model.target
        if target is not None and target != inferred:
            raise ValueError("Declared evaluation target scale must match ensemble members")
        target = inferred
    if not isinstance(target, TargetSpec):
        raise ValueError("All members unavailable: declare target metadata for coverage reporting")
    regions, queries = _query_inputs(region, pairs)
    supports = _support_inputs(support, regions, queries, target, regime)
    keys = tuple((name, pair) for name in sorted(queries) for pair in queries[name])
    member_ids = ensemble.expected_member_ids
    by_id = {member.identity.member_id: member for member in ensemble.members}
    values = np.full((len(member_ids), len(keys)), np.nan)
    variances = None if regime.prediction_mode == "marginal" else np.full_like(values, np.nan)
    reasons, failures, prediction_failures, access_records, provenance_records = {}, {}, {}, {}, {}
    statuses = {member_id: by_id[member_id].status for member_id in member_ids}
    for row, member_id in enumerate(member_ids):
        member = by_id[member_id]
        member_reasons = [None] * len(keys)
        reasons[member_id] = member_reasons
        if member.status != "completed":
            member_reasons[:] = [f"member_{member.status}"] * len(keys)
            failures[member_id] = member.failure or MemberFailure(
                "member", "Pending", "Member is pending"
            )
            continue
        access_records[member_id], provenance_records[member_id] = {}, {}
        for name, prepared in regions.items():
            columns = [index for index, (region_name, _) in enumerate(keys) if region_name == name]
            try:
                access = _access(member.model, name, supports.get(name), regime)
                access_records[member_id][name] = access
                eligible = []
                for column in columns:
                    reason = _exclusion(
                        keys[column][1], member.fold.held_out_units.get(name, ()), access, regime
                    )
                    member_reasons[column] = reason
                    if reason is None:
                        eligible.append(column)
                if not eligible:
                    continue
                predicted, model_variance, provenance = _member_predict(
                    member.model,
                    prepared,
                    tuple(keys[column][1] for column in eligible),
                    regime,
                    supports.get(name),
                    access,
                )
                values[row, eligible] = predicted
                if model_variance is not None:
                    variances[row, eligible] = model_variance
                if provenance is not None:
                    provenance_records[member_id][name] = provenance
            except Exception as error:
                prediction_failures.setdefault(member_id, {})[name] = MemberFailure(
                    "prediction", type(error).__name__, str(error)
                )
                for column in columns:
                    if member_reasons[column] is None:
                        member_reasons[column] = "prediction_failed"
    counts = np.isfinite(values).sum(axis=0)
    covered = counts > 0
    mean, spread = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
    for column in np.flatnonzero(covered):
        mean[column], spread[column] = _summarize_members(
            values[np.isfinite(values[:, column]), column]
        )
    return OOFPrediction(
        keys,
        mean,
        target,
        regime,
        member_ids,
        values,
        spread,
        counts,
        covered,
        {member_id: tuple(items) for member_id, items in reasons.items()},
        statuses,
        failures,
        prediction_failures,
        access_records,
        provenance_records,
        variances,
    )


def score_out_of_fold(prediction: OOFPrediction, observations) -> OOFEvaluation:
    """Read query targets only after prediction; pool every unique covered pair once."""
    if not isinstance(prediction, OOFPrediction) or prediction.scale != "original":
        raise ValueError("Provide original-scale OOFPrediction values")
    names = {name for name, _ in prediction.keys}
    if isinstance(observations, PairwiseObservations) and len(names) == 1:
        observations = {next(iter(names)): observations}
    if not isinstance(observations, Mapping) or set(observations) != names:
        raise ValueError("Scoring observations must match prediction region identities")
    if len(set(prediction.keys)) != len(prediction.keys):
        raise ValueError("Each canonical region/pair must be scored once")
    lookup = {}
    for name, observed in observations.items():
        if not isinstance(observed, PairwiseObservations) or observed.target != prediction.target:
            raise ValueError("Scoring observations require the same declared target scale")
        lookup.update(
            {
                (name, tuple(sorted(pair))): value
                for pair, value in zip(
                    observed.observed_pairs, observed.observed_values, strict=True
                )
            }
        )
    try:
        targets = np.array([lookup[key] for key in prediction.keys], dtype=np.float64)
    except KeyError as error:
        raise ValueError("Every requested query must have an observed scoring target") from error
    n = int(np.sum(prediction.covered_mask))
    if n:
        residual = prediction.values[prediction.covered_mask] - targets[prediction.covered_mask]
        mse, mae = float(np.mean(residual**2)), float(np.mean(np.abs(residual)))
        rmse = float(np.sqrt(mse))
        if not np.isfinite([mse, mae, rmse]).all():
            raise FloatingPointError("Pooled original-scale evaluation metrics are nonfinite")
    else:
        mse = mae = rmse = None
    return OOFEvaluation(prediction, targets, n, len(targets), prediction.coverage, mse, rmse, mae)


def evaluate_ensemble(
    ensemble: Ensemble,
    region,
    observations,
    *,
    partitions=None,
    regime: EvaluationRegime | None = None,
    support=None,
) -> OOFEvaluation:
    """Predict from declared query identities, then score original-scale observations."""
    inputs = _normalize_inputs(region, observations, partitions)
    regions, queries, observed_by_region = {}, {}, {}
    target = inputs[0][1].target
    for prepared, observed, partition in inputs:
        if observed.target != target:
            raise ValueError("Regional scoring targets must have identical scale contracts")
        if not set(observed.sampling_unit_ids).issubset(prepared.sampling_unit_ids):
            raise ValueError("Scoring observation labels require prepared query locations")
        measured = _canonical_pairs(observed.observed_pairs)
        if partition is not None:
            if partition.region_name != prepared.name or partition.role != "query":
                raise ValueError("Evaluation partitions require matching query roles and regions")
            if not set(partition.pairs).issubset(measured):
                raise ValueError("Query partition selects unobserved scoring pairs")
            measured = partition.pairs
        regions[prepared.name], queries[prepared.name] = prepared, measured
        observed_by_region[prepared.name] = observed
    prediction = predict_out_of_fold(
        ensemble, regions, queries, regime=regime, support=support, target=target
    )
    return score_out_of_fold(prediction, observed_by_region)
