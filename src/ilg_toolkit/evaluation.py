"""Predict held-out pairs without query targets, then score fixed predictions.

Deployment prediction uses every ensemble member and imposes no holdout rule.
Out-of-fold (OOF) prediction instead selects members for each region/species-qualified
unordered pair: its own target must never have been consumed, and the required
query endpoints must be both nominal fold holdouts and untouched by actual
training, validation-based selection and calibration target access. Declared
support targets also count as access under the default strict policy.

``predict_out_of_fold`` accepts locations and pair identities, never query
targets. ``score_out_of_fold`` subsequently reads measurements; ``evaluate_ensemble``
combines these steps without passing those measurements to member predictors.
Marginal prediction is the default. Known-effect prediction uses saved MLPE
population effects; support prediction additionally consumes explicitly declared,
disjoint support targets. Neither conditional mode relaxes own-query exclusion.

Results contain original-target-scale points, coverage and visible member failures.
Each unique covered region/species/pair receives equal metric weight after eligible member
points are averaged. Conditional model-scale variance and descriptive original-unit
member disagreement are separate diagnostics. See ``docs/evaluation.md`` for a
complete synthetic workflow and the operation-specific precision contracts.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .calibration import UnpooledMLPECalibrationModule
from .data import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
)
from .ensemble import Ensemble, MemberFailure, _group_mapping, _summarize_members
from .prediction import Prediction
from .training import _normalize_inputs


@dataclass(frozen=True)
class EvaluationRegime:
    """Choose endpoint eligibility and marginal or conditional prediction.

    A qualifying endpoint belongs to the member's nominal regional holdout and
    has no actual training, validation-selection or calibration target access.
    The query pair's own target is excluded in every mode, including support.
    This configuration governs OOF evaluation, not deployment prediction.

    Attributes:
        endpoint_regime: ``"both_unseen"`` (default) requires two qualifying
            endpoints; ``"at_least_one_unseen"`` requires one. Holdout and
            untouched status must apply to the same endpoint.
        prediction_mode: ``"marginal"`` (default) uses population effects
            marginalised to zero. ``"known_effects"`` uses the saved regional
            MLPE effect posterior. ``"support"`` conditions that posterior on
            an explicit ``EvaluationSupport`` declaration. Conditional modes
            require regional MLPE calibration and retain its access history.
        support_endpoint_policy: ``"unseen"`` (default) counts support endpoints
            as accessed before checking eligibility. ``"allow_declared_support"``
            checks endpoint eligibility before disjoint declared support is
            consumed and explicitly reports conditional prediction. It is valid
            only with support mode; it never permits the query target as support.

    Raises:
        ValueError: A field is unsupported, or permissive support policy is used
            outside support mode.
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
    """Declare every target that conditional prediction may consume as support.

    Support is distinct from query measurements used only for scoring. The
    evaluation call checks regional alignment, target compatibility and disjoint
    query identities. Each member additionally rejects support already consumed
    by its regional calibration; such member failures remain visible in OOF.

    Attributes:
        observations: Original-scale ``PairwiseObservations`` containing exactly
            the measured support pairs, with the same ``TargetSpec`` as queries.
            Its labelled universe may include units without support measurements.
        partition: ``ObservationPartition`` with role ``"support"`` and the
            query region's name. Its unordered pairs must equal the measured
            support identities, including canonical treatment of reversals.

    Raises:
        ValueError: Inputs have incompatible types, the role is not support, or
            the measured and declared support pair sets differ.
    """

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
    """Actual target-use snapshot for one member's region/species group.

    Entirely absent training provenance is unknown. A complete declared manifest
    may contain no consumed targets for a particular queried group. Population
    IDs in an MLPE posterior are not themselves evidence of target access.
    """

    region_name: str
    species_name: str
    training_pairs: tuple[tuple[str, str], ...]
    validation_pairs: tuple[tuple[str, str], ...]
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    support_pairs: tuple[tuple[str, str], ...]
    prediction_mode: str
    encoder_access_declared: bool


@dataclass(frozen=True)
class OOFPrediction:
    """Label-free original-scale predictions and complete member diagnostics.

    ``predictions[region][species]`` stores common Prediction records. All flat
    arrays follow sorted ``(region, species, canonical_pair)`` keys. Member
    values and optional conditional model variances have shape (members, pairs).
    Excluded/failed entries are NaN; eligible_counts counts successful finite
    contributions. Spread is descriptive population SD in original units, not
    conditional predictive variance. Access, prediction failures and conditional
    provenance are nested member -> region -> species. No query targets are stored.
    """

    predictions: dict[str, dict[str, Prediction]]
    regime: EvaluationRegime
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array
    eligible_counts: np.ndarray
    covered_mask: np.ndarray
    exclusion_reasons: dict[str, tuple[str | None, ...]]
    member_statuses: dict[str, str]
    member_failures: dict[str, MemberFailure]
    prediction_failures: dict[str, dict[str, dict[str, MemberFailure]]]
    access: dict[str, dict[str, dict[str, EvaluationAccess]]]
    conditioning_provenance: dict = field(default_factory=dict)
    member_model_variances: jax.Array | None = None
    variance_scale: str = "model"
    excluded_uncertainty: tuple[str, ...] = (
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    )

    @property
    def keys(self) -> tuple[tuple[str, str, tuple[str, str]], ...]:
        return tuple(
            (region, species, pair)
            for region in sorted(self.predictions)
            for species in sorted(self.predictions[region])
            for pair in self.predictions[region][species].pairs
        )

    @property
    def values(self) -> jax.Array:
        values = [
            self.predictions[r][s].values
            for r in sorted(self.predictions)
            for s in sorted(self.predictions[r])
        ]
        with jax.enable_x64(
            jax.config.read("jax_enable_x64") or any(v.dtype == jnp.float64 for v in values)
        ):
            return jnp.concatenate(values)

    @property
    def target(self) -> TargetSpec:
        return next(iter(next(iter(self.predictions.values())).values())).target

    @property
    def scale(self) -> str:
        return next(iter(next(iter(self.predictions.values())).values())).scale

    @property
    def eligible_member_ids(self):
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
class EvaluationMetrics:
    """Counts and equal-covered-pair errors for an explicitly retained query set."""

    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None


@dataclass(frozen=True)
class OOFEvaluation:
    """Fixed OOF predictions scored once per unique covered group/pair.

    Overall errors give every covered pair equal weight, independent of its
    contributor count. Zero coverage retains the queried identities and returns
    None errors. This scoring policy is distinct from the training objective.
    """

    predictions: OOFPrediction
    observed_values: jax.Array
    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None
    by_group: dict[str, dict[str, EvaluationMetrics]]
    by_species: dict[str, EvaluationMetrics]


def _canonical_pairs(pairs, universe=None):
    """Deduplicate evaluation identities; ordinary prediction rejects duplicates."""
    try:
        pairs = tuple(pairs)
        if any(isinstance(pair, (str, bytes)) for pair in pairs):
            raise ValueError("Pairs require two explicit labels")
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
        raise ValueError("Query pair endpoints need locations in the prepared region/species")
    return canonical


def _query_inputs(region, pairs):
    if isinstance(region, Mapping):
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
        raise ValueError("Provide a nonempty collection of prepared query regions")
    if not regions:
        raise ValueError("Provide at least one prepared query region")
    assigned = {}
    for name, prepared in regions.items():
        for species, units in prepared.sampling_units.items():
            for label in units.ids:
                identity = (species, label)
                if identity in assigned and assigned[identity] != name:
                    raise ValueError(
                        "A sampling unit must have one consistent region assignment "
                        "within its species"
                    )
                assigned[identity] = name
    selections = _group_mapping(pairs, "Query pairs")
    if {name for name, _ in selections} != set(regions):
        raise ValueError("Query pair mapping keys must match the prepared regions")
    queries = {}
    for (name, species), selected in sorted(selections.items()):
        if species not in regions[name].sampling_units:
            raise ValueError("Query species require sampling units in the prepared region")
        if isinstance(selected, ObservationPartition):
            if (selected.region_name, selected.species_name, selected.role) != (
                name,
                species,
                "query",
            ):
                raise ValueError(
                    "Evaluation partitions require matching query roles "
                    "and region/species identities"
                )
            selected = selected.pairs
        queries[name, species] = _canonical_pairs(
            selected, set(regions[name].sampling_units[species].ids)
        )
        if not queries[name, species]:
            raise ValueError("Each query group requires at least one pair")
    return regions, queries


def _support_inputs(support, regions, queries, target, regime):
    if regime.prediction_mode != "support":
        if support is not None:
            raise ValueError("Support targets require explicit support prediction mode")
        return {}
    if support is None:
        raise ValueError("Support prediction requires declared support observations and partitions")
    supports = _group_mapping(support, "Support")
    if set(supports) != set(queries):
        raise ValueError("Support mapping must declare every query group exactly")
    for (name, species), declared in supports.items():
        if not isinstance(declared, EvaluationSupport) or (
            declared.partition.region_name,
            declared.partition.species_name,
        ) != (name, species):
            raise ValueError(
                "Support requires matching EvaluationSupport region/species declarations"
            )
        if declared.observations.target != target:
            raise ValueError("Support target scale must match the ensemble target")
        declared.observations.aligned_pairs(regions[name], species, declared.partition)
        if set(declared.partition.pairs) & set(queries[name, species]):
            raise ValueError("Support/query overlap: query targets cannot be used as support")
    return supports


def _access(model, name, species, support, regime):
    head = (
        model.calibration.groups.get(name, {}).get(species)
        if isinstance(model.calibration, UnpooledMLPECalibrationModule)
        else None
    )
    return EvaluationAccess(
        name,
        species,
        _canonical_pairs(model.training_pairs.get(name, {}).get(species, ())),
        _canonical_pairs(model.validation_pairs.get(name, {}).get(species, ())),
        () if head is None else tuple((min(a, b), max(a, b)) for a, b in head.calibration_pairs),
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
        return "group_not_held_out"
    seen = {label for observed in consumed for label in observed}
    if regime.support_endpoint_policy == "unseen":
        seen.update(label for observed in access.support_pairs for label in observed)
    untouched_holdouts = set(held_out) - seen
    threshold = 2 if regime.endpoint_regime == "both_unseen" else 1
    if len(set(pair) & untouched_holdouts) < threshold:
        return "insufficient_held_out_unseen_endpoints"
    return None


def _pair_roles(pairs, roles):
    if len(pairs) != len(roles):
        raise ValueError("Target-access pairs and roles must remain aligned")
    return tuple(
        sorted((tuple(sorted(pair)), role) for pair, role in zip(pairs, roles, strict=True))
    )


def _member_predict(model, region, species, pairs, regime, support, access):
    provenance = variances = None
    selected = {species: pairs}
    if regime.prediction_mode == "marginal":
        prediction = model.predict(region, species=species, pairs=selected)[species]
    else:
        if regime.prediction_mode == "known_effects":
            result = model.predict_known_effects(region, species=species, pairs=selected)[species]
        else:
            result = model.predict_with_support(
                region,
                {species: support.observations},
                species=species,
                pairs=selected,
                support_partition={species: support.partition},
            )[species]
        provenance, variances, prediction = (
            result.provenance,
            result.model_variance,
            result.prediction,
        )
        if (
            result.variance_scale != "model"
            or provenance.region_name != region.name
            or provenance.species_name != species
            or provenance.mode != regime.prediction_mode
            or _pair_roles(provenance.calibration_pairs, provenance.calibration_roles)
            != _pair_roles(access.calibration_pairs, access.calibration_roles)
            or _pair_roles(provenance.support_pairs, provenance.support_roles)
            != _pair_roles(access.support_pairs, ("support",) * len(access.support_pairs))
        ):
            raise ValueError(
                "Conditional prediction target-access provenance differs from its declaration"
            )
    if (
        prediction.scale != "original"
        or prediction.target != model.target
        or prediction.region_name != region.name
        or prediction.species_name != species
        or prediction.pairs != pairs
        or prediction.sampling_unit_ids != region.sampling_units[species].ids
    ):
        raise ValueError(
            "Member prediction target, identity or pair order differs from its declaration"
        )
    values = prediction.values
    with jax.enable_x64(
        jax.config.read("jax_enable_x64")
        or values.dtype == jnp.float64
        or (variances is not None and variances.dtype == jnp.float64)
    ):
        if values.shape != (len(pairs),) or not bool(jnp.isfinite(values).all()):
            raise FloatingPointError(
                "Member prediction must return aligned finite original-scale values"
            )
        if variances is not None and (
            variances.shape != values.shape
            or not bool((jnp.isfinite(variances) & (variances >= 0)).all())
        ):
            raise FloatingPointError(
                "Conditional member model variance must be aligned, nonnegative and finite"
            )
    return values, variances, provenance


def predict_out_of_fold(
    ensemble: Ensemble,
    region,
    pairs,
    *,
    regime: EvaluationRegime | None = None,
    support: Mapping[str, Mapping[str, EvaluationSupport]] | None = None,
    target: TargetSpec | None = None,
) -> OOFPrediction:
    """Predict nested group queries without consuming query measurements.

    Regions are a nonempty collection; pairs and optional support are region then
    species mappings. Reversed/repeated queries deduplicate within their group.
    Each member must satisfy both nominal holdout and actual-access eligibility.
    Only eligible pairs are predicted/inverted. Failures remain explicit and do
    not remove other member/group contributions. Conditional variance stays in
    model units; original-scale member disagreement is a separate statistic.
    """
    if not isinstance(ensemble, Ensemble):
        raise ValueError("Provide an Ensemble with its complete requested composition")
    regime = EvaluationRegime() if regime is None else regime
    if not isinstance(regime, EvaluationRegime):
        raise ValueError("regime must be an EvaluationRegime")
    completed = [member for member in ensemble.members if member.status == "completed"]
    registry = None
    if completed:
        model = completed[0].model
        assert model is not None
        if target is not None and target != model.target:
            raise ValueError("Declared evaluation target scale must match ensemble members")
        target, registry = model.target, model.species
    if not isinstance(target, TargetSpec):
        raise ValueError("All members unavailable: declare target metadata for coverage reporting")
    regions, queries = _query_inputs(region, pairs)
    if registry is not None and any(species not in registry for _, species in queries):
        raise ValueError("Query species must belong to the ensemble registry")
    supports = _support_inputs(support, regions, queries, target, regime)
    keys = tuple(
        (name, species, pair) for (name, species), selected in queries.items() for pair in selected
    )
    columns_by_group = {
        group: [i for i, key in enumerate(keys) if key[:2] == group] for group in queries
    }
    member_ids = ensemble.expected_member_ids
    assert member_ids is not None
    by_id = {member.identity.member_id: member for member in ensemble.members}
    values = jnp.full((len(member_ids), len(keys)), jnp.nan)
    variances = None if regime.prediction_mode == "marginal" else jnp.full_like(values, jnp.nan)
    reasons, failures, prediction_failures, access_records, provenance_records = {}, {}, {}, {}, {}
    statuses = {member_id: by_id[member_id].status for member_id in member_ids}
    for row, member_id in enumerate(member_ids):
        member = by_id[member_id]
        member_reasons: list[str | None] = [None] * len(keys)
        reasons[member_id] = member_reasons
        if member.status != "completed":
            member_reasons[:] = [f"member_{member.status}"] * len(keys)
            failures[member_id] = member.failure or MemberFailure(
                "member", "Pending", "Member is pending"
            )
            continue
        model = member.model
        assert model is not None
        access_records[member_id], provenance_records[member_id] = {}, {}
        for (name, species), columns in columns_by_group.items():
            declared = supports.get((name, species))
            try:
                access = _access(model, name, species, declared, regime)
                access_records[member_id].setdefault(name, {})[species] = access
                eligible = []
                for column in columns:
                    reason = _exclusion(
                        keys[column][2],
                        member.fold.held_out_units.get(name, {}).get(species, ()),
                        access,
                        regime,
                    )
                    member_reasons[column] = reason
                    if reason is None:
                        eligible.append(column)
                if not eligible:
                    continue
                predicted, model_variance, provenance = _member_predict(
                    model,
                    regions[name],
                    species,
                    tuple(keys[column][2] for column in eligible),
                    regime,
                    declared,
                    access,
                )
                # Validate the destination precision before accepting either output.
                # A finite scoped-float64 conditional result may overflow float32.
                predicted = jnp.asarray(predicted, dtype=values.dtype)
                if not bool(jnp.isfinite(predicted).all()):
                    raise FloatingPointError("Member points overflow OOF storage precision")
                if model_variance is not None:
                    assert variances is not None
                    model_variance = jnp.asarray(model_variance, dtype=variances.dtype)
                    if not bool(jnp.isfinite(model_variance).all()):
                        raise FloatingPointError(
                            "Conditional variance overflows OOF storage precision"
                        )
                values = values.at[row, jnp.asarray(eligible)].set(predicted)
                if model_variance is not None:
                    assert variances is not None
                    variances = variances.at[row, jnp.asarray(eligible)].set(model_variance)
                if provenance is not None:
                    provenance_records[member_id].setdefault(name, {})[species] = provenance
            except Exception as error:
                prediction_failures.setdefault(member_id, {}).setdefault(name, {})[species] = (
                    MemberFailure("prediction", type(error).__name__, str(error))
                )
                for column in columns:
                    if member_reasons[column] is None:
                        member_reasons[column] = "prediction_failed"
    counts = np.asarray(jnp.isfinite(values).sum(axis=0))
    covered = counts > 0
    mean, spread = jnp.full(len(keys), jnp.nan), jnp.full(len(keys), jnp.nan)
    for column in np.flatnonzero(covered):
        member_mean, member_spread = _summarize_members(
            values[jnp.isfinite(values[:, column]), column]
        )
        mean = mean.at[column].set(member_mean.astype(mean.dtype))
        spread = spread.at[column].set(member_spread.astype(spread.dtype))
    predictions = {}
    for (name, species), columns in columns_by_group.items():
        predictions.setdefault(name, {})[species] = Prediction(
            mean[jnp.asarray(columns)],
            queries[name, species],
            name,
            species,
            target,
            regions[name].sampling_units[species].ids,
        )
    return OOFPrediction(
        predictions,
        regime,
        member_ids,
        values,
        spread,
        counts,
        covered,
        {identity: tuple(items) for identity, items in reasons.items()},
        statuses,
        failures,
        prediction_failures,
        access_records,
        provenance_records,
        variances,
    )


def score_out_of_fold(prediction: OOFPrediction, observations) -> OOFEvaluation:
    """Read query targets only after prediction, scoring each covered pair once.

    Observations are region/species mappings matching every queried group. Each
    query requires a measured target, even when uncovered; extra measured pairs
    are ignored. No fitting or prediction occurs here. Arithmetic follows the
    configured JAX precision, and errors are in original genetic target units.
    """
    if not isinstance(prediction, OOFPrediction) or prediction.scale != "original":
        raise ValueError("Provide original-scale OOFPrediction values")
    observed_groups = _group_mapping(observations, "Scoring observations")
    if set(observed_groups) != {key[:2] for key in prediction.keys}:
        raise ValueError("Scoring observations must match prediction group identities")
    if len(set(prediction.keys)) != len(prediction.keys):
        raise ValueError("Each canonical region/species/pair must be scored once")
    lookup = {}
    for (name, species), observed in observed_groups.items():
        if not isinstance(observed, PairwiseObservations) or observed.target != prediction.target:
            raise ValueError("Scoring observations require the same declared target scale")
        lookup.update(
            {
                (name, species, tuple(sorted(pair))): value
                for pair, value in zip(
                    observed.observed_pairs, observed.observed_values, strict=True
                )
            }
        )
    try:
        targets = jnp.asarray([lookup[key] for key in prediction.keys])
    except KeyError as error:
        raise ValueError("Every requested query must have an observed scoring target") from error

    def summarize(columns):
        columns = np.asarray(columns)
        mask = prediction.covered_mask[columns]
        n = int(mask.sum())
        if n:
            selected = columns[mask]
            residual = prediction.values[selected] - targets[selected]
            squared_error = jnp.mean(residual**2)
            mse, mae = float(squared_error), float(jnp.mean(jnp.abs(residual)))
            rmse = float(jnp.sqrt(squared_error))
            if not np.isfinite([mse, mae, rmse]).all():
                raise FloatingPointError("Pooled original-scale evaluation metrics are nonfinite")
        else:
            mse = mae = rmse = None
        return EvaluationMetrics(n, len(columns), n / len(columns), mse, rmse, mae)

    overall = summarize(range(len(targets)))
    by_group, by_species = {}, {}
    for name, species in sorted(observed_groups):
        columns = [i for i, key in enumerate(prediction.keys) if key[:2] == (name, species)]
        by_group.setdefault(name, {})[species] = summarize(columns)
    for species in sorted({key[1] for key in prediction.keys}):
        by_species[species] = summarize(
            [i for i, key in enumerate(prediction.keys) if key[1] == species]
        )
    return OOFEvaluation(
        prediction,
        targets,
        overall.n_pairs,
        overall.n_query_pairs,
        overall.coverage,
        overall.mse,
        overall.rmse,
        overall.mae,
        by_group,
        by_species,
    )


def evaluate_ensemble(
    ensemble: Ensemble,
    region,
    observations,
    *,
    partitions=None,
    regime: EvaluationRegime | None = None,
    support=None,
) -> OOFEvaluation:
    """Extract nested query identities, predict without measurements, then score.

    Regions/observations follow native fit collection forms. Optional partitions
    are matching query-role group mappings selecting observed pairs. Omitting
    partitions requests every measured pair, including zero-coverage queries.
    """
    inputs = _normalize_inputs(region, observations, partitions)
    regions, queries, observed_by_region = {}, {}, {}
    target = None
    for prepared, species, observed, partition in inputs:
        if target is None:
            target = observed.target
        elif observed.target != target:
            raise ValueError("Group scoring targets must have identical scale contracts")
        if not set(observed.sampling_unit_ids).issubset(prepared.sampling_units[species].ids):
            raise ValueError("Scoring observation labels require prepared query locations")
        measured = _canonical_pairs(observed.observed_pairs)
        if partition is not None:
            if (partition.region_name, partition.species_name, partition.role) != (
                prepared.name,
                species,
                "query",
            ):
                raise ValueError(
                    "Evaluation partitions require matching query roles and group identities"
                )
            if not set(partition.pairs).issubset(measured):
                raise ValueError("Query partition selects unobserved scoring pairs")
            measured = partition.pairs
        regions[prepared.name] = prepared
        queries.setdefault(prepared.name, {})[species] = measured
        observed_by_region.setdefault(prepared.name, {})[species] = observed
    prediction = predict_out_of_fold(
        ensemble, regions, queries, regime=regime, support=support, target=target
    )
    return score_out_of_fold(prediction, observed_by_region)
