"""Predict held-out pairs without query targets, then score fixed predictions.

Deployment prediction uses every ensemble member and imposes no holdout rule.
Out-of-fold (OOF) prediction instead selects members for each region-qualified
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
Each unique covered region/pair receives equal metric weight after eligible member
points are averaged. Conditional model-scale variance and descriptive original-unit
member disagreement are separate diagnostics. See ``docs/evaluation.md`` for a
complete synthetic workflow and the operation-specific precision contracts.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .calibration import MLPECalibrationModule
from .data import (
    ObservationPartition,
    PairwiseObservations,
    RegionBatch,
    TargetSpec,
)
from .ensemble import Ensemble, MemberFailure, _summarize_members
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
    """Describe one member's actual target access in one query region.

    Pair identities are unordered and region-qualified by ``region_name``.
    Listed endpoints count as seen, except declared support under the explicitly
    permissive support policy. MLPE population IDs with prior-only effects do
    not themselves imply observed-target access.

    Attributes:
        region_name: Region to which all recorded pairs belong.
        training_pairs: Targets used to fit the distance model.
        validation_pairs: Targets used to select its fitted state.
        calibration_pairs: Targets used by the current regional MLPE fit; empty
            for direct calibration or a region without an MLPE fit.
        calibration_roles: Observation-use role for each calibration pair,
            aligned with ``calibration_pairs``.
        support_pairs: Exactly declared support targets for this call; empty
            outside support mode.
        prediction_mode: ``"marginal"``, ``"known_effects"`` or ``"support"``.
        encoder_access_declared: Whether the model declares distance-model
            training access. An entirely empty ``training_pairs={}`` means
            unknown access and is conservatively excluded. An untrained model
            can truthfully declare ``training_pairs={region_name: ()}``.
    """

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
    """Eligible member points, original-scale means and per-pair coverage.

    Let ``k`` be the complete requested member count and ``m`` the number of
    unique region/pair queries. Columns follow ``keys``: sorted region names,
    then lexicographically sorted endpoint labels and pairs. Numerical arrays
    follow configured JAX precision; labels, access records and NumPy coverage
    metadata live on the host. No query targets are stored here.

    Attributes:
        predictions: Region-name mapping to common ``Prediction`` records.
            Each contains original-scale mean values ``(m_region,)``, canonical
            pairs, the shared target and the full prepared-region sampling-unit
            order. Uncovered points are NaN. ``as_matrix()`` represents stored
            points only, with missing off-diagonals and structural zero diagonal
            for dissimilarities.
        regime: Endpoint, prediction and support policies actually requested.
        member_ids: Complete requested member order, length ``k``, including
            failed and pending members.
        member_values: JAX array ``(k, m)`` of original-scale points. A finite
            entry is a successful contribution; excluded or failed entries are
            NaN. Each member's target inverse precedes averaging.
        member_spread: JAX vector ``(m,)`` of descriptive population SD
            (ddof=0) over successful member points in original target units.
            One contributor gives zero; no contributors gives NaN. This is
            neither predictive variance nor a confidence interval.
        eligible_counts: Host NumPy integer vector ``(m,)`` counting successful
            finite contributions, not merely nominally eligible members.
        covered_mask: Host NumPy boolean vector ``(m,)``, true where counts
            exceed zero.
        exclusion_reasons: Member-ID mapping to ``m`` reasons in column order.
            ``None`` denotes success. Other entries include unknown training
            access, own-target access, no regional holdout, insufficient unseen
            holdouts, unavailable members and ``"prediction_failed"``.
        member_statuses: Every requested member's ``"completed"``, ``"failed"``
            or ``"pending"`` fitting status. Prediction failure does not alter
            a completed fit's status.
        member_failures: Unavailable-member diagnostics as ``MemberFailure``;
            pending members receive a synthetic pending diagnostic.
        prediction_failures: Member-ID then region-name mapping to prediction
            failures, including numerical, unsupported-mode or provenance
            failures. Other members may still provide coverage.
        access: Member-ID then region-name mapping to ``EvaluationAccess``
            snapshots, retained when that access was successfully obtained.
        conditioning_provenance: Member-ID then region-name mapping to
            successful ``MLPEPredictionProvenance`` records. Marginal mode has
            no regional records; conditional records must agree with declared
            calibration/support use.
        member_model_variances: ``None`` for marginal mode (unreported, not zero).
            Conditional modes return a JAX array ``(k, m)`` with per-member
            effect-posterior plus residual and jitter predictive variance;
            unavailable entries are NaN. No ensemble variance is inferred.
        variance_scale: Always ``"model"``: variances use squared transformed
            target units, separately from original-unit member disagreement.
        excluded_uncertainty: Uncertainty excluded from conditional variance:
            distance-model parameters (``"encoder"``), fixed-effect coefficients
            and variance parameters.
        keys: Derived ``m`` canonical ``(region_name, (left, right))`` identities.
        values: Derived JAX vector ``(m,)`` of regional mean points in key order.
        target: Shared genetic meaning, original units and fitted transform.
        scale: Always ``"original"`` for point values.
        eligible_member_ids: Tuple of ``m`` tuples naming successful contributors
            in requested member order.
        coverage: Fraction of unique queries with at least one contributor.
        coverage_status: ``"complete"``, ``"partial"`` or ``"none"`` according
            to the covered mask, independently of unavailable-member status.
    """

    predictions: dict[str, Prediction]
    regime: EvaluationRegime
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array
    eligible_counts: np.ndarray
    covered_mask: np.ndarray
    exclusion_reasons: dict[str, tuple[str | None, ...]]
    member_statuses: dict[str, str]
    member_failures: dict[str, MemberFailure]
    prediction_failures: dict[str, dict[str, MemberFailure]]
    access: dict[str, dict[str, EvaluationAccess]]
    conditioning_provenance: dict = field(default_factory=dict)
    member_model_variances: jax.Array | None = None
    variance_scale: str = "model"
    excluded_uncertainty: tuple[str, ...] = (
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    )

    @property
    def keys(self) -> tuple[tuple[str, tuple[str, str]], ...]:
        """Return sorted region/pair identities in member-output column order."""
        return tuple(
            (name, pair)
            for name in sorted(self.predictions)
            for pair in self.predictions[name].pairs
        )

    @property
    def values(self) -> jax.Array:
        """Original-scale means in the same canonical order as ``keys``."""
        values = [self.predictions[name].values for name in sorted(self.predictions)]
        with jax.enable_x64(
            jax.config.read("jax_enable_x64") or any(value.dtype == jnp.float64 for value in values)
        ):
            return jnp.concatenate(values)

    @property
    def target(self) -> TargetSpec:
        """Return the shared target meaning, original units and fitted transform."""
        return next(iter(self.predictions.values())).target

    @property
    def scale(self) -> str:
        """Return ``"original"`` for the regional and derived point values."""
        return next(iter(self.predictions.values())).scale

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
        """Return the fraction of unique queries with a successful contributor."""
        return float(np.mean(self.covered_mask))

    @property
    def coverage_status(self):
        """Return ``"complete"``, ``"partial"`` or ``"none"`` point coverage."""
        return (
            "complete"
            if self.covered_mask.all()
            else "partial"
            if self.covered_mask.any()
            else "none"
        )


@dataclass(frozen=True)
class OOFEvaluation:
    """Score fixed original-scale OOF means with equal covered-pair weights.

    Aggregation precedes scoring: multiple member predictions do not multiply
    a pair's metric weight. Regions with more covered pairs contribute more
    pooled weight, independently of equal-region weighting during training.

    Attributes:
        predictions: The original ``OOFPrediction`` passed to scoring, retaining
            all query identities, uncovered points, access and failures.
        observed_values: JAX vector ``(m,)`` of original-scale measured targets
            aligned with ``predictions.keys``, including uncovered queries.
            Conversion and metric arithmetic follow configured JAX precision.
        n_pairs: Number of covered unique region/pair identities scored.
        n_query_pairs: Total requested unique identities, covered or uncovered.
        coverage: ``n_pairs / n_query_pairs``.
        mse: Mean squared error in squared original target units, or ``None``
            when ``n_pairs == 0``.
        rmse: Root mean squared error in original units, or ``None`` at zero
            coverage.
        mae: Mean absolute error in original units, or ``None`` at zero coverage.
    """

    predictions: OOFPrediction
    observed_values: jax.Array
    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None


def _canonical_pairs(pairs, universe=None):
    """Deduplicate evaluation identities; ordinary prediction rejects duplicates."""
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
    """Read the fitted model's access, rather than infer it from nominal folds."""
    head = (
        model.calibration.regions.get(name)
        if isinstance(model.calibration, MLPECalibrationModule)
        else None
    )
    calibration = (
        () if head is None else tuple((min(a, b), max(a, b)) for a, b in head.calibration_pairs)
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
    """Require the same endpoint to satisfy nominal holdout and actual non-access."""
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
    """Accept conditional points only when returned access matches the declaration."""
    provenance = None
    variances = None
    if regime.prediction_mode == "marginal":
        values = model.predict(region, pairs).values
    else:
        if regime.prediction_mode == "known_effects":
            result = model.predict_known_effects(region, pairs)
        else:
            result = model.predict_with_support(
                region, pairs, support.observations, support_partition=support.partition
            )
        provenance = result.provenance
        if (
            result.prediction.scale != "original"
            or result.prediction.target != model.target
            or result.prediction.region_name != region.name
            or tuple(tuple(sorted(pair)) for pair in result.prediction.pairs) != pairs
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
        values, variances = result.prediction.values, result.model_variance
    if values.shape != (len(pairs),) or not bool(jnp.isfinite(values).all()):
        raise FloatingPointError(
            "Member prediction must return aligned finite original-scale values"
        )
    if variances is not None and (
        variances.shape != values.shape or not bool(jnp.isfinite(variances).all())
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
    """Predict query identities with eligible members before reading query targets.

    A member contributes only when the query's own target was never consumed and
    the regime's required endpoints are both nominal regional holdouts and
    untouched by actual training, validation-selection and calibration access.
    Strict support policy also treats support endpoints as seen. These checks
    apply separately to each unique region/pair, before its genetic prediction.
    Only eligible pairs are calibrated or inverse-transformed, so unrequested or
    excluded inverse-transform failures do not prevent valid selected points.

    Args:
        ensemble: ``Ensemble`` retaining every requested completed, failed or
            pending member and its population fold. Completed members must
            share target and feature contracts.
        region: One ``RegionBatch``, a nonempty sequence with unique names, or a
            mapping whose keys equal each batch's name. Each batch supplies
            query/support locations and compatible feature meanings; no query
            genetic measurements are needed.
        pairs: For one region, a nonempty iterable of labelled pairs or matching
            query-role ``ObservationPartition``. For shared regions, a mapping
            from every region name to one such selection. Labels must identify
            distinct located endpoints. Reversals and repeats are deduplicated;
            output sorts region names and canonical unordered pair identities.
            This differs from ordinary prediction, which rejects duplicates.
        regime: ``EvaluationRegime``; ``None`` selects marginal prediction with
            ``both_unseen`` endpoints and strict ``unseen`` support policy.
            Known-effect/support modes require regional MLPE calibration.
        support: Required only in support mode. Supply ``EvaluationSupport``
            for one region, or a mapping declaring every query region exactly.
            Each support target contract must match the ensemble; measured
            support must equal its support-role partition and be disjoint from
            all query identities. Calibration-target reuse fails per member.
        target: Optional ``TargetSpec`` metadata, never query measurements.
            Inferred from completed members when available, and must agree if
            supplied. Required if every member is failed or pending.

    Returns:
        ``OOFPrediction`` with one common ``Prediction`` per region and aligned
        member points ``(n_members, m)``, counts, access and visible failures.
        Successful original-scale member points are equally averaged per query.
        Uncovered values/spreads are NaN and counts are zero. OOF numerical
        arrays follow configured JAX precision; standalone conditional operations
        and aggregation use scoped float64, which does not change global policy.
        Resistance prediction requires explicitly enabled x64. No model is
        refitted or mutated and no query targets are read.

    Raises:
        ValueError: Composition, regime, regional/pair forms, labels, target
            contracts or support declarations are incompatible; or all members
            are unavailable without explicit target metadata. Per-member
            prediction/provenance failures are recorded in the result instead.
        FloatingPointError: Aggregating finite contributors produces nonfinite
            mean or spread. Member numerical prediction failures are recorded
            and can leave coverage from other members intact.
    """
    if not isinstance(ensemble, Ensemble):
        raise ValueError("Provide an Ensemble with its complete requested composition")
    regime = EvaluationRegime() if regime is None else regime
    if not isinstance(regime, EvaluationRegime):
        raise ValueError("regime must be an EvaluationRegime")
    completed = [member for member in ensemble.members if member.status == "completed"]
    if completed:
        completed_model = completed[0].model
        assert completed_model is not None  # Ensemble validates completed members.
        inferred = completed_model.target
        if target is not None and target != inferred:
            raise ValueError("Declared evaluation target scale must match ensemble members")
        target = inferred
    if not isinstance(target, TargetSpec):
        raise ValueError("All members unavailable: declare target metadata for coverage reporting")
    regions, queries = _query_inputs(region, pairs)
    supports = _support_inputs(support, regions, queries, target, regime)
    keys = tuple((name, pair) for name in sorted(queries) for pair in queries[name])
    member_ids = ensemble.expected_member_ids
    assert member_ids is not None  # Ensemble normalizes its requested composition.
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
        assert model is not None  # A completed member retains its inference model.
        access_records[member_id], provenance_records[member_id] = {}, {}
        for name, prepared in regions.items():
            columns = [index for index, (region_name, _) in enumerate(keys) if region_name == name]
            try:
                access = _access(model, name, supports.get(name), regime)
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
                    model,
                    prepared,
                    tuple(keys[column][1] for column in eligible),
                    regime,
                    supports.get(name),
                    access,
                )
                values = values.at[row, jnp.asarray(eligible)].set(predicted)
                if model_variance is not None:
                    assert variances is not None
                    variances = variances.at[row, jnp.asarray(eligible)].set(model_variance)
                if provenance is not None:
                    provenance_records[member_id][name] = provenance
            except Exception as error:
                prediction_failures.setdefault(member_id, {})[name] = MemberFailure(
                    "prediction", type(error).__name__, str(error)
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
    regional_predictions = {}
    for name in sorted(queries):
        columns = jnp.asarray(
            [index for index, (region_name, _) in enumerate(keys) if region_name == name]
        )
        regional_predictions[name] = Prediction(
            mean[columns], queries[name], name, target, regions[name].sampling_unit_ids
        )
    return OOFPrediction(
        regional_predictions,
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
    """Score fixed OOF means once per covered unique region/pair.

    Query targets enter only here; changing them cannot alter the supplied
    predictions, access or coverage. Eligible original-scale member points have
    already been averaged. Every covered pair has equal pooled metric weight,
    irrespective of contributor count or region; uncovered pairs are retained
    for coverage reporting and excluded from errors.

    Args:
        prediction: Original-scale ``OOFPrediction`` generated without query
            targets, containing unique canonical region/pair identities.
        observations: ``PairwiseObservations`` for a single prediction region,
            or a mapping with exactly the prediction region names. Targets must
            have the identical ``TargetSpec`` and contain every requested query,
            even uncovered ones. Extra measured pairs are ignored; unmeasured
            pairs are never filled with zeros.

    Returns:
        ``OOFEvaluation`` retaining the unchanged prediction object and aligned
        original-scale measured targets ``(m,)``. MSE uses squared original
        units; RMSE and MAE use original units. All three are ``None`` when no
        query is covered. Target conversion and metric arithmetic follow the
        configured JAX precision. No model, prediction or observations are
        mutated, and no fitting or new prediction is performed.

    Raises:
        ValueError: Prediction scale/identities, regional observations or target
            metadata are incompatible, or any query lacks a measured target.
        FloatingPointError: Covered-pair metric arithmetic is nonfinite.
    """
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
        targets = jnp.asarray([lookup[key] for key in prediction.keys])
    except KeyError as error:
        raise ValueError("Every requested query must have an observed scoring target") from error
    n = int(np.sum(prediction.covered_mask))
    if n:
        residual = prediction.values[prediction.covered_mask] - targets[prediction.covered_mask]
        squared_error = jnp.mean(residual**2)
        mse, mae = float(squared_error), float(jnp.mean(jnp.abs(residual)))
        rmse = float(jnp.sqrt(squared_error))
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
    """Generate eligible predictions, then score measured query targets.

    This convenience operation extracts pair identities and target metadata from
    observations, calls ``predict_out_of_fold`` without query measurements, then
    calls ``score_out_of_fold``. Use those two calls explicitly to retain a fixed
    prediction before reading or changing scoring targets. Endpoint and actual
    target-access rules apply even when partitions call a pair a query.

    Args:
        ensemble: Complete requested ``Ensemble`` composition, including any
            failed or pending outcomes.
        region: One ``RegionBatch``, a nonempty sequence of uniquely named
            batches, or a mapping keyed by matching region names.
        observations: Original-scale ``PairwiseObservations`` for one region.
            For shared regions, supply a sequence aligned with the supplied
            region order or a mapping with exactly the region names. All targets
            must have identical ``TargetSpec`` contracts and observation labels
            must have prepared locations. Missing measurements remain absent.
        partitions: Optional query-role ``ObservationPartition`` for one region,
            or an aligned sequence/name-matching mapping for shared regions.
            Each partition must select only observed pairs in its own region.
            ``None`` requests every measured pair; ineligible pairs remain
            present with zero coverage. A selected region needs at least one pair.
        regime: ``EvaluationRegime``; ``None`` uses marginal, ``both_unseen``
            prediction with strict support policy. Conditional modes require
            regional MLPE calibration and retain actual target-access rules.
        support: Support-mode ``EvaluationSupport`` for one region, or a mapping
            covering every query region. It must declare exactly measured,
            target-compatible support disjoint from all requested queries.
            Omit outside support mode.

    Returns:
        ``OOFEvaluation`` with canonical unique-pair predictions, complete member
        diagnostics and coverage, followed by equal covered-pair metrics on
        original units. Several contributors are averaged before one pair is
        scored. Zero coverage gives NaN points and ``None`` metrics. Precision
        and failure reporting follow ``predict_out_of_fold`` and
        ``score_out_of_fold``. No model is refitted or mutated.

    Raises:
        ValueError: Inputs cannot be aligned, labels/targets/roles are incompatible,
            selected pairs lack observations, or query/support declarations are
            invalid. Target metadata comes from observations even if all members
            are unavailable; member prediction failures remain in the result.
        FloatingPointError: Aggregate or scoring arithmetic is nonfinite.
    """
    inputs = _normalize_inputs(region, observations, partitions)
    regions, queries, observed_by_region = {}, {}, {}
    target = None
    for prepared, observed, partition in inputs:
        if observed is None:
            raise ValueError("Evaluation scoring requires pairwise observations")
        if target is None:
            target = observed.target
        elif observed.target != target:
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
