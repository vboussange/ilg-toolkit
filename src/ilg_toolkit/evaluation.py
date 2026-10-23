"""Out-of-fold (OOF) predictions, coverage, and original-scale evaluation.

Eligibility tracks observation use separately for each region–species group.
Query measurements enter only after eligible ensemble predictions are fixed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
from .ensemble import (
    Ensemble,
    FoldKey,
    MemberFailure,
    _group_names,
    _group_sort_key,
    _summarize_members,
)
from .models.base import region_for_species, select_species
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
    """Observation use for one ensemble member and region–species group.

    Training, validation, calibration, and support pairs identify sampling units
    whose genetic observations were used. ``encoder_access_declared`` records
    whether the model supplies any training-pair provenance.
    """

    region_name: str
    training_pairs: tuple[tuple[str, str], ...]
    validation_pairs: tuple[tuple[str, str], ...]
    calibration_pairs: tuple[tuple[str, str], ...]
    calibration_roles: tuple[str, ...]
    support_pairs: tuple[tuple[str, str], ...]
    prediction_mode: str
    encoder_access_declared: bool
    species: str | None = None


@dataclass(frozen=True)
class OOFPrediction:
    """Out-of-fold predictions, eligible-member counts, coverage, and failures.

    Group keys are region names for unnamed species and ``(region, species)`` for
    named species; ``keys`` orders all arrays by region, species, and query pair.
    Uncovered predictions and descriptive member spread are NaN, while conditional
    variances remain on the model scale. ``species`` retains the modeled species
    order so scoring can include per-species metrics for multispecies models.
    """

    predictions: dict[FoldKey, Prediction]
    target: TargetSpec
    regime: EvaluationRegime
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array
    eligible_counts: np.ndarray
    covered_mask: np.ndarray
    exclusion_reasons: dict[str, tuple[str | None, ...]]
    member_statuses: dict[str, str]
    member_failures: dict[str, MemberFailure]
    prediction_failures: dict[str, dict[FoldKey, MemberFailure]]
    access: dict[str, dict[FoldKey, EvaluationAccess]]
    member_model_variances: jax.Array | None = None
    variance_scale: str = "model"
    excluded_uncertainty: tuple[str, ...] = (
        "encoder",
        "fixed_effect_coefficients",
        "variance_parameters",
    )
    species: tuple[str, ...] | None = None

    @property
    def keys(self) -> tuple[tuple[FoldKey, tuple[str, str]], ...]:
        """Return sorted region/pair identities in member-output column order."""
        return tuple(
            (name, pair)
            for name in sorted(self.predictions, key=_group_sort_key)
            for pair in self.predictions[name].pairs
        )

    @property
    def values(self) -> jax.Array:
        """Original-scale means in the same canonical order as ``keys``."""
        values = [
            self.predictions[name].values for name in sorted(self.predictions, key=_group_sort_key)
        ]
        with jax.enable_x64(
            jax.config.read("jax_enable_x64") or any(value.dtype == jnp.float64 for value in values)
        ):
            return jnp.concatenate(values)

    @property
    def scale(self) -> str:
        """Return ``"original"`` for the regional and derived point values."""
        return "original"

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
class EvaluationMetrics:
    """Covered-pair counts and errors on the target's original scale."""

    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None


@dataclass(frozen=True)
class OOFEvaluation:
    """Original-scale errors after averaging eligible predictions for each query pair.

    Every covered pair has equal weight, and uncovered pairs contribute only to
    coverage counts. ``mse``, ``rmse``, and ``mae`` are mean squared error, root
    mean squared error, and mean absolute error, or None at zero coverage.
    ``by_species`` contains the same metrics for queried species when the
    ensemble models more than one species.
    """

    predictions: OOFPrediction
    observed_values: jax.Array
    n_pairs: int
    n_query_pairs: int
    coverage: float
    mse: float | None
    rmse: float | None
    mae: float | None
    by_species: dict[str, EvaluationMetrics] | None = None


def _evaluation_metrics(values, targets, covered):
    n = int(np.sum(covered))
    if n:
        residual = values[covered] - targets[covered]
        mse = float(jnp.mean(residual**2))
        mae = float(jnp.mean(jnp.abs(residual)))
        rmse = float(jnp.sqrt(mse))
        if not np.isfinite([mse, mae, rmse]).all():
            raise FloatingPointError("Original-scale evaluation metrics are nonfinite")
    else:
        mse = mae = rmse = None
    return EvaluationMetrics(n, len(targets), n / len(targets), mse, rmse, mae)


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


def _query_inputs(region, pairs, order, species):
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
    selected_species = select_species(order, species)
    if (
        order is not None
        and len(order) > 1
        and any(prepared.species is None for prepared in regions.values())
    ):
        raise ValueError("Multispecies queries require sampling-unit species labels")
    groups = {
        name if species_name is None else (name, species_name): prepared
        for name, prepared in regions.items()
        for species_name, _ in selected_species
        if prepared.species is None or species_name in prepared.species
    }
    if not groups or {prepared.name for prepared in groups.values()} != set(regions):
        raise ValueError("Every query region requires sampling units for a selected species")
    if not isinstance(pairs, Mapping):
        if len(regions) != 1:
            raise ValueError("Regional query pairs must be mapped by region name")
        pairs = {identity: pairs for identity in groups}
    elif set(pairs) == set(regions):
        pairs = {identity: pairs[prepared.name] for identity, prepared in groups.items()}
    if not set(pairs).issubset(groups):
        raise ValueError("Query pair mapping keys must match the selected region–species groups")
    groups = {identity: groups[identity] for identity in pairs}
    if {prepared.name for prepared in groups.values()} != set(regions):
        raise ValueError("Every query region requires selected pairs")
    queries = {}
    for identity in sorted(groups, key=_group_sort_key):
        name, species_name = _group_names(identity)
        prepared = region_for_species(groups[identity], species_name)
        selected = pairs[identity]
        if isinstance(selected, ObservationPartition):
            if selected.region_name != name or selected.role != "query":
                raise ValueError("Evaluation partitions require matching query roles and regions")
            selected = selected.pairs
        queries[identity] = _canonical_pairs(selected, set(prepared.sampling_unit_ids))
        if not queries[identity]:
            raise ValueError("Each query region requires at least one pair")
    return groups, queries


def _observation_species(observations: PairwiseObservations, order: tuple[str, ...] | None):
    if observations.species is None and order is not None and len(order) == 1:
        return order[0]
    return observations.species


def _ensemble_species(ensemble: Ensemble) -> tuple[str, ...] | None:
    for member in ensemble.members:
        if member.model is not None:
            return member.model.species
    names = {
        key[1]
        for member in ensemble.members
        for key in member.fold.held_out_units
        if isinstance(key, tuple)
    }
    return tuple(sorted(names)) or None


def _support_inputs(support, regions, queries, target, regime, order):
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
    for identity, declared in support.items():
        name, species = _group_names(identity)
        if not isinstance(declared, EvaluationSupport) or declared.partition.region_name != name:
            raise ValueError("Support requires matching EvaluationSupport region declarations")
        if _observation_species(declared.observations, order) != species:
            raise ValueError("Support observations must match the query species")
        if declared.observations.target != target:
            raise ValueError("Support target scale must match the ensemble target")
        prepared = region_for_species(regions[identity], species)
        declared.observations.aligned_values(prepared)
        if set(declared.partition.pairs) & set(queries[identity]):
            raise ValueError("Support/query overlap: query targets cannot be used as support")
    return support


def _access(model, identity, support, regime):
    """Read the fitted model's access, rather than infer it from nominal folds."""
    name, species = _group_names(identity)
    index = select_species(model.species, species)[0][1]
    key = (name, index) if model.distance_model.num_species > 1 else name
    head = (
        model.calibration.regions.get(key)
        if isinstance(model.calibration, MLPECalibrationModule)
        else None
    )
    calibration = (
        () if head is None else tuple((min(a, b), max(a, b)) for a, b in head.calibration_pairs)
    )
    return EvaluationAccess(
        name,
        _canonical_pairs(model.training_pairs.get(key, ())),
        _canonical_pairs(model.validation_pairs.get(key, ())),
        calibration,
        () if head is None else head.calibration_roles,
        () if support is None else support.partition.pairs,
        regime.prediction_mode,
        bool(model.training_pairs),
        species,
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


def _member_predict(model, region, pairs, regime, support, species):
    """Genetic predictions and conditional variance for eligible query pairs."""
    result = model.predict(
        region,
        species=species,
        pairs=pairs,
        known_effects=regime.prediction_mode == "known_effects",
        support=None if support is None else support.observations,
    )
    values = result.values
    variances = result.extras.get("model_variance")
    if values.shape != (len(pairs),) or not bool(jnp.isfinite(values).all()):
        raise FloatingPointError(
            "Member prediction must return aligned finite original-scale values"
        )
    if variances is not None and (
        variances.shape != values.shape or not bool(jnp.isfinite(variances).all())
    ):
        raise FloatingPointError("Conditional member model variance must be aligned and finite")
    return values, variances


def predict_out_of_fold(
    ensemble: Ensemble,
    region,
    pairs,
    *,
    species: str | Sequence[str] | None = None,
    regime: EvaluationRegime | None = None,
    support: EvaluationSupport | Mapping[FoldKey, EvaluationSupport] | None = None,
    target: TargetSpec | None = None,
) -> OOFPrediction:
    """Genetic predictions from ensemble members with unseen query sampling units.

    ``species`` selects one name or a sequence; multispecies models require a
    selection, and ``pairs`` can map region names or ``(region, species)`` keys
    to query pairs or query partitions. Eligibility combines each member's
    holdout with training, validation, calibration, and support observation use,
    under ``regime``; support declarations use the same group keys as queries.
    The result reports successful-member coverage and failures, and every-member
    failure requires ``target`` metadata for reporting.
    """
    if not isinstance(ensemble, Ensemble):
        raise ValueError("Provide an Ensemble with its complete requested composition")
    regime = EvaluationRegime() if regime is None else regime
    if not isinstance(regime, EvaluationRegime):
        raise ValueError("regime must be an EvaluationRegime")
    completed = [member for member in ensemble.members if member.status == "completed"]
    order = _ensemble_species(ensemble)
    if completed:
        completed_model = completed[0].model
        assert completed_model is not None  # Ensemble validates completed members.
        inferred = completed_model.target
        if target is not None and target != inferred:
            raise ValueError("Declared evaluation target scale must match ensemble members")
        target = inferred
    if not isinstance(target, TargetSpec):
        raise ValueError("All members unavailable: declare target metadata for coverage reporting")
    regions, queries = _query_inputs(region, pairs, order, species)
    supports = _support_inputs(support, regions, queries, target, regime, order)
    keys = tuple((name, pair) for name in queries for pair in queries[name])
    member_ids = ensemble.expected_member_ids
    assert member_ids is not None  # Ensemble normalizes its requested composition.
    by_id = {member.identity.member_id: member for member in ensemble.members}
    values = jnp.full((len(member_ids), len(keys)), jnp.nan)
    variances = None if regime.prediction_mode == "marginal" else jnp.full_like(values, jnp.nan)
    reasons, failures, prediction_failures, access_records = {}, {}, {}, {}
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
        access_records[member_id] = {}
        for name, prepared in regions.items():
            columns = [index for index, (region_name, _) in enumerate(keys) if region_name == name]
            try:
                access = _access(model, name, supports.get(name), regime)
                access_records[member_id][name] = access
                holdout_key = name
                if model.distance_model.num_species == 1 and name not in member.fold.held_out_units:
                    holdout_key = prepared.name
                held_out = member.fold.held_out_units.get(holdout_key, ())
                eligible = []
                for column in columns:
                    reason = _exclusion(keys[column][1], held_out, access, regime)
                    member_reasons[column] = reason
                    if reason is None:
                        eligible.append(column)
                if not eligible:
                    continue
                predicted, model_variance = _member_predict(
                    model,
                    prepared,
                    tuple(keys[column][1] for column in eligible),
                    regime,
                    supports.get(name),
                    _group_names(name)[1],
                )
                values = values.at[row, jnp.asarray(eligible)].set(predicted)
                if model_variance is not None:
                    assert variances is not None
                    variances = variances.at[row, jnp.asarray(eligible)].set(model_variance)
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
    for name in queries:
        columns = jnp.asarray(
            [index for index, (region_name, _) in enumerate(keys) if region_name == name]
        )
        regional_predictions[name] = Prediction(
            mean[columns],
            queries[name],
            regions[name].name,
            region_for_species(regions[name], _group_names(name)[1]).sampling_unit_ids,
            _group_names(name)[1],
        )
    return OOFPrediction(
        regional_predictions,
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
        variances,
        species=order,
    )


def score_out_of_fold(prediction: OOFPrediction, observations) -> OOFEvaluation:
    """Score fixed out-of-fold predictions on the original target scale.

    Supply one observations object or a species-labeled sequence for one region,
    or a mapping from region names to those inputs. Each covered region–species
    query pair receives equal metric weight after eligible members are averaged.
    Scoring preserves predictions, observation-use records, and coverage.
    """
    if not isinstance(prediction, OOFPrediction) or prediction.scale != "original":
        raise ValueError("Provide original-scale OOFPrediction values")
    names = {_group_names(identity)[0] for identity in prediction.predictions}
    if not isinstance(observations, Mapping) and len(names) == 1:
        observations = {next(iter(names)): observations}
    if not isinstance(observations, Mapping) or set(observations) != names:
        raise ValueError("Scoring observations must match prediction region identities")
    if len(set(prediction.keys)) != len(prediction.keys):
        raise ValueError("Each canonical region/pair must be scored once")
    lookup = {}
    identities = set()
    for name, regional in observations.items():
        regional = (regional,) if isinstance(regional, PairwiseObservations) else regional
        if not isinstance(regional, Sequence) or not regional:
            raise ValueError("Scoring requires observations for each region–species group")
        for observed in regional:
            if (
                not isinstance(observed, PairwiseObservations)
                or observed.target != prediction.target
            ):
                raise ValueError("Scoring observations require the same target scale")
            species = _observation_species(observed, prediction.species)
            identity = name if species is None else (name, species)
            if identity in identities:
                raise ValueError(
                    "Scoring requires one observations object per region–species group"
                )
            identities.add(identity)
            lookup.update(
                {
                    (identity, tuple(sorted(pair))): value
                    for pair, value in zip(
                        observed.observed_pairs, observed.observed_values, strict=True
                    )
                }
            )
    if identities != set(prediction.predictions):
        raise ValueError("Scoring observations must match prediction region–species groups")
    try:
        targets = jnp.asarray([lookup[key] for key in prediction.keys])
    except KeyError as error:
        raise ValueError("Every requested query must have an observed scoring target") from error
    values = prediction.values
    metrics = _evaluation_metrics(values, targets, prediction.covered_mask)
    by_species = None
    if prediction.species is not None and len(prediction.species) > 1:
        by_species = {}
        for species in sorted(
            {
                value.species
                for value in prediction.predictions.values()
                if value.species is not None
            }
        ):
            columns = np.asarray(
                [
                    i
                    for i, (identity, _) in enumerate(prediction.keys)
                    if _group_names(identity)[1] == species
                ],
                dtype=np.intp,
            )
            by_species[species] = _evaluation_metrics(
                values[columns], targets[columns], prediction.covered_mask[columns]
            )
    return OOFEvaluation(
        prediction,
        targets,
        metrics.n_pairs,
        metrics.n_query_pairs,
        metrics.coverage,
        metrics.mse,
        metrics.rmse,
        metrics.mae,
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
    """Predict eligible query pairs, then score their genetic observations.

    Observations and optional query partitions align with regions and species
    as in ``fit``. The ``regime`` and ``support`` arguments follow
    ``predict_out_of_fold``. Per-species errors appear only when the ensemble
    models more than one species.
    """
    inputs = _normalize_inputs(region, observations, partitions)
    regions, queries, observed_by_region = {}, {}, {}
    order = _ensemble_species(ensemble)
    species = set()
    target = None
    for prepared, observed, partition in inputs:
        if observed is None:
            raise ValueError("Evaluation scoring requires pairwise observations")
        if target is None:
            target = observed.target
        elif observed.target != target:
            raise ValueError("Regional scoring targets must have identical scale contracts")
        name = _observation_species(observed, order)
        selected = region_for_species(prepared, name)
        if not set(observed.sampling_unit_ids).issubset(selected.sampling_unit_ids):
            raise ValueError("Scoring observation labels require prepared query locations")
        measured = _canonical_pairs(observed.observed_pairs)
        if partition is not None:
            if partition.region_name != prepared.name or partition.role != "query":
                raise ValueError("Evaluation partitions require matching query roles and regions")
            if not set(partition.pairs).issubset(measured):
                raise ValueError("Query partition selects unobserved scoring pairs")
            measured = partition.pairs
        identity = prepared.name if name is None else (prepared.name, name)
        if identity in queries:
            raise ValueError("Provide one observations object per region–species group")
        regions[prepared.name], queries[identity] = prepared, measured
        observed_by_region.setdefault(prepared.name, []).append(observed)
        if name is not None:
            species.add(name)
    prediction = predict_out_of_fold(
        ensemble,
        regions,
        queries,
        species=tuple(sorted(species)) or None,
        regime=regime,
        support=support,
        target=target,
    )
    return score_out_of_fold(prediction, observed_by_region)
