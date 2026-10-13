"""Independent population-holdout fitting and calibrated ensemble predictions."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .config import TrainingConfig
from .data import ObservationPartition, RegionBatch
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel
from .prediction import Prediction
from .training import TrainingState, _normalize_inputs, fit


@dataclass(frozen=True)
class PopulationFold:
    """Declared endpoint holdout and observed target roles for each region.

    Training pairs cannot touch held-out units. Validation is optional and its
    target access is retained by the model, even when it overlaps the query
    partition. A nominal query assignment does not itself establish eligibility
    for out-of-fold evaluation after validation, recalibration or conditioning.
    """

    fold_id: str
    held_out_units: Mapping[str, Mapping[str, tuple[str, ...]]]
    training: Mapping[str, Mapping[str, ObservationPartition]]
    query: Mapping[str, Mapping[str, ObservationPartition]] = field(default_factory=dict)
    validation: Mapping[str, Mapping[str, ObservationPartition]] = field(default_factory=dict)
    query_regime: str = "both_unseen"

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a nonempty stable identifier")
        if self.query_regime not in {"both_unseen", "at_least_one_unseen"}:
            raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
        declared = _group_mapping(self.held_out_units, "held-out units")
        held_out = {}
        for (region, species), labels in declared.items():
            if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
                raise ValueError("Each fold group requires distinct labelled held-out units")
            labels = tuple(labels)
            if (
                not labels
                or any(not isinstance(v, str) or not v for v in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError("Each fold group requires distinct labelled held-out units")
            held_out.setdefault(region, {})[species] = tuple(sorted(labels))
        object.__setattr__(self, "held_out_units", held_out)
        flattened = {}
        for role in ("training", "query", "validation"):
            partitions = _group_mapping(getattr(self, role), role, allow_empty=role != "training")
            if role == "training" and set(partitions) != set(declared):
                raise ValueError("Fold training and held-out group identities must match exactly")
            if not set(partitions).issubset(declared):
                raise ValueError("Fold partition groups must have declared endpoint holdouts")
            nested = {}
            for (region, species), partition in partitions.items():
                if not isinstance(partition, ObservationPartition) or (
                    partition.region_name,
                    partition.species_name,
                    partition.role,
                ) != (region, species, role):
                    raise ValueError(
                        f"Fold {role} partitions require matching "
                        "region/species identities and roles"
                    )
                nested.setdefault(region, {})[species] = partition
            flattened[role] = partitions
            object.__setattr__(self, role, nested)
        for (region, species), partition in flattened["training"].items():
            if any(set(pair) & set(held_out[region][species]) for pair in partition.pairs):
                raise ValueError("Fold training pairs cannot include held-out endpoints")
        for (region, species), partition in flattened["query"].items():
            count = 2 if self.query_regime == "both_unseen" else 1
            if any(
                len(set(pair) & set(held_out[region][species])) < count for pair in partition.pairs
            ):
                raise ValueError("Fold query pairs violate the declared endpoint holdout regime")
        for identity, partition in flattened["validation"].items():
            if set(partition.pairs) & set(flattened["training"][identity].pairs):
                raise ValueError("Fold training and validation pairs must be disjoint")


def _group_mapping(value, label, *, allow_empty=False):
    """Validate native host group mappings without accepting flat one-group forms."""
    if not isinstance(value, Mapping) or (not value and not allow_empty):
        raise ValueError(f"{label} requires a region-then-species mapping")
    flattened = {}
    for region, groups in value.items():
        if (
            not isinstance(region, str)
            or not region
            or not isinstance(groups, Mapping)
            or not groups
        ):
            raise ValueError(f"{label} requires nonempty named region/species groups")
        for species, item in groups.items():
            if not isinstance(species, str) or not species:
                raise ValueError(f"{label} requires named species identities")
            flattened[region, species] = item
    return flattened


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def generate_population_folds(
    region,
    observations,
    *,
    n_folds: int,
    holdout_size: int | Mapping[str, Mapping[str, int]],
    seed: int = 0,
    query_regime: str = "both_unseen",
) -> tuple[PopulationFold, ...]:
    """Generate repeated seeded population holdouts, with no validation target access.

    Draws use one seeded random generator and may overlap between folds.
    Training uses observed pairs with neither endpoint held out. Queries
    use both held-out endpoints by default, or at least one when declared.
    Unobserved pairs stay absent. A draw with no observed training/query pairs
    fails explicitly; supplied folds can express a different study design.
    """
    n_folds = _integer(n_folds, "n_folds", minimum=1)
    seed = _integer(seed, "seed")
    if query_regime not in {"both_unseen", "at_least_one_unseen"}:
        raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
    inputs = _normalize_inputs(region, observations, None)
    catalogue = {(item.region.name, item.species) for item in inputs}
    sizes = (
        _group_mapping(holdout_size, "holdout_size") if isinstance(holdout_size, Mapping) else None
    )
    if sizes is not None and set(sizes) != catalogue:
        raise ValueError("holdout_size mapping must match region/species groups exactly")
    folds = []
    rng = np.random.default_rng(seed)
    for index in range(n_folds):
        held_out, training, query = {}, {}, {}
        fold_id = f"fold-{index:04d}"
        for prepared, species, observed, _ in inputs:
            observed.aligned_pairs(prepared, species)
            units = prepared.sampling_units[species]
            size = _integer(
                sizes[prepared.name, species] if sizes is not None else holdout_size,
                "holdout_size",
                minimum=1,
            )
            if size > len(units.ids) - 2:
                raise ValueError("holdout_size must leave at least two training sampling units")
            selected = set(rng.choice(sorted(units.ids), size=size, replace=False).tolist())
            held_out.setdefault(prepared.name, {})[species] = tuple(sorted(selected))
            pairs = tuple(sorted((min(a, b), max(a, b)) for a, b in observed.observed_pairs))
            train_pairs = tuple(pair for pair in pairs if not selected.intersection(pair))
            threshold = 2 if query_regime == "both_unseen" else 1
            query_pairs = tuple(
                pair for pair in pairs if len(selected.intersection(pair)) >= threshold
            )
            if not train_pairs or not query_pairs:
                raise ValueError(
                    f"{fold_id}/{prepared.name}/{species}: holdout has no observed "
                    "training or query pairs; supply explicit folds or adjust the holdout"
                )
            training.setdefault(prepared.name, {})[species] = ObservationPartition(
                prepared.name, species, train_pairs, "training"
            )
            query.setdefault(prepared.name, {})[species] = ObservationPartition(
                prepared.name, species, query_pairs, "query"
            )
        folds.append(PopulationFold(fold_id, held_out, training, query, query_regime=query_regime))
    return tuple(folds)


@dataclass(frozen=True)
class MemberIdentity:
    """Readable identity from a fold and its declared initialization seed."""

    fold_id: str
    initialization_seed: int

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a nonempty stable identifier")
        seed = _integer(self.initialization_seed, "initialization_seed")
        if seed >= 2**32:
            raise ValueError("initialization_seed must be in [0, 2**32)")
        object.__setattr__(self, "initialization_seed", seed)

    @property
    def member_id(self) -> str:
        return f"{self.fold_id}:seed={self.initialization_seed}"


def ensemble_member_identity(fold_id: str, initialization_seed: int) -> MemberIdentity:
    """Identify a member without changing its declared initialization seed."""
    return MemberIdentity(fold_id, initialization_seed)


@dataclass(frozen=True)
class MemberFailure:
    """A fitting, callback or prediction failure retained for caller inspection.

    Attributes:
        stage: Operation that failed, such as ``"fit"``, ``"prediction"`` or
            ``"member"`` for an unavailable OOF member.
        error_type: Exception class name, or ``"Pending"`` for a pending member.
        message: Human-readable diagnostic; no failed member is silently dropped.
    """

    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EnsembleMember:
    """One requested member, with its own fit/provenance or an explicit failure."""

    identity: MemberIdentity
    fold: PopulationFold
    status: str
    model: LandscapeGeneticsModel | None = None
    training_state: TrainingState | None = None
    failure: MemberFailure | None = None

    def __post_init__(self):
        if not isinstance(self.identity, MemberIdentity) or not isinstance(
            self.fold, PopulationFold
        ):
            raise ValueError("Member requires an explicit identity and population fold")
        if self.identity.fold_id != self.fold.fold_id:
            raise ValueError("Member identity and fold must match")
        if self.status not in {"completed", "failed", "pending"}:
            raise ValueError("Member status must be completed, failed or pending")
        if self.status == "completed" and (
            not isinstance(self.model, LandscapeGeneticsModel) or self.failure is not None
        ):
            raise ValueError("A completed member requires a model and no failure")
        if self.status == "failed" and (self.failure is None or self.model is not None):
            raise ValueError("A failed member requires an explicit failure and no model")
        if self.status == "pending" and (self.model is not None or self.failure is not None):
            raise ValueError("A pending member has no model or failure yet")


@dataclass(frozen=True)
class EnsemblePrediction:
    """Deployment means over every member, after each target inverse transform.

    This result imposes no holdout or actual-access eligibility rule. All members
    must succeed. Numerical arrays use scoped float64 aggregation without
    changing global JAX precision. No common model-scale mean, predictive
    variance or covariance is inferred.

    Attributes:
        prediction: Common ``Prediction`` with original-scale mean points
            ``(m,)``, ordered/oriented pairs, region, target and full query-region
            sampling-unit order. ``as_matrix()`` represents these points only.
        member_ids: Complete requested member order, of length ``k``.
        member_values: Finite original-scale member points ``(k, m)``, with rows
            in ``member_ids`` order and columns in ``prediction.pairs`` order.
            Each inverse transformation occurs before equal-weight averaging.
        member_spread: Descriptive population SD (ddof=0) ``(m,)`` in original
            target units. One member gives zero disagreement. This is neither
            statistical predictive variance nor a confidence interval and
            excludes within-member conditional variance.
    """

    prediction: Prediction
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array


@dataclass(frozen=True)
class EnsembleSurface:
    """Descriptive conductance summaries; no resistance is solved from this mean."""

    values: jax.Array
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array
    region_name: str
    species_name: str


@jax.enable_x64()
def aggregate_predictions(
    predictions: Mapping[str, Prediction],
    *,
    expected_member_ids: Sequence[str] | None = None,
) -> EnsemblePrediction:
    """Average aligned original-scale points with complete member composition.

    Args:
        predictions: Nonempty member-ID mapping to common ``Prediction``
            records already inverse-transformed into original target units.
            Regions, target contracts, ordered/oriented pairs, sampling-unit
            matrix orders and finite vector shapes must agree exactly.
        expected_member_ids: Optional unique complete member order. Defaults to
            mapping insertion order. Missing or extra members are rejected,
            rather than silently reducing the requested ensemble.

    Returns:
        ``EnsemblePrediction`` containing equal-weight original-scale means,
        member values ``(k, m)`` and descriptive population SD ``(m,)``. Scoped
        float64 aggregation leaves global JAX precision unchanged and performs
        no additional prediction, calibration or model mutation.

    Raises:
        ValueError: Composition, labelled identities, targets, shapes or finite
            member values are incompatible.
        FloatingPointError: Aggregated mean or spread is nonfinite.
    """
    if not predictions:
        raise ValueError("An ensemble aggregate requires at least one member prediction")
    ids = tuple(predictions) if expected_member_ids is None else tuple(expected_member_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("Missing, extra or duplicate ensemble members; composition cannot shrink")
    if any(not isinstance(value, Prediction) for value in predictions.values()):
        raise ValueError("Aggregate inputs must be labelled original-scale Prediction values")
    first = predictions[ids[0]]
    values = []
    for member_id in ids:
        prediction = predictions[member_id]
        if prediction.target != first.target or prediction.scale != "original":
            raise ValueError("Ensemble predictions require compatible original target scales")
        if (
            prediction.sampling_unit_ids != first.sampling_unit_ids
            or prediction.pairs != first.pairs
            or prediction.region_name != first.region_name
            or prediction.species_name != first.species_name
        ):
            raise ValueError(
                "Ensemble predictions must align sampling-unit, pair and region/species identities"
            )
        member_values = jnp.asarray(prediction.values, dtype=jnp.float64)
        if member_values.shape != first.values.shape or not bool(jnp.isfinite(member_values).all()):
            raise ValueError("Ensemble member predictions must have aligned finite values")
        values.append(member_values)
    stacked = jnp.stack(values)
    mean, spread = _summarize_members(stacked)
    return EnsemblePrediction(
        Prediction(
            mean,
            first.pairs,
            first.region_name,
            first.species_name,
            first.target,
            first.sampling_unit_ids,
        ),
        ids,
        stacked,
        spread,
    )


@jax.jit
def _member_moments(values):
    # Divide before summing and scale before squaring to avoid avoidable overflow.
    # Binary exponents avoid the underflowed reciprocal of a very large scale
    # in XLA's division lowering (which would silently erase member spread).
    mean = jnp.sum(values / len(values), axis=0)
    _, exponent = jnp.frexp(jnp.max(jnp.abs(values), axis=0))
    normalized = jnp.ldexp(values, -exponent)
    return mean, jnp.ldexp(jnp.std(normalized, axis=0, ddof=0), exponent)


@jax.enable_x64()
def _summarize_members(values):
    mean, spread = _member_moments(jnp.asarray(values, dtype=jnp.float64))
    if not bool(jnp.isfinite(mean).all() & jnp.isfinite(spread).all()):
        raise FloatingPointError("Ensemble mean or descriptive spread is nonfinite")
    return mean, spread


@dataclass(frozen=True)
class Ensemble:
    """Retain complete requested outcomes for deployment and OOF evaluation.

    Attributes:
        members: Nonempty tuple of uniquely identified ``EnsembleMember``
            outcomes, including completed, failed and pending fits. Completed
            models must share target and feature contracts.
        expected_member_ids: Unique complete requested member order. Defaults to
            ``members`` order; explicit identities must match those outcomes
            exactly. Deployment requires every member completed; OOF can report
            partial coverage with failures retained.
        failures: Derived member-ID mapping to recorded failed-fit diagnostics.
            Pending fits and OOF prediction failures are reported separately.

    Raises:
        ValueError: Composition or completed models' target/feature contracts
            are incompatible.
    """

    members: tuple[EnsembleMember, ...]
    expected_member_ids: tuple[str, ...] | None = None

    def __post_init__(self):
        members = tuple(self.members)
        ids = tuple(member.identity.member_id for member in members)
        expected = ids if self.expected_member_ids is None else tuple(self.expected_member_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or len(set(expected)) != len(expected)
            or set(expected) != set(ids)
        ):
            raise ValueError(
                "Missing, extra or duplicate ensemble members; composition cannot shrink"
            )
        completed = []
        for member in members:
            if member.status == "completed":
                if member.model is None:
                    raise ValueError("A completed member requires a model")
                completed.append(member.model)
        if completed:
            first = completed[0]
            if any(
                p.target != first.target
                or p.feature_count != first.feature_count
                or p.feature_names != first.feature_names
                or p.species != first.species
                for p in completed
            ):
                raise ValueError(
                    "Ensemble members require compatible target, feature "
                    "and ordered species contracts"
                )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "expected_member_ids", expected)

    @property
    def failures(self):
        """Return failed-fit diagnostics keyed by member ID; omit pending fits."""
        return {
            member.identity.member_id: member.failure
            for member in self.members
            if member.status == "failed"
        }

    def _completed(self):
        unavailable = [
            f"{member.identity.member_id} ({member.status})"
            for member in self.members
            if member.status != "completed"
        ]
        if unavailable:
            raise RuntimeError("Ensemble has unavailable members: " + ", ".join(unavailable))
        models = {}
        for member in self.members:
            if member.model is None:
                raise ValueError("A completed member requires a model")
            models[member.identity.member_id] = member.model
        return models

    def predict(
        self, region: RegionBatch, *, species=None, pairs=None
    ) -> dict[str, EnsemblePrediction]:
        """Average original-scale predictions per selected species over all members.

        Selection mirrors LandscapeGeneticsModel.predict: multispecies models
        require explicit species, and pairs is a species-keyed mapping. Every
        requested member must succeed. This deployment operation imposes no
        holdout eligibility rule and consumes no query targets.
        """
        if pairs is not None:
            if not isinstance(pairs, Mapping):
                raise ValueError("pairs must be a mapping keyed by selected species")
            pairs = {name: tuple(selected) for name, selected in pairs.items()}
        outputs = {
            identity: model.predict(region, species=species, pairs=pairs)
            for identity, model in self._completed().items()
        }
        selected = next(iter(outputs.values()))
        return {
            name: aggregate_predictions(
                {identity: result[name] for identity, result in outputs.items()},
                expected_member_ids=self.expected_member_ids,
            )
            for name in selected
        }

    def ecological_distances(
        self, region: RegionBatch, *, species=None
    ) -> dict[str, dict[str, jax.Array]]:
        """Species then member keyed ecological matrices; no genetic calibration."""
        outputs = {
            identity: model.ecological_distances(region, species=species)
            for identity, model in self._completed().items()
        }
        return {
            name: {identity: result[name] for identity, result in outputs.items()}
            for name in next(iter(outputs.values()))
        }

    @jax.enable_x64()
    def conductance_surfaces(
        self, region: RegionBatch, *, species=None
    ) -> dict[str, EnsembleSurface]:
        """Species-keyed descriptive surfaces, distinct from averaged distances."""
        outputs = {
            identity: model.conductance_surface(region, species=species)
            for identity, model in self._completed().items()
        }
        ids = self.expected_member_ids
        assert ids is not None
        summaries = {}
        for name in next(iter(outputs.values())):
            stacked = jnp.stack(
                [jnp.asarray(outputs[identity][name], dtype=jnp.float64) for identity in ids]
            )
            mean, spread = _summarize_members(stacked)
            summaries[name] = EnsembleSurface(mean, ids, stacked, spread, region.name, name)
        return summaries


class _ProgressFailure(Exception):
    def __init__(self, error):
        self.error = error


def _fold_inputs(region, observations, fold):
    if not isinstance(fold, PopulationFold):
        raise ValueError("fold must be a PopulationFold")
    inputs = _normalize_inputs(region, observations, fold.training)
    for prepared, species, observed, partition in inputs:
        if not set(fold.held_out_units[prepared.name][species]).issubset(
            prepared.sampling_units[species].ids
        ):
            raise ValueError("Fold held-out endpoints must belong to the prepared region/species")
        observed.aligned_pairs(prepared, species, partition)
        for partitions in (fold.validation, fold.query):
            selected = partitions.get(prepared.name, {}).get(species)
            if selected is not None:
                observed.aligned_pairs(prepared, species, selected)
    return inputs


def fit_ensemble_member(
    region,
    observations,
    *,
    fold: PopulationFold,
    initialization_seed: int,
    config: TrainingConfig | None = None,
    model_factory: Callable[[jax.Array], AbstractDistanceModel] | None = None,
    state: TrainingState | None = None,
    on_step: Callable[[MemberIdentity, TrainingState], None] | None = None,
) -> EnsembleMember:
    """Fit or continue one independent member, the persistence orchestration seam.

    Fresh fitting requires ``config`` and ``model_factory``. The factory receives
    ``jax.random.PRNGKey(initialization_seed)`` directly and
    must construct a fresh encoder. Continuation uses the state encoder instead.
    Callback errors propagate so interrupted persistence cannot be mistaken for
    an ordinary failed member. Training/initialization errors return a failure.
    """
    inputs = _fold_inputs(region, observations, fold)
    identity = ensemble_member_identity(fold.fold_id, initialization_seed)
    if state is None and (config is None or model_factory is None):
        raise ValueError("Fresh ensemble members require config and model_factory")
    if state is not None and state.config.seed != identity.initialization_seed:
        raise ValueError("Continuation state must match the declared initialization_seed")
    config = state.config if config is None and state is not None else config
    assert config is not None
    config = replace(config, seed=identity.initialization_seed)
    regions = {item.region.name: item.region for item in inputs}
    observed = {}
    for item in inputs:
        observed.setdefault(item.region.name, {})[item.species] = item.observations
    validation = None
    if fold.validation:
        validation = (
            {name: regions[name] for name in fold.validation},
            {
                name: {species: observed[name][species] for species in groups}
                for name, groups in fold.validation.items()
            },
        )

    def progress(current: TrainingState):
        assert on_step is not None
        try:
            on_step(identity, current)
        except Exception as error:
            raise _ProgressFailure(error) from error

    stage = "initialization"
    try:
        model = None
        if state is None:
            assert model_factory is not None
            model = model_factory(jax.random.PRNGKey(identity.initialization_seed))
            if not isinstance(model, AbstractDistanceModel):
                raise TypeError("model_factory must return an AbstractDistanceModel")
        stage = "fit"
        result = fit(
            regions,
            observed,
            distance_model=model,
            config=config,
            partition=fold.training,
            validation=validation,
            validation_partition=fold.validation if fold.validation else None,
            state=state,
            on_step=progress if on_step is not None else None,
        )
    except _ProgressFailure as failure:
        raise failure.error from failure
    except Exception as error:
        return EnsembleMember(
            identity, fold, "failed", failure=MemberFailure(stage, type(error).__name__, str(error))
        )
    return EnsembleMember(identity, fold, "completed", result.model, result)


def _prepare_ensemble(
    region,
    observations,
    *,
    config,
    folds,
    n_folds,
    holdout_size,
    fold_seed,
    query_regime,
    initialization_seeds,
):
    """Validate composition without initializing or fitting any member."""
    if folds is None:
        if n_folds is None or holdout_size is None:
            raise ValueError("Supply explicit folds or both n_folds and holdout_size")
        folds = generate_population_folds(
            region,
            observations,
            n_folds=n_folds,
            holdout_size=holdout_size,
            seed=fold_seed,
            query_regime=query_regime,
        )
    elif n_folds is not None or holdout_size is not None:
        raise ValueError("Use explicit folds or generated-fold options, not both")
    folds = tuple(folds)
    if not folds or any(not isinstance(fold, PopulationFold) for fold in folds):
        raise ValueError("An ensemble requires nonempty PopulationFold inputs")
    if len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError("Ensemble fold_id values must be unique")
    folds = tuple(sorted(folds, key=lambda fold: fold.fold_id))
    for fold in folds:
        _fold_inputs(region, observations, fold)
    seeds = (config.seed,) if initialization_seeds is None else tuple(initialization_seeds)
    seeds = tuple(sorted(_integer(seed, "initialization_seed") for seed in seeds))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("initialization_seeds must be nonempty and unique")
    identities = [ensemble_member_identity(fold.fold_id, seed) for fold in folds for seed in seeds]
    return folds, seeds, identities


def fit_ensemble(
    region,
    observations,
    *,
    folds: Sequence[PopulationFold] | None = None,
    n_folds: int | None = None,
    holdout_size: int | Mapping[str, Mapping[str, int]] | None = None,
    fold_seed: int = 0,
    query_regime: str = "both_unseen",
    initialization_seeds: Sequence[int] | None = None,
    config: TrainingConfig,
    model_factory: Callable[[jax.Array], AbstractDistanceModel] | None = None,
    member_states: Mapping[str, TrainingState] | None = None,
    on_step: Callable[[MemberIdentity, TrainingState], None] | None = None,
    on_member: Callable[[EnsembleMember], None] | None = None,
) -> Ensemble:
    """Sequential independent fits from explicit or reproducibly generated folds.

    Default generated holdouts are query-only. Explicit validation uses its
    targets for encoder selection and is recorded. All requested outcomes are
    returned, including failures; prediction refuses to drop failed members.
    Per-member continuation and step/member callbacks support external durable
    orchestration without making persistence mandatory.
    """
    folds, seeds, identities = _prepare_ensemble(
        region,
        observations,
        config=config,
        folds=folds,
        n_folds=n_folds,
        holdout_size=holdout_size,
        fold_seed=fold_seed,
        query_regime=query_regime,
        initialization_seeds=initialization_seeds,
    )
    states = {} if member_states is None else dict(member_states)
    if not set(states).issubset(identity.member_id for identity in identities):
        raise ValueError(
            "Continuation state keys must belong to the requested ensemble composition"
        )
    if model_factory is None and any(identity.member_id not in states for identity in identities):
        raise ValueError("Fresh ensemble members require model_factory")
    members = []
    for fold in folds:
        for seed in seeds:
            identity = ensemble_member_identity(fold.fold_id, seed)
            member = fit_ensemble_member(
                region,
                observations,
                fold=fold,
                initialization_seed=seed,
                config=config,
                model_factory=model_factory,
                state=states.get(identity.member_id),
                on_step=on_step,
            )
            members.append(member)
            if on_member is not None:
                on_member(member)
    return Ensemble(tuple(members), tuple(identity.member_id for identity in identities))
