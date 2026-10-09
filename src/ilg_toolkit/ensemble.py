"""Independent population-holdout fits and calibrated deployment prediction."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral

import jax
import numpy as np

from .config import FitConfig
from .data import ObservationPartition, PreparedRegion, TargetSpec
from .models import ConductanceModel, EmbeddingDistanceModel
from .predictor import Prediction, Predictor
from .training import FitResult, TrainingState, _normalize_inputs, fit


@dataclass(frozen=True)
class PopulationFold:
    """Declared endpoint holdout and observed target roles for each region.

    Training pairs cannot touch held-out units. Validation is optional and its
    target access is retained by the predictor, even when it overlaps the query
    partition. A nominal query assignment does not itself establish eligibility
    for out-of-fold evaluation after validation, recalibration or conditioning.
    """

    fold_id: str
    held_out_units: Mapping[str, tuple[str, ...]]
    training: Mapping[str, ObservationPartition]
    query: Mapping[str, ObservationPartition] = field(default_factory=dict)
    validation: Mapping[str, ObservationPartition] = field(default_factory=dict)
    query_regime: str = "both_unseen"

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a nonempty stable identifier")
        if self.query_regime not in {"both_unseen", "at_least_one_unseen"}:
            raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
        held_out = {name: tuple(labels) for name, labels in self.held_out_units.items()}
        if not held_out or any(
            not isinstance(name, str)
            or not name
            or not labels
            or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)
            for name, labels in held_out.items()
        ):
            raise ValueError("Each fold region requires distinct labelled held-out units")
        held_out = {name: tuple(sorted(labels)) for name, labels in held_out.items()}
        object.__setattr__(self, "held_out_units", held_out)
        for role in ("training", "query", "validation"):
            partitions = dict(getattr(self, role))
            if any(
                not isinstance(partition, ObservationPartition)
                or name != partition.region_name
                or partition.role != role
                for name, partition in partitions.items()
            ):
                raise ValueError(
                    f"Fold {role} partitions must have matching region names and roles"
                )
            if not set(partitions).issubset(held_out):
                raise ValueError("Fold partition regions must have declared endpoint holdouts")
            object.__setattr__(self, role, partitions)
        if set(self.training) != set(held_out):
            raise ValueError("Fold training and held-out region names must match exactly")
        for name, partition in self.training.items():
            if any(set(pair) & set(held_out[name]) for pair in partition.pairs):
                raise ValueError("Fold training pairs cannot include held-out endpoints")
        for name, partition in self.query.items():
            count = 2 if self.query_regime == "both_unseen" else 1
            if any(len(set(pair) & set(held_out[name])) < count for pair in partition.pairs):
                raise ValueError("Fold query pairs violate the declared endpoint holdout regime")
        for name, partition in self.validation.items():
            if set(partition.pairs) & set(self.training[name].pairs):
                raise ValueError("Fold training and validation pairs must be disjoint")


def _seed(*parts):
    encoded = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def generate_population_folds(
    region,
    observations,
    *,
    n_folds: int,
    holdout_size: int | Mapping[str, int],
    seed: int = 0,
    query_regime: str = "both_unseen",
) -> tuple[PopulationFold, ...]:
    """Generate repeated seeded population holdouts, with no validation target access.

    Draws are independent by fold and region identity and may overlap between
    folds. Training uses observed pairs with neither endpoint held out. Queries
    use both held-out endpoints by default, or at least one when declared.
    Unobserved pairs stay absent. A draw with no observed training/query pairs
    fails explicitly; supplied folds can express a different study design.
    """
    n_folds = _integer(n_folds, "n_folds", minimum=1)
    seed = _integer(seed, "seed")
    if query_regime not in {"both_unseen", "at_least_one_unseen"}:
        raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
    inputs = _normalize_inputs(region, observations, None)
    names = {prepared.name for prepared, _, _ in inputs}
    if isinstance(holdout_size, Mapping) and set(holdout_size) != names:
        raise ValueError("holdout_size mapping must match region names exactly")
    folds = []
    for index in range(n_folds):
        held_out, training, query = {}, {}, {}
        fold_id = f"fold-{index:04d}"
        for prepared, observed, _ in inputs:
            if any(kind != "population" for kind in prepared.sampling_unit_kinds):
                raise ValueError("Population holdout generation requires population sampling units")
            observed.aligned_pairs(prepared)
            size = _integer(
                holdout_size[prepared.name] if isinstance(holdout_size, Mapping) else holdout_size,
                "holdout_size",
                minimum=1,
            )
            if size > len(prepared.sampling_unit_ids) - 2:
                raise ValueError("holdout_size must leave at least two training sampling units")
            labels = sorted(prepared.sampling_unit_ids)
            rng = np.random.default_rng(_seed("ilg-holdout-v1", seed, index, prepared.name))
            selected = set(rng.choice(labels, size=size, replace=False).tolist())
            held_out[prepared.name] = tuple(sorted(selected))
            pairs = tuple(sorted(tuple(sorted(pair)) for pair in observed.observed_pairs))
            train_pairs = tuple(pair for pair in pairs if not selected.intersection(pair))
            threshold = 2 if query_regime == "both_unseen" else 1
            query_pairs = tuple(
                pair for pair in pairs if len(selected.intersection(pair)) >= threshold
            )
            if not train_pairs or not query_pairs:
                raise ValueError(
                    f"{fold_id}/{prepared.name}: holdout has no observed training or query pairs; "
                    "supply explicit folds or adjust the holdout"
                )
            training[prepared.name] = ObservationPartition(prepared.name, train_pairs, "training")
            query[prepared.name] = ObservationPartition(prepared.name, query_pairs, "query")
        folds.append(PopulationFold(fold_id, held_out, training, query, query_regime=query_regime))
    return tuple(folds)


@dataclass(frozen=True)
class MemberIdentity:
    """Stable fold/declared-seed identity and actual uint32 initialization seed."""

    fold_id: str
    initialization_seed: int
    effective_seed: int
    member_id: str

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a nonempty stable identifier")
        seed = _integer(self.initialization_seed, "initialization_seed")
        if (
            self.effective_seed != _seed("ilg-member-v1", self.fold_id, seed)
            or self.member_id != f"{self.fold_id}:seed={seed}"
        ):
            raise ValueError("Member identity does not match the stable fold/seed derivation")


def ensemble_member_identity(fold_id: str, initialization_seed: int) -> MemberIdentity:
    """Derive initialization from identity, independent of execution order."""
    seed = _integer(initialization_seed, "initialization_seed")
    if not isinstance(fold_id, str) or not fold_id:
        raise ValueError("fold_id must be a nonempty stable identifier")
    return MemberIdentity(
        fold_id, seed, _seed("ilg-member-v1", fold_id, seed), f"{fold_id}:seed={seed}"
    )


@dataclass(frozen=True)
class MemberFailure:
    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EnsembleMember:
    """One requested member, with its own fit/provenance or an explicit failure."""

    identity: MemberIdentity
    fold: PopulationFold
    status: str
    predictor: Predictor | None = None
    fit_result: FitResult | None = None
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
            not isinstance(self.predictor, Predictor) or self.failure is not None
        ):
            raise ValueError("A completed member requires a predictor and no failure")
        if self.status == "failed" and (self.failure is None or self.predictor is not None):
            raise ValueError("A failed member requires an explicit failure and no predictor")
        if self.status == "pending" and (self.predictor is not None or self.failure is not None):
            raise ValueError("A pending member has no predictor or failure yet")


@dataclass(frozen=True)
class EnsemblePrediction:
    """Equal-weight original-scale means and descriptive population SD (ddof=0)."""

    values: np.ndarray
    sampling_unit_ids: tuple[str, ...]
    target: TargetSpec
    member_ids: tuple[str, ...]
    member_values: np.ndarray
    member_spread: np.ndarray
    scale: str = "original"


@dataclass(frozen=True)
class EnsembleSurface:
    """Descriptive conductance summaries; no resistance is solved from this mean."""

    values: np.ndarray
    member_ids: tuple[str, ...]
    member_values: np.ndarray
    member_spread: np.ndarray
    region_name: str


def aggregate_predictions(
    predictions: Mapping[str, Prediction],
    *,
    expected_member_ids: Sequence[str] | None = None,
) -> EnsemblePrediction:
    """Average compatible predictions AFTER each member's inverse transformation."""
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
        if prediction.sampling_unit_ids != first.sampling_unit_ids:
            raise ValueError("Ensemble predictions must align sampling-unit identities and order")
        member_values = np.asarray(prediction.values)
        if (
            member_values.shape != np.asarray(first.values).shape
            or not np.isfinite(member_values).all()
        ):
            raise ValueError("Ensemble member predictions must have aligned finite values")
        values.append(member_values)
    stacked = np.stack(values)
    mean, spread = _summarize_members(stacked)
    return EnsemblePrediction(
        mean,
        first.sampling_unit_ids,
        first.target,
        ids,
        stacked,
        spread,
    )


def _summarize_members(values):
    values = np.asarray(values, dtype=np.float64)
    # Divide before summing and scale before squaring to avoid avoidable overflow
    # when individual original-scale predictions are large but finite.
    mean = np.sum(values / len(values), axis=0)
    scale = np.max(np.abs(values), axis=0)
    normalized = values / np.where(scale > 0, scale, 1)
    spread = np.std(normalized, axis=0, ddof=0) * scale
    if not np.isfinite(mean).all() or not np.isfinite(spread).all():
        raise FloatingPointError("Ensemble mean or descriptive spread is nonfinite")
    return mean, spread


@dataclass(frozen=True)
class Ensemble:
    """All requested member outcomes; deployment prediction requires every member."""

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
        completed = [member.predictor for member in members if member.status == "completed"]
        if completed:
            first = completed[0]
            if any(
                p.target != first.target
                or p.feature_count != first.feature_count
                or p.feature_names != first.feature_names
                for p in completed
            ):
                raise ValueError("Ensemble members require compatible target and feature contracts")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "expected_member_ids", expected)

    @property
    def failures(self):
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
        return {member.identity.member_id: member.predictor for member in self.members}

    def predict(self, region: PreparedRegion) -> EnsemblePrediction:
        """Deployment marginal means, without query targets or eligibility filtering."""
        predictors = self._completed()
        predictions = {
            member_id: predictor.predict(region) for member_id, predictor in predictors.items()
        }
        return aggregate_predictions(predictions, expected_member_ids=self.expected_member_ids)

    def landscape_scores(self, region: PreparedRegion) -> dict[str, np.ndarray]:
        """Return each member's raw scores separately; their scales can differ."""
        return {
            member_id: predictor.landscape_scores(region)
            for member_id, predictor in self._completed().items()
        }

    def conductance_surfaces(self, region: PreparedRegion) -> EnsembleSurface:
        """Return an explicitly descriptive surface mean, separate from genetic means."""
        surfaces = {
            member_id: predictor.conductance_surface(region)
            for member_id, predictor in self._completed().items()
        }
        ids = self.expected_member_ids
        stacked = np.stack([surfaces[member_id] for member_id in ids])
        mean, spread = _summarize_members(stacked)
        return EnsembleSurface(mean, ids, stacked, spread, region.name)


class _ProgressFailure(Exception):
    def __init__(self, error):
        self.error = error


def _fold_inputs(region, observations, fold):
    if not isinstance(fold, PopulationFold):
        raise ValueError("fold must be a PopulationFold")
    inputs = _normalize_inputs(region, observations, fold.training)
    if set(fold.held_out_units) != {prepared.name for prepared, _, _ in inputs}:
        raise ValueError("Fold region identities must match all supplied regions")
    for prepared, observed, partition in inputs:
        if any(kind != "population" for kind in prepared.sampling_unit_kinds):
            raise ValueError("Population folds require population sampling units")
        if not set(fold.held_out_units[prepared.name]).issubset(prepared.sampling_unit_ids):
            raise ValueError("Fold held-out endpoints must belong to the prepared region")
        observed.aligned_pairs(prepared, partition)
        for partitions in (fold.validation, fold.query):
            if prepared.name in partitions:
                observed.aligned_pairs(prepared, partitions[prepared.name])
    return inputs


def fit_ensemble_member(
    region,
    observations,
    *,
    fold: PopulationFold,
    initialization_seed: int,
    config: FitConfig | None = None,
    model_factory: Callable[[jax.Array], ConductanceModel | EmbeddingDistanceModel] | None = None,
    state: TrainingState | None = None,
    on_epoch: Callable[[MemberIdentity, TrainingState], None] | None = None,
) -> EnsembleMember:
    """Fit or continue one independent member, the persistence orchestration seam.

    ``model_factory`` receives the member's deterministic initialization key and
    must construct a fresh encoder. Continuation uses the state encoder instead.
    Callback errors propagate so interrupted persistence cannot be mistaken for
    an ordinary failed member. Training/initialization errors return a failure.
    """
    inputs = _fold_inputs(region, observations, fold)
    identity = ensemble_member_identity(fold.fold_id, initialization_seed)
    config = replace(
        config or (state.config if state is not None else FitConfig()), seed=identity.effective_seed
    )
    regions = {prepared.name: prepared for prepared, _, _ in inputs}
    observed = {prepared.name: values for prepared, values, _ in inputs}
    validation = None
    if fold.validation:
        validation = (
            {name: regions[name] for name in fold.validation},
            {name: observed[name] for name in fold.validation},
        )
    options = dict(
        config=config,
        partition=fold.training,
        validation=validation,
        validation_partition=fold.validation if fold.validation else None,
    )
    if on_epoch is not None:

        def progress(current):
            try:
                on_epoch(identity, current)
            except Exception as error:
                raise _ProgressFailure(error) from error

        options["on_epoch"] = progress
    stage = "initialization"
    try:
        if state is not None:
            options["state"] = state
        elif model_factory is not None:
            init_key, _ = jax.random.split(jax.random.PRNGKey(identity.effective_seed))
            options["model"] = model_factory(init_key)
            if not isinstance(options["model"], (ConductanceModel, EmbeddingDistanceModel)):
                raise TypeError("model_factory must return a distance or conductance encoder")
        stage = "fit"
        result = fit(regions, observed, **options)
    except _ProgressFailure as failure:
        raise failure.error from failure
    except Exception as error:
        return EnsembleMember(
            identity, fold, "failed", failure=MemberFailure(stage, type(error).__name__, str(error))
        )
    return EnsembleMember(identity, fold, "completed", result.predictor, result)


def fit_ensemble(
    region,
    observations,
    *,
    folds: Sequence[PopulationFold] | None = None,
    n_folds: int | None = None,
    holdout_size: int | Mapping[str, int] | None = None,
    fold_seed: int = 0,
    query_regime: str = "both_unseen",
    initialization_seeds: Sequence[int] | None = None,
    config: FitConfig | None = None,
    model_factory: Callable[[jax.Array], ConductanceModel | EmbeddingDistanceModel] | None = None,
    member_states: Mapping[str, TrainingState] | None = None,
    on_epoch: Callable[[MemberIdentity, TrainingState], None] | None = None,
    on_member: Callable[[EnsembleMember], None] | None = None,
) -> Ensemble:
    """Sequential independent fits from explicit or reproducibly generated folds.

    Default generated holdouts are query-only. Explicit validation uses its
    targets for encoder selection and is recorded. All requested outcomes are
    returned, including failures; prediction refuses to drop failed members.
    Per-member continuation and epoch/member callbacks support external durable
    orchestration without making persistence mandatory.
    """
    config = config or FitConfig()
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
    if len({identity.effective_seed for identity in identities}) != len(identities):
        raise ValueError("Initialization seed collision; use different fold identifiers or seeds")
    states = {} if member_states is None else dict(member_states)
    if not set(states).issubset(identity.member_id for identity in identities):
        raise ValueError(
            "Continuation state keys must belong to the requested ensemble composition"
        )
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
                on_epoch=on_epoch,
            )
            members.append(member)
            if on_member is not None:
                on_member(member)
    return Ensemble(tuple(members), tuple(identity.member_id for identity in identities))
