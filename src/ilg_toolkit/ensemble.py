"""Fold ensemble fitting and original-scale genetic predictions."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral
from typing import cast, overload

import jax
import jax.numpy as jnp
import numpy as np

from .config import TrainingConfig
from .data import ObservationPartition, RegionBatch
from .model import LandscapeGeneticsModel
from .models import AbstractDistanceModel, ConductanceModel
from .models.base import region_for_species
from .prediction import Prediction
from .training import TrainingState, _normalize_inputs, fit

type FoldKey = str | tuple[str, str]


def _group_identity(region: RegionBatch, observations) -> FoldKey:
    name = observations.species
    if name is None and region.species is not None:
        names = set(region.species)
        if len(names) != 1:
            raise ValueError("Multispecies observations require species labels")
        name = next(iter(names))
    return region.name if name is None else (region.name, name)


def _group_names(identity: FoldKey) -> tuple[str, str | None]:
    return (identity, None) if isinstance(identity, str) else identity


def _group_sort_key(identity: FoldKey) -> tuple[str, str]:
    region, species = _group_names(identity)
    return region, species or ""


def _fold_groups(region, observations):
    groups = {}
    for item in _normalize_inputs(region, observations, None):
        identity = _group_identity(item.region, item.observations)
        if identity in groups:
            raise ValueError("Provide one observations object per region–species group")
        groups[identity] = item
    return dict(sorted(groups.items(), key=lambda item: _group_sort_key(item[0])))


@dataclass(frozen=True)
class PopulationFold:
    """Sampling-unit holdouts and observation roles for each region–species group.

    Keys are region names for unnamed species and ``(region, species)`` for named
    species. Training pairs use sampling units outside the holdout, and optional
    validation pairs record target access for model selection.
    """

    fold_id: str
    held_out_units: Mapping[FoldKey, tuple[str, ...]]
    training: Mapping[FoldKey, ObservationPartition]
    query: Mapping[FoldKey, ObservationPartition] = field(default_factory=dict)
    validation: Mapping[FoldKey, ObservationPartition] = field(default_factory=dict)
    query_regime: str = "both_unseen"

    def __post_init__(self):
        if not isinstance(self.fold_id, str) or not self.fold_id:
            raise ValueError("fold_id must be a nonempty stable identifier")
        if self.query_regime not in {"both_unseen", "at_least_one_unseen"}:
            raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
        held_out = {name: tuple(labels) for name, labels in self.held_out_units.items()}
        if not held_out or any(
            not (
                isinstance(name, str)
                and bool(name)
                or isinstance(name, tuple)
                and len(name) == 2
                and all(isinstance(part, str) and part for part in name)
            )
            or not labels
            or len(set(labels)) != len(labels)
            or any(not isinstance(label, str) or not label for label in labels)
            for name, labels in held_out.items()
        ):
            raise ValueError("Each fold region requires distinct labeled held-out units")
        held_out = {name: tuple(sorted(labels)) for name, labels in held_out.items()}
        object.__setattr__(self, "held_out_units", held_out)
        for role in ("training", "query", "validation"):
            partitions = dict(getattr(self, role))
            if any(
                not isinstance(partition, ObservationPartition)
                or _group_names(name)[0] != partition.region_name
                or partition.role != role
                for name, partition in partitions.items()
            ):
                raise ValueError(
                    f"Fold {role} partitions must have matching region names and roles"
                )
            if not set(partitions).issubset(held_out):
                raise ValueError("Fold partition regions must have sampling-unit holdouts")
            object.__setattr__(self, role, partitions)
        if set(self.training) != set(held_out):
            raise ValueError("Fold training and held-out region names must match exactly")
        for name, partition in self.training.items():
            if any(set(pair) & set(held_out[name]) for pair in partition.pairs):
                raise ValueError("Fold training pairs cannot include held-out sampling units")
        for name, partition in self.query.items():
            count = 2 if self.query_regime == "both_unseen" else 1
            if any(len(set(pair) & set(held_out[name])) < count for pair in partition.pairs):
                raise ValueError("Fold query pairs violate the sampling-unit holdout regime")
        for name, partition in self.validation.items():
            if set(partition.pairs) & set(self.training[name].pairs):
                raise ValueError("Fold training and validation pairs must be disjoint")


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def generate_population_folds(
    region,
    observations,
    *,
    n_folds: int,
    holdout_size: int | Mapping[FoldKey, int],
    seed: int = 0,
    query_regime: str = "both_unseen",
) -> tuple[PopulationFold, ...]:
    """Seeded population holdouts drawn separately for each region–species group.

    Training uses observed pairs outside the holdout, and queries use both held-out
    sampling units by default or at least one when requested. ``holdout_size`` is
    one size or a mapping with the same keys as ``PopulationFold.held_out_units``.
    Every draw requires observed training and query pairs.
    """
    n_folds = _integer(n_folds, "n_folds", minimum=1)
    seed = _integer(seed, "seed")
    if query_regime not in {"both_unseen", "at_least_one_unseen"}:
        raise ValueError("query_regime must be both_unseen or at_least_one_unseen")
    inputs = _fold_groups(region, observations)
    if isinstance(holdout_size, Mapping) and set(holdout_size) != set(inputs):
        raise ValueError("holdout_size mapping must match region–species groups")
    folds = []
    rng = np.random.default_rng(seed)
    for index in range(n_folds):
        held_out, training, query = {}, {}, {}
        fold_id = f"fold-{index:04d}"
        for identity, (region, observed, _) in inputs.items():
            prepared = region_for_species(region, _group_names(identity)[1])
            assert prepared.sampling_unit_kinds is not None
            if any(kind != "population" for kind in prepared.sampling_unit_kinds):
                raise ValueError("Population holdout generation requires population sampling units")
            observed.aligned_pairs(prepared)
            size = _integer(
                holdout_size[identity] if isinstance(holdout_size, Mapping) else holdout_size,
                "holdout_size",
                minimum=1,
            )
            if size > len(prepared.sampling_unit_ids) - 2:
                raise ValueError("holdout_size must leave at least two training sampling units")
            labels = sorted(prepared.sampling_unit_ids)
            selected = set(rng.choice(labels, size=size, replace=False).tolist())
            held_out[identity] = tuple(sorted(selected))
            pairs = tuple(sorted((min(a, b), max(a, b)) for a, b in observed.observed_pairs))
            train_pairs = tuple(pair for pair in pairs if not selected.intersection(pair))
            threshold = 2 if query_regime == "both_unseen" else 1
            query_pairs = tuple(
                pair for pair in pairs if len(selected.intersection(pair)) >= threshold
            )
            if not train_pairs or not query_pairs:
                raise ValueError(
                    f"{fold_id}/{prepared.name}: holdout has no observed training or query pairs; "
                    "supply folds or adjust the holdout"
                )
            training[identity] = ObservationPartition(prepared.name, train_pairs, "training")
            query[identity] = ObservationPartition(prepared.name, query_pairs, "query")
        folds.append(PopulationFold(fold_id, held_out, training, query, query_regime=query_regime))
    return tuple(folds)


@dataclass(frozen=True)
class MemberIdentity:
    """An ensemble member identity combining its fold and initialization seed."""

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
    """An ensemble member identity for the supplied fold and initialization seed."""
    return MemberIdentity(fold_id, initialization_seed)


@dataclass(frozen=True)
class MemberFailure:
    """A fitting, callback, or prediction failure retained for caller inspection.

    ``stage`` identifies the failed operation, ``error_type`` names the exception,
    and ``message`` describes the failure.
    """

    stage: str
    error_type: str
    message: str


@dataclass(frozen=True)
class EnsembleMember:
    """One ensemble member with its fitted model, observation-use records, or failure."""

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
            raise ValueError("Member requires an identity and population fold")
        if self.identity.fold_id != self.fold.fold_id:
            raise ValueError("Member identity and fold must match")
        if self.status not in {"completed", "failed", "pending"}:
            raise ValueError("Member status must be completed, failed or pending")
        if self.status == "completed" and (
            not isinstance(self.model, LandscapeGeneticsModel) or self.failure is not None
        ):
            raise ValueError("A completed member requires a model and no failure")
        if self.status == "failed" and (self.failure is None or self.model is not None):
            raise ValueError("A failed member requires a failure and no model")
        if self.status == "pending" and (self.model is not None or self.failure is not None):
            raise ValueError("A pending member has no model or failure yet")


@dataclass(frozen=True)
class EnsemblePrediction:
    """Deployment genetic predictions averaged across every ensemble member on the original scale.

    ``prediction`` contains the mean, and ``member_values`` has rows in
    ``member_ids`` order and columns in query-pair order.
    Every member must succeed; its target inverse transform precedes float64
    averaging, and ``member_spread`` gives the population standard deviation.

    Note: Member spread describes disagreement, not statistical predictive uncertainty.
    """

    prediction: Prediction
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array


@dataclass(frozen=True)
class EnsembleSurface:
    """Mean and descriptive spread of ensemble members' conductance surfaces."""

    values: jax.Array
    member_ids: tuple[str, ...]
    member_values: jax.Array
    member_spread: jax.Array
    region_name: str


@jax.enable_x64()
def aggregate_predictions(
    predictions: Mapping[str, Prediction],
    *,
    expected_member_ids: Sequence[str] | None = None,
) -> EnsemblePrediction:
    """Equal-weight means and descriptive spread of aligned member predictions.

    Region, species, query pairs, sampling-unit order and finite values must agree.
    Callers supply values in common units; fitted ensembles validate target
    compatibility on their models.
    """
    if not predictions:
        raise ValueError("An ensemble aggregate requires at least one member prediction")
    ids = tuple(predictions) if expected_member_ids is None else tuple(expected_member_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("Missing, extra or duplicate ensemble members; composition cannot shrink")
    if any(not isinstance(value, Prediction) for value in predictions.values()):
        raise ValueError("Aggregate inputs must be labeled original-scale Prediction values")
    first = predictions[ids[0]]
    values = []
    for member_id in ids:
        prediction = predictions[member_id]
        if (
            prediction.sampling_unit_ids != first.sampling_unit_ids
            or prediction.pairs != first.pairs
            or prediction.species != first.species
            or prediction.region_name != first.region_name
        ):
            raise ValueError(
                "Ensemble predictions must align sampling-unit, pair and region identities"
            )
        member_values = jnp.asarray(prediction.values, dtype=jnp.float64)
        if member_values.shape != first.values.shape or not bool(jnp.isfinite(member_values).all()):
            raise ValueError("Ensemble member predictions must have aligned finite values")
        values.append(member_values)
    stacked = jnp.stack(values)
    mean, spread = _summarize_members(stacked)
    return EnsemblePrediction(
        Prediction(mean, first.pairs, first.region_name, first.sampling_unit_ids, first.species),
        ids,
        stacked,
        spread,
    )


@jax.jit
def _member_moments(values):
    # Divide before summing and scale before squaring to prevent overflow.
    # Binary exponent scaling preserves member spread when a very large scale
    # would cause its reciprocal to underflow.
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
    """Fold ensemble retaining every requested member's outcome.

    Completed members share their species order, target, and feature contract.
    Deployment requires every member to succeed; out-of-fold evaluation reports
    partial coverage with unavailable members and failures retained.
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
                    "Ensemble members require matching target, features and species order"
                )
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "expected_member_ids", expected)

    @property
    def failures(self):
        """Failed-fit diagnostics keyed by ensemble member identifier."""
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

    @overload
    def predict(
        self, region: RegionBatch, pairs=None, *, species: str | None = None
    ) -> EnsemblePrediction: ...

    @overload
    def predict(
        self, region: RegionBatch, pairs=None, *, species: Sequence[str]
    ) -> dict[str, EnsemblePrediction]: ...

    def predict(
        self, region: RegionBatch, pairs=None, *, species: str | Sequence[str] | None = None
    ) -> EnsemblePrediction | dict[str, EnsemblePrediction]:
        """Deployment genetic predictions averaged over every ensemble member.

        Species selection follows ``LandscapeGeneticsModel.predict``; a sequence
        returns a mapping and shares each member's pooled backbone. Each member's
        target inverse transform precedes averaging, with descriptive spread in
        original target units. Every requested member must succeed.
        """
        models = self._completed()
        pairs = None if pairs is None else tuple(pairs)
        if isinstance(species, str) or species is None:
            return aggregate_predictions(
                {
                    key: model.predict(region, pairs=pairs, species=species)
                    for key, model in models.items()
                },
                expected_member_ids=self.expected_member_ids,
            )
        predictions = {
            key: model.predict(region, pairs=pairs, species=species)
            for key, model in models.items()
        }
        return {
            name: aggregate_predictions(
                {key: values[name] for key, values in predictions.items()},
                expected_member_ids=self.expected_member_ids,
            )
            for name in species
        }

    @overload
    def ecological_distances(
        self, region: RegionBatch, *, species: str | None = None
    ) -> dict[str, jax.Array]: ...

    @overload
    def ecological_distances(
        self, region: RegionBatch, *, species: Sequence[str]
    ) -> dict[str, dict[str, jax.Array]]: ...

    def ecological_distances(
        self, region: RegionBatch, *, species: str | Sequence[str] | None = None
    ) -> dict[str, jax.Array] | dict[str, dict[str, jax.Array]]:
        """Ecological-distance matrices for each ensemble member on its own distance scale.

        A species sequence returns a species-to-member mapping and shares each
        member's pooled backbone.
        """
        models = self._completed()
        if isinstance(species, str) or species is None:
            return {
                key: model.distance_model.predict(
                    region, species=species, species_order=model.species
                ).as_matrix()
                for key, model in models.items()
            }
        predictions = {
            key: model.distance_model.predict(region, species=species, species_order=model.species)
            for key, model in models.items()
        }
        return {
            name: {key: values[name].as_matrix() for key, values in predictions.items()}
            for name in species
        }

    @overload
    def conductance_surfaces(
        self, region: RegionBatch, *, species: str | None = None
    ) -> EnsembleSurface: ...

    @overload
    def conductance_surfaces(
        self, region: RegionBatch, *, species: Sequence[str]
    ) -> dict[str, EnsembleSurface]: ...

    @jax.enable_x64()
    def conductance_surfaces(
        self, region: RegionBatch, *, species: str | Sequence[str] | None = None
    ) -> EnsembleSurface | dict[str, EnsembleSurface]:
        """Mean and descriptive spread of ensemble members' conductance surfaces.

        A species sequence returns a mapping and shares each member's pooled backbone.
        """
        surfaces = {}
        for member_id, model in self._completed().items():
            model._validate_region(region)
            if not isinstance(model.distance_model, ConductanceModel):
                raise TypeError("This distance model does not produce a conductance surface")
            surfaces[member_id] = model.distance_model.conductance_surface(
                region, species=species, species_order=model.species
            )
        ids = self.expected_member_ids
        assert ids is not None

        def summarize(values: Mapping[str, jax.Array]) -> EnsembleSurface:
            stacked = jnp.stack([jnp.asarray(values[key], dtype=jnp.float64) for key in ids])
            mean, spread = _summarize_members(stacked)
            return EnsembleSurface(mean, ids, stacked, spread, region.name)

        if isinstance(species, str) or species is None:
            return summarize(cast(dict[str, jax.Array], surfaces))
        return {
            name: summarize({key: values[name] for key, values in surfaces.items()})
            for name in species
        }


class _ProgressFailure(Exception):
    def __init__(self, error):
        self.error = error


def _fold_inputs(region, observations, fold):
    if not isinstance(fold, PopulationFold):
        raise ValueError("fold must be a PopulationFold")
    inputs = _fold_groups(region, observations)
    if set(fold.held_out_units) != set(inputs):
        raise ValueError("Fold identities must match all supplied region–species groups")
    for identity, (region, observed, _) in inputs.items():
        prepared = region_for_species(region, _group_names(identity)[1])
        assert prepared.sampling_unit_kinds is not None
        if any(kind != "population" for kind in prepared.sampling_unit_kinds):
            raise ValueError("Population folds require population sampling units")
        if not set(fold.held_out_units[identity]).issubset(prepared.sampling_unit_ids):
            raise ValueError("Fold held-out sampling units must belong to the region batch")
        observed.aligned_pairs(prepared, fold.training[identity])
        for partitions in (fold.validation, fold.query):
            if identity in partitions:
                observed.aligned_pairs(prepared, partitions[identity])
    return inputs


def fit_ensemble_member(
    region,
    observations,
    *,
    fold: PopulationFold,
    initialization_seed: int,
    species: Sequence[str] | None = None,
    config: TrainingConfig | None = None,
    model_factory: Callable[[jax.Array], AbstractDistanceModel] | None = None,
    state: TrainingState | None = None,
    on_step: Callable[[MemberIdentity, TrainingState], None] | None = None,
) -> EnsembleMember:
    """An ensemble member fitted or continued using its population fold.

    Fresh fitting requires ``config`` and a ``model_factory`` receiving the
    initialization seed as a JAX random key. ``species`` gives the factory model's
    parameter order, as in ``fit``. Callback errors propagate, while initialization
    and training errors return a failed member.
    """
    inputs = _fold_inputs(region, observations, fold)
    identity = ensemble_member_identity(fold.fold_id, initialization_seed)
    if state is None and (config is None or model_factory is None):
        raise ValueError("Fresh ensemble members require config and model_factory")
    if state is not None and state.config.seed != identity.initialization_seed:
        raise ValueError("Continuation state must match the initialization_seed")
    config = state.config if config is None and state is not None else config
    assert config is not None
    config = replace(config, seed=identity.initialization_seed)

    def selected_inputs(partitions):
        regions, observed, selected = {}, {}, {}
        for identity, partition in partitions.items():
            prepared, values, _ = inputs[identity]
            regions[prepared.name] = prepared
            observed.setdefault(prepared.name, []).append(values)
            selected.setdefault(prepared.name, []).append(partition)
        return regions, observed, selected

    regions, observed, training_partition = selected_inputs(fold.training)
    validation_regions, validation_observed, validation_partition = selected_inputs(fold.validation)
    validation = (validation_regions, validation_observed) if fold.validation else None

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
            species=species,
            partition=training_partition,
            validation=validation,
            validation_partition=validation_partition if fold.validation else None,
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
    """Validation of the requested folds and ensemble member identities."""
    if folds is None:
        if n_folds is None or holdout_size is None:
            raise ValueError("Supply folds or both n_folds and holdout_size")
        folds = generate_population_folds(
            region,
            observations,
            n_folds=n_folds,
            holdout_size=holdout_size,
            seed=fold_seed,
            query_regime=query_regime,
        )
    elif n_folds is not None or holdout_size is not None:
        raise ValueError("Use folds or generated-fold options, not both")
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
    holdout_size: int | Mapping[FoldKey, int] | None = None,
    fold_seed: int = 0,
    query_regime: str = "both_unseen",
    initialization_seeds: Sequence[int] | None = None,
    species: Sequence[str] | None = None,
    config: TrainingConfig,
    model_factory: Callable[[jax.Array], AbstractDistanceModel] | None = None,
    member_states: Mapping[str, TrainingState] | None = None,
    on_step: Callable[[MemberIdentity, TrainingState], None] | None = None,
    on_member: Callable[[EnsembleMember], None] | None = None,
) -> Ensemble:
    """Fold ensemble fitted from supplied or seeded population holdouts.

    ``species`` gives each factory model's parameter order, as in ``fit``.
    Every requested member outcome is retained, including initialization or
    training failures. Member states and callbacks support continuation.
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
                species=species,
                config=config,
                model_factory=model_factory,
                state=states.get(identity.member_id),
                on_step=on_step,
            )
            members.append(member)
            if on_member is not None:
                on_member(member)
    return Ensemble(tuple(members), tuple(identity.member_id for identity in identities))
