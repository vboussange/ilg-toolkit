"""Landscape genetics models combining ecological distances and genetic calibration."""

from collections.abc import Sequence
from typing import cast, final, overload

import equinox as eqx
import jax.numpy as jnp

from .calibration import AbstractCalibrationModule
from .data import PairwiseObservations, RegionBatch, TargetSpec
from .models import AbstractDistanceModel
from .models.base import (
    _query_pairs,
    prepare_distance_context,
    query_distances_from_latent,
    region_for_species,
    select_species,
    validate_species_order,
)
from .models.common import with_dropout_mode
from .prediction import Prediction


@final
class LandscapeGeneticsModel(eqx.Module):
    """A distance model composed with fitted genetic calibration.

    ``species`` associates names with the distance model's parameter rows.
    Feature meanings define the query contract. The calibration owns target
    semantics and statistical prediction behavior.
    """

    distance_model: AbstractDistanceModel
    calibration: AbstractCalibrationModule
    feature_count: int
    feature_names: tuple[str, ...] | None = None
    species: tuple[str, ...] | None = eqx.field(default=None, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement AbstractDistanceModel")
        if not isinstance(self.calibration, AbstractCalibrationModule):
            raise TypeError("calibration must implement AbstractCalibrationModule")
        self.species = validate_species_order(self.species, self.distance_model.num_species)

    @property
    def target(self) -> TargetSpec:
        """The target contract owned by the calibration module."""
        return self.calibration.target

    def _validate_region(self, region: RegionBatch):
        if region.feature_array.shape[-1] != self.feature_count:
            raise ValueError(f"Expected {self.feature_count} feature channels")
        if region.feature_names != self.feature_names:
            raise ValueError(
                "Query feature contract must match training feature meanings and order"
            )

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction: ...

    @overload
    def predict(
        self,
        region: RegionBatch,
        *,
        species: Sequence[str],
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> dict[str, Prediction]: ...

    def predict(
        self,
        region: RegionBatch,
        *,
        species: str | Sequence[str] | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Predict genetic relationships on the target's original scale.

        Requested pairs retain their order and orientation; omitted pairs select
        every distinct pair. A species sequence returns a mapping of predictions.
        The distance model computes ecological distances, and fitted calibration
        converts these into genetic predictions. Maximum-likelihood population effects
        (MLPE) defaults to marginal means;
        ``known_effects=True`` and ``support=...`` condition population effects.

        | Mode | Population effects | Statistical extras |
        | --- | --- | --- |
        | Default | Marginal mean, without fitted effects | `model_values` |
        | `known_effects=True` | Saved posterior from calibration | Mean + variances |
        | `support=observations` | Saved posterior updated with support | Mean + variances |

        Examples
        --------
        Prepare one landscape containing six located populations and fit MLPE
        calibration from measurements among its first four populations::

            import coordax as cx
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            from ilg_toolkit import (
                MLPELoss, PairwiseObservations, RegionBatch, TargetSpec, calibrate,
            )
            from ilg_toolkit.models import LogLinearConductance

            region = RegionBatch(
                "valley", cx.field(jnp.arange(6.0).reshape(1, 6, 1),
                                   "row", "column", "feature"),
                tuple(f"p{i}" for i in range(6)),
                [[0, i] for i in range(6)], feature_names=("elevation",),
            )
            observations = PairwiseObservations.from_pairs(
                [("p0", "p1"), ("p0", "p2"), ("p0", "p3"),
                 ("p1", "p2"), ("p1", "p3"), ("p2", "p3")],
                [1.2, 1.6, 2.9, 1.7, 2.0, 1.3],
                target=TargetSpec("genetic divergence", units="index"),
            )
            model = calibrate(region, observations,
                              LogLinearConductance(1, patch_size=1), loss=MLPELoss())
            query_pairs = [("p4", "p2"), ("p5", "p3")]

        Marginal predictions use only fitted intercept, slope, and distance
        standardization. Population identities do not add saved effects::

            marginal = model.predict(region, pairs=query_pairs)
            marginal.values                 # Genetic means in original units.
            marginal.extras["model_values"] # Means before the target inverse transform.
            marginal.pairs                  # Same alignment as query_pairs.

        Saved-effect predictions add posterior population means learned from the
        calibration observations. Unobserved populations retain zero prior mean
        and prior population variance; calibration/query overlap is allowed::

            saved = model.predict(region, pairs=query_pairs, known_effects=True)
            saved.extras["effect_variance"] # Uncertainty in the summed population effects.
            saved.extras["model_variance"]  # Effect + residual variance + fixed jitter.

        Additional support observations update that saved posterior, including
        newly observed populations, for this prediction only. The landscape
        parameters, coefficients, standardization, and variance components remain
        fixed; the model itself is unchanged::

            support = PairwiseObservations.from_pairs(
                [("p4", "p0"), ("p4", "p1")], [2.7, 2.2], target=model.target,
            )
            conditioned = model.predict(region, pairs=query_pairs, support=support)
            conditioned.values
            conditioned.extras["model_values"]
            conditioned.extras["model_variance"]
            conditioned.extras["effect_variance"]

        All point predictions inverse-transform model-scale means. Conditional
        variances stay on the model scale. Use ``as_matrix()`` or
        ``as_matrix("effect_variance")`` to arrange the corresponding values by
        sampling-unit identity.

        Notes
        -----
        With nonlinear target transforms, conditional variances are not original-scale
        variances, and inverse-transformed means are not bias-corrected expectations.

        Support must have matching target/species metadata and located sampling
        units, with pairs disjoint from both calibration and query pairs, including
        reversals. Shared population identities are allowed. Omitting ``pairs``
        with support requests all pairs and consequently creates invalid overlap.
        Passing both ``known_effects=True`` and support is invalid; direct
        calibration supports neither conditional mode. Every MLPE mode requires
        existing calibration for the query region/species and population units.

        Support conditioning does not establish calibration in a new region.
        Use [`calibrate`][ilg_toolkit.calibrate] to fit a fresh relationship there,
        passing ``species=model.species`` when transferring a multispecies model.
        See the [comprehensive MLPE workflow notebook](../examples/mlpe_workflow.md)
        for training, fresh calibration, evaluation, and checkpoint continuation.
        """
        self._validate_region(region)
        selected = select_species(self.species, species)
        if self.distance_model.num_species > 1 and region.species is None:
            raise ValueError("Multispecies queries require sampling-unit species labels")
        support_by_species = self._support_observations(support, selected)
        distance_model = with_dropout_mode(self.distance_model, inference=True, key=None)
        latent = distance_model.backbone(region.feature_array)
        context = prepare_distance_context(distance_model, region.feature_array)
        results = {}
        for name, index in selected:
            prepared = region_for_species(region, name)
            observations = support_by_species.get(name)
            query_pairs = pairs
            if observations is not None:
                labels, _ = _query_pairs(prepared, pairs)
                query_pairs = labels + observations.observed_pairs
            labels, distances = query_distances_from_latent(
                distance_model,
                latent,
                prepared,
                query_pairs,
                species_index=index,
                context=context,
            )
            if observations is not None:
                labels = labels[: len(labels) - len(observations.observed_pairs)]
            result = self.calibration.predict(
                distances[: len(labels)],
                labels,
                region_name=region.name,
                species_index=index,
                sampling_unit_kinds=prepared.sampling_unit_kinds,
                known_effects=known_effects,
                support=observations,
                support_distances=None if observations is None else distances[len(labels) :],
            )
            results[name] = Prediction(
                jnp.asarray(result.values),
                labels,
                region.name,
                prepared.sampling_unit_ids,
                species=name,
                extras=result.extras,
            )
        if isinstance(species, str) or species is None:
            return results[selected[0][0]]
        return cast(dict[str, Prediction], results)

    def _support_observations(
        self,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None,
        selected: tuple[tuple[str | None, int], ...],
    ) -> dict[str | None, PairwiseObservations]:
        if support is None:
            return {}
        observations = (support,) if isinstance(support, PairwiseObservations) else support
        if (
            not isinstance(observations, Sequence)
            or not observations
            or any(not isinstance(value, PairwiseObservations) for value in observations)
        ):
            raise ValueError("Support requires observations for each selected species")
        by_species = {value.species: value for value in observations}
        if len(by_species) != len(observations):
            raise ValueError("Support requires one observations object per species")
        if self.distance_model.num_species == 1 and set(by_species) == {None}:
            by_species = {selected[0][0]: observations[0]}
        if set(by_species) != {name for name, _ in selected}:
            raise ValueError("Support observations must match the selected species")
        return by_species
