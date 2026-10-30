"""Landscape genetics models composing a distance model and a calibration head."""

from collections.abc import Mapping, Sequence
from typing import final, overload

import equinox as eqx
import jax.numpy as jnp

from ._checks import _require_float64
from .calibration import AbstractCalibrationHead, MLPECalibrationHead
from .data import _feature_names, PairwiseObservations, RegionBatch, TargetSpec
from .models import AbstractDistanceModel
from .models.base import (
    _model_species,
    _query_results,
    _resolve_query,
    _species_distances,
    _SpeciesQuery,
)
from .prediction import Prediction


@final
class LandscapeGeneticsModel[DistanceModel: AbstractDistanceModel](eqx.Module):
    """Landscape genetics model: a distance model composed with a calibration head.

    The distance model maps landscape features to ecological distances, and the
    calibration head maps ecological distances to genetic predictions on the
    original scale. Species names come from the distance model, and the target
    contract from the calibration head. The class is generic in its distance model
    type, so ``model.distance_model`` keeps the type of the distance model it holds.

    Args:
        distance_model: Distance model giving ecological distances.
        calibration_head: Calibration head from training or
            [`fit_calibration_head`][ilg_toolkit.fit_calibration_head].
        feature_names: Feature names of the training regions, which query regions
            match in meaning and order.

    Raises:
        TypeError: If ``distance_model`` or ``calibration_head`` has an unsupported
            type.
        ValueError: If the feature names are empty or repeat a name, or an MLPE
            calibration head holds species that the distance model lacks.
    """

    distance_model: DistanceModel
    calibration_head: AbstractCalibrationHead
    feature_names: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.distance_model, AbstractDistanceModel):
            raise TypeError("distance_model must implement AbstractDistanceModel")
        if not isinstance(self.calibration_head, AbstractCalibrationHead):
            raise TypeError("calibration_head must implement AbstractCalibrationHead")
        if isinstance(self.calibration_head, MLPECalibrationHead):
            fitted = {species for _, species in self.calibration_head.calibrations}
            if not fitted <= set(_model_species(self.distance_model)):
                raise ValueError(
                    f"MLPE calibration species {sorted(fitted, key=str)!r} must be species of "
                    f"the distance model, {self.distance_model.species!r}"
                )
        self.feature_names = _feature_names(self.feature_names)

    @property
    def species(self) -> tuple[str, ...] | None:
        """Species names of the distance model, or ``None`` for one unnamed species."""
        return self.distance_model.species

    @property
    def target(self) -> TargetSpec:
        """Target contract of the calibration head."""
        return self.calibration_head.target

    def _validate_region(self, region: RegionBatch):
        """Raises a ``ValueError`` if a region's feature names differ from ``feature_names``."""
        if region.feature_names != self.feature_names:
            raise ValueError(
                f"Query features {region.feature_names} must match the training features "
                f"{self.feature_names} in meaning and order"
            )

    @overload
    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None],
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> dict[str, Prediction]: ...

    @overload
    def predict(
        self,
        region: RegionBatch,
        query: Sequence[tuple[str, str]] | None = None,
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction: ...

    def predict(
        self,
        region: RegionBatch,
        query: Mapping[str, Sequence[tuple[str, str]] | None]
        | Sequence[tuple[str, str]]
        | None = None,
        *,
        known_effects: bool = False,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None = None,
    ) -> Prediction | dict[str, Prediction]:
        """Returns genetic predictions on the original scale for query pairs.

        The distance model computes ecological distances, and the calibration head
        converts them into genetic predictions. ``query`` maps species names of the
        distance model to query pairs, or to ``None`` for every distinct pair, and
        returns predictions by species in query order. A single-species model also
        accepts a sequence of pairs, or ``None`` for every distinct pair, and returns
        one ``Prediction``. Query pairs keep their order and orientation.

        An MLPE calibration head gives marginal predictions by default;
        ``known_effects=True`` and ``support=...`` condition population effects.
        A direct calibration head supports neither conditional mode.

        | Mode | Population effects | Extras |
        | --- | --- | --- |
        | Default | Marginal mean | `model_values` |
        | `known_effects=True` | Saved posterior from calibration | Mean and variances |
        | `support=observations` | Saved posterior updated with support | Mean and variances |

        Point predictions are inverse-transformed model-scale means, and the
        ``model_variance`` and ``effect_variance`` extras stay on the model scale.

        Args:
            region: Region batch with the landscape features and the query pairs'
                sampling units.
            query: Mapping from species name to query pairs, or to ``None`` for
                every distinct pair; for a single-species model, a sequence of query
                pairs, or ``None`` for every distinct pair.
            known_effects: Whether to include the saved population-effect posterior
                of the MLPE calibration.
            support: Support observations, one per queried species, that update the
                saved population-effect posterior for this call. Support requires
                explicit query pairs for each queried species; support pairs are
                disjoint from the calibration pairs and from the query pairs,
                including reversals; shared sampling units are allowed.

        Returns:
            Genetic predictions by species for a mapping query, or one
            ``Prediction`` for a sequence of pairs or ``None``.

        Raises:
            ValueError: If the region's feature names differ from ``feature_names``,
                the query names an unknown species, a multispecies model receives a
                sequence of pairs or ``None``, the query pairs are empty, repeat a
                pair or name an unknown sampling unit, ``support`` does not match the
                queried species or comes without explicit query pairs, or the
                calibration head rejects the conditioning mode or the region.
            MLPEError: If ``known_effects=True`` is combined with ``support``, or the
                support observations or their pairs are invalid.
            ImportError: If the distance model's ``solver_config`` selects ``"cudss"``
                without spineax.
            RuntimeError: If float64 is disabled, or the distance model's
                ``solver_config`` selects ``"cudss"`` without a CUDA-enabled GPU backend.

        Examples:
            Prepare one landscape with six located populations and fit an MLPE
            calibration head from measurements among its first four populations:

                import coordax as cx
                import jax
                import jax.numpy as jnp
                import numpy as np

                jax.config.update("jax_enable_x64", True)
                from ilg_toolkit import (
                    LandscapeGeneticsModel, MLPELoss, PairwiseObservations, RegionBatch,
                    TargetSpec, fit_calibration_head,
                )
                from ilg_toolkit.models import LogLinearConductance

                features = cx.field(jnp.arange(6.0).reshape(1, 6, 1), "row", "column",
                                    cx.LabeledAxis("feature", np.array(["elevation"])))
                region = RegionBatch("valley", features, tuple(f"p{i}" for i in range(6)),
                                     [[0, i] for i in range(6)])
                observations = PairwiseObservations.from_pairs(
                    [("p0", "p1"), ("p0", "p2"), ("p0", "p3"),
                     ("p1", "p2"), ("p1", "p3"), ("p2", "p3")],
                    [1.2, 1.6, 2.9, 1.7, 2.0, 1.3],
                    target=TargetSpec("genetic divergence", units="index"),
                )
                distance_model = LogLinearConductance(1, patch_size=1)
                head = fit_calibration_head(region, observations, distance_model,
                                            loss=MLPELoss())
                model = LandscapeGeneticsModel(distance_model, head, region.feature_names)
                query_pairs = [("p4", "p2"), ("p5", "p3")]

            Marginal predictions use the fitted intercept, slope and distance
            standardization, with population effects averaged out:

                marginal = model.predict(region, query_pairs)
                marginal.values                 # Genetic means in original units.
                marginal.extras["model_values"] # Means before the inverse target transform.
                marginal.pairs                  # Same alignment as query_pairs.

            Saved-effect predictions add the posterior population effects estimated
            from the calibration observations. Populations without calibration
            observations keep a zero prior mean and the prior population variance;
            query pairs may overlap calibration pairs:

                saved = model.predict(region, query_pairs, known_effects=True)
                saved.extras["effect_variance"] # Variance of the summed population effects.
                saved.extras["model_variance"]  # Effect, residual and jitter variance.

            Support observations update the saved posterior for this call, including
            populations without calibration observations. The distance model,
            coefficients, distance standardization and variance components stay
            fixed:

                support = PairwiseObservations.from_pairs(
                    [("p4", "p0"), ("p4", "p1")], [2.7, 2.2], target=model.target,
                )
                conditioned = model.predict(region, query_pairs, support=support)
                conditioned.extras["model_variance"]

            ``as_matrix()`` and ``as_matrix("effect_variance")`` arrange the values by
            sampling unit.
        """
        _require_float64()
        self._validate_region(region)
        if region.individual_based and isinstance(self.calibration_head, MLPECalibrationHead):
            raise ValueError("MLPE calibration heads reject individual-based region batches")
        requests, by_species = _resolve_query(self.species, query)
        support_by_species = self._support_observations(support, requests)
        distances_by_species = _species_distances(
            self.distance_model,
            region,
            requests,
            {name: value.observed_pairs for name, value in support_by_species.items()},
        )
        results = {}
        for request in requests:
            prepared, labels, distances = distances_by_species[request.name]
            observations = support_by_species.get(request.name)
            query_count = len(labels) - (
                0 if observations is None else len(observations.observed_pairs)
            )
            result = self.calibration_head.predict(
                distances[:query_count],
                labels[:query_count],
                region_name=region.name,
                species=request.name,
                individual_based=prepared.individual_based,
                known_effects=known_effects,
                support=observations,
                support_distances=None if observations is None else distances[query_count:],
            )
            results[request.name] = Prediction(
                jnp.asarray(result.values),
                labels[:query_count],
                region.name,
                prepared.sampling_unit_ids,
                species=request.name,
                extras=result.extras,
            )
        return _query_results(results, by_species)

    def _support_observations(
        self,
        support: PairwiseObservations | Sequence[PairwiseObservations] | None,
        requests: tuple[_SpeciesQuery, ...],
    ) -> dict[str | None, PairwiseObservations]:
        if support is None:
            return {}
        if any(request.pairs is None for request in requests):
            raise ValueError("Support requires explicit query pairs for each queried species")
        observations = (support,) if isinstance(support, PairwiseObservations) else support
        if (
            not isinstance(observations, Sequence)
            or not observations
            or any(not isinstance(value, PairwiseObservations) for value in observations)
        ):
            raise ValueError("Support requires observations for each queried species")
        by_species = {value.species: value for value in observations}
        if len(by_species) != len(observations):
            raise ValueError("Support requires one observations object per species")
        if len(_model_species(self.distance_model)) == 1 and set(by_species) == {None}:
            by_species = {requests[0].name: observations[0]}
        if set(by_species) != {request.name for request in requests}:
            raise ValueError("Support observations must match the queried species")
        return by_species
