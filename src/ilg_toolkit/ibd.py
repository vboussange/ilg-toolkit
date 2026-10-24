"""Isolation by distance (IBD) reference relating geographic distances and genetic dissimilarity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .data import TargetSpec


def _nonnegative_array(values: ArrayLike, name: str) -> jax.Array:
    array = jnp.asarray(values)
    if not (jnp.issubdtype(array.dtype, jnp.integer) or jnp.issubdtype(array.dtype, jnp.floating)):
        raise ValueError(f"{name} must contain real numeric values")
    if not bool(jnp.isfinite(array).all()) or bool((array < 0).any()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array.astype(jnp.float64)


def _validate_metadata(distance_units: str, target: TargetSpec) -> None:
    if not isinstance(distance_units, str) or not distance_units.strip():
        raise ValueError("distance_units must explicitly name nonempty geographic units")
    if (
        not isinstance(target, TargetSpec)
        or target.kind != "dissimilarity"
        or target.transform != "identity"
    ):
        raise ValueError(
            "IBD requires an original-scale dissimilarity TargetSpec (identity transform)"
        )


@final
@dataclass(frozen=True)
class IBDModel:
    """An isolation-by-distance model predicting original-scale genetic dissimilarity.

    Geographic distances must be Euclidean separations in ``distance_units``;
    convert raster positions to those units before fitting or prediction.
    Fitting compares genetic predictions and observations using log1p.
    Enable JAX float64 before creating inputs for fitting or prediction.
    """

    slope: float
    intercept: float
    distance_units: str
    target: TargetSpec

    def __post_init__(self) -> None:
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not isfinite(value)
            or value < 0
            for value in (self.slope, self.intercept)
        ):
            raise ValueError("IBD coefficients must be finite and nonnegative real numbers")
        _validate_metadata(self.distance_units, self.target)

    def predict(self, distances: ArrayLike, *, distance_units: str) -> jax.Array:
        """Genetic predictions as a vector or square matrix, with a structural zero diagonal."""
        if distance_units != self.distance_units:
            raise ValueError("Prediction distance_units must match the fitted model")
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("IBD prediction requires JAX float64; enable x64 explicitly")
        array = _nonnegative_array(distances, "Prediction distances")
        if (
            not array.size
            or array.ndim not in {1, 2}
            or (array.ndim == 2 and array.shape[0] != array.shape[1])
        ):
            raise ValueError("Prediction distances must be a nonempty vector or square matrix")
        values = self.slope * array + self.intercept
        if values.ndim == 2:
            diagonal = jnp.diag_indices(values.shape[0])
            values = values.at[diagonal].set(0)
        if not bool(jnp.all(jnp.isfinite(values))):
            raise FloatingPointError("IBD prediction produced nonfinite values")
        return values

    @classmethod
    def fit(
        cls,
        distances_by_region: Mapping[str, ArrayLike],
        targets_by_region: Mapping[str, ArrayLike],
        *,
        distance_units: str,
        target: TargetSpec,
        max_iterations: int = 2000,
    ) -> IBDModel:
        """An isolation-by-distance model fitted with equal regional influence."""
        _validate_metadata(distance_units, target)
        if not jax.config.read("jax_enable_x64"):
            raise ValueError("IBD fitting requires JAX float64; enable x64 explicitly")
        if (
            isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            not isinstance(distances_by_region, Mapping)
            or not isinstance(targets_by_region, Mapping)
            or not distances_by_region
            or set(distances_by_region) != set(targets_by_region)
        ):
            raise ValueError("Distances and targets require matching nonempty regions")
        if any(not isinstance(name, str) or not name.strip() for name in distances_by_region):
            raise ValueError("Each region name must be a nonempty string")
        vectors = []
        for name in sorted(distances_by_region):
            distances = _nonnegative_array(distances_by_region[name], f"{name}: distances")
            targets = _nonnegative_array(targets_by_region[name], f"{name}: targets")
            if distances.ndim != 1 or targets.ndim != 1 or not distances.size:
                raise ValueError(f"{name}: distances and targets must be nonempty vectors")
            if distances.shape != targets.shape:
                raise ValueError(f"{name}: distance and target vectors must be aligned")
            vectors.append((distances, targets))

        @jax.jit
        @jax.value_and_grad
        def loss(parameters):
            slope, intercept = parameters
            losses = [
                jnp.mean((jnp.log1p(slope * distances + intercept) - jnp.log1p(targets)) ** 2)
                for distances, targets in vectors
            ]
            return jnp.mean(jnp.stack(losses))

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            value, gradient = loss(jnp.asarray(parameters))
            return float(value), np.asarray(gradient)

        result = minimize(
            objective,
            np.zeros(2),
            method="L-BFGS-B",
            jac=True,
            bounds=((0.0, None), (0.0, None)),
            options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": 1e-10},
        )
        if (
            not result.success
            or not np.isfinite(result.x).all()
            or not np.isfinite(result.fun)
            or not np.isfinite(result.jac).all()
        ):
            raise RuntimeError(f"IBD calibration did not converge: {result.message}")
        return cls(float(result.x[0]), float(result.x[1]), distance_units, target)
