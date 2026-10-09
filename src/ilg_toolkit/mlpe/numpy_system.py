"""Float64 endpoint-system operations for the standalone host optimizer."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve

from .system import MAX_ROUNDOFF_BOUND


def endpoint_gram(left, right, n_populations):
    """Count endpoint intersections in n*n space, including incomplete pair sets."""
    gram = np.zeros((n_populations, n_populations), dtype=np.float64)
    np.add.at(gram, (left, left), 1)
    np.add.at(gram, (right, right), 1)
    np.add.at(gram, (left, right), 1)
    np.add.at(gram, (right, left), 1)
    return gram


@dataclass
class NumpyPopulationSystem:
    """A single exact covariance factorization at declared variance components."""

    left: np.ndarray
    right: np.ndarray
    gram: np.ndarray
    unit: float
    residual: float

    def __post_init__(self):
        self.ratio = self.unit / self.residual
        bound = np.finfo(np.float64).eps * (1 + 2 * self.ratio * np.diag(self.gram).max())
        if not np.isfinite(bound) or bound > MAX_ROUNDOFF_BOUND:
            raise np.linalg.LinAlgError(
                "MLPE variance ratio exceeds the float64 endpoint-system numerical bound "
                "eps*(1+2*max_degree*unit/(residual+jitter)) <= .01; "
                "declare appropriate variance constraints rather than increasing jitter silently"
            )
        self.factor = np.linalg.cholesky(np.eye(len(self.gram)) + self.ratio * self.gram)
        self.logdet = (
            len(self.left) * np.log(self.residual) + 2 * np.log(np.diag(self.factor)).sum()
        )

    def residual_components(self, values):
        sums = np.zeros((len(self.gram), values.shape[1]), dtype=np.float64)
        np.add.at(sums, self.left, values)
        np.add.at(sums, self.right, values)
        effects = self.ratio * cho_solve((self.factor, True), sums)
        remaining = values - effects[self.left] - effects[self.right]
        return remaining, effects

    def inner_products(self, values):
        remaining, effects = self.residual_components(values)
        inner = remaining.T @ remaining / self.residual + effects.T @ effects / self.unit
        return inner, remaining, effects

    def posterior(self, residual):
        _, effects = self.residual_components(residual[:, None])
        covariance = self.unit * cho_solve((self.factor, True), np.eye(len(self.gram)))
        return effects[:, 0], (covariance + covariance.T) / 2, self.factor / np.sqrt(self.unit)
