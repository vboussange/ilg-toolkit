"""Run from the checkout: JAX_ENABLE_X64=true python -m experiments.wishart."""

import coordax as cx
import jax
import numpy as np
from scipy.linalg import helmert

from ilg_toolkit.data import RegionBatch, SamplingUnits
from ilg_toolkit.models import ResNet9Conductance

from .diagnostic import GaussianMarkerDistances, diagnose_wishart


def main():
    """Demonstrate known-parameter Gaussian-marker diagnostics through a real graph."""
    labels = ("west", "east", "north", "south")
    region = RegionBatch(
        "synthetic-valley",
        cx.Field(
            np.random.default_rng(4).normal(size=(8, 8, 2)), dims=("row", "column", "feature")
        ),
        {"fox": SamplingUnits(labels, np.array([[0, 0], [0, 7], [7, 0], [7, 7]]))},
        feature_names=("elevation", "habitat"),
    )
    encoder = ResNet9Conductance(2, species=("fox",), patch_size=4, key=jax.random.key(6))
    resistances = np.asarray(encoder.ecological_distances(region, species="fox")["fox"])
    basis = helmert(len(labels))
    scale, nugget, marker_count = 1.3, 0.2, 20
    covariance = -0.5 * scale * basis @ resistances @ basis.T + nugget * np.eye(3)
    markers = (
        basis.T
        @ np.linalg.cholesky(covariance)
        @ np.random.default_rng(7).normal(size=(3, marker_count))
    )
    distances = np.mean((markers[:, None] - markers[None, :]) ** 2, axis=-1)
    observations = GaussianMarkerDistances(
        region.name, labels, distances, marker_count, "average", "squared_gaussian_marker_distance"
    )
    report = diagnose_wishart(
        region,
        observations,
        encoder=encoder,
        species="fox",
        scale=scale,
        nugget=nugget,
        training_unit_ids=labels[:2],
        anchor_id=labels[0],
        parameter_source="fixed",
    )
    print("Reference log likelihood:", report.log_likelihood)
    print("Conditional held-out log likelihood:", report.heldout_score.log_likelihood)
    print("Held-out score measure:", report.heldout_score.coordinate_measure)
    print("Experimental training:", report.training_gate)
    for reason in report.gate_reasons:
        print(" -", reason)


if __name__ == "__main__":
    main()
