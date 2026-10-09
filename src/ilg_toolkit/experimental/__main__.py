"""Run the strict synthetic reference: JAX_ENABLE_X64=true python -m ilg_toolkit.experimental."""

import jax
import numpy as np

from ..data import PreparedRegion
from ..models import ResNet9Conductance
from .wishart import GaussianMarkerDistances, _helmert, diagnose_wishart


def main():
    """Demonstrate known-parameter Gaussian-marker diagnostics through a real graph."""
    labels = ("west", "east", "north", "south")
    region = PreparedRegion(
        "synthetic-valley",
        np.random.default_rng(4).normal(size=(8, 8, 2)),
        labels,
        np.array([[0, 0], [0, 7], [7, 0], [7, 7]]),
    )
    encoder = ResNet9Conductance(2, patch_size=4, key=jax.random.key(6))
    scores = np.asarray(encoder.predict_distances(region.features, region.pixel_nodes))
    basis = _helmert(len(labels))
    scale, nugget, marker_count = 1.3, 0.2, 20
    covariance = -0.5 * scale * basis @ scores @ basis.T + nugget * np.eye(3)
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
