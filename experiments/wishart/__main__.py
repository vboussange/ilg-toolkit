"""Run from the checkout: JAX_ENABLE_X64=true python -m experiments.wishart."""

import coordax as cx
import jax
import numpy as np
from ilg_toolkit.data import RegionBatch
from ilg_toolkit.models import ResNet9Conductance
from scipy.linalg import helmert

from .diagnostic import diagnose_wishart, GaussianMarkerDistances


def main():
    """Runs the Wishart diagnostic on a ResNet9 patch encoder with known parameters."""
    labels = ("west", "east", "north", "south")
    region = RegionBatch(
        "synthetic-valley",
        cx.field(
            np.random.default_rng(4).normal(size=(8, 8, 2)),
            "row",
            "column",
            cx.LabeledAxis("feature", np.array(["elevation", "canopy"])),
        ),
        labels,
        np.array([[0, 0], [0, 7], [7, 0], [7, 7]]),
    )
    patch_encoder = ResNet9Conductance(2, patch_size=4, key=jax.random.key(6))
    resistances = np.array(patch_encoder.predict(region).as_matrix())
    np.fill_diagonal(resistances, 0)
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
        patch_encoder=patch_encoder,
        scale=scale,
        nugget=nugget,
        training_unit_ids=labels[:2],
        anchor_id=labels[0],
        parameter_source="fixed",
    )
    print("Log likelihood:", report.log_likelihood)
    print("Conditional held-out log likelihood:", report.heldout_score.log_likelihood)
    print("Held-out score measure:", report.heldout_score.coordinate_measure)
    print("Training gate:", report.training_gate)
    for reason in report.gate_reasons:
        print(" -", reason)


if __name__ == "__main__":
    main()
