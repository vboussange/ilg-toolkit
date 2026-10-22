"""Scientific behavior of models.common."""

import jax.numpy as jnp
import numpy as np


def test_encoder_distance_boundary_accepts_array_like_rasters_and_nodes(ScalarEmbedding):
    encoder = ScalarEmbedding(jnp.asarray(1.0))
    distances = encoder.predict_distances([[[0.0], [1.0], [2.0]]], [0, 1, 2])
    np.testing.assert_array_equal(distances, [[0, 1, 4], [1, 0, 1], [4, 1, 0]])
