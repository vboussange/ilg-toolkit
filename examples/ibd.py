"""Fit a geographic reference without landscape encoders or paper data.

Run with JAX_ENABLE_X64=true to retain double-precision fitted coefficients.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import IBDModel, TargetSpec

jax.config.update("jax_enable_x64", True)

# Raster positions alone have no physical units. Declare row/column pixel sizes
# explicitly before computing Euclidean separation in metres.
pixel_size_metres = jnp.array([20.0, 10.0])
positions = {
    "headwaters": jnp.array([[0, 0], [3, 0], [0, 4]]),
    "lowlands": jnp.array([[0, 0], [1, 2], [4, 0], [3, 4]]),
}
distance_matrices = {}
for name, grid_positions in positions.items():
    points_metres = grid_positions * pixel_size_metres
    displacement = points_metres[:, None, :] - points_metres[None, :, :]
    distance_matrices[name] = jnp.linalg.norm(displacement, axis=-1)

distances = {
    name: matrix[jnp.triu_indices(len(matrix), k=1)] for name, matrix in distance_matrices.items()
}
targets = {name: 0.003 * values + 0.1 for name, values in distances.items()}
model = IBDModel.fit(
    distances,
    targets,
    distance_units="m",
    target=TargetSpec("synthetic genetic divergence", units="index"),
)
prediction = model.predict(distance_matrices["lowlands"], distance_units="m")
print(f"Fitted slope: {model.slope:.6f} index/m; intercept: {model.intercept:.6f} index")
print("Each region has equal weight, despite its different number of observed pairs.")
print("Original-scale genetic predictions (structural zero diagonal):")
print(prediction)

# This is a converged numerical optimization, not an exact algebraic solve.
np.testing.assert_allclose([model.slope, model.intercept], [0.003, 0.1], rtol=1e-7, atol=1e-9)
