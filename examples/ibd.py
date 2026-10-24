"""An isolation-by-distance (IBD) reference fitted to geographic separation.

Run with JAX_ENABLE_X64=true to retain double-precision fitted coefficients.
"""

import jax.numpy as jnp

from ilg_toolkit import IBDModel, TargetSpec

# Row and column pixel sizes convert grid positions to distances in meters.
pixel_size_meters = jnp.array([20.0, 10.0])
positions = {
    "headwaters": jnp.array([[0, 0], [3, 0], [0, 4]]),
    "lowlands": jnp.array([[0, 0], [1, 2], [4, 0], [3, 4]]),
}
distance_matrices = {}
for name, grid_positions in positions.items():
    points_meters = grid_positions * pixel_size_meters
    displacement = points_meters[:, None, :] - points_meters[None, :, :]
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
print("Each region has equal weight in the fit.")
print("Original-scale genetic predictions (structural zero diagonal):")
print(prediction)
