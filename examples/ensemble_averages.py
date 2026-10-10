"""Known-output checks: average genetic predictions after transforms, not surfaces."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ilg_toolkit import CalibratedModel, Prediction, RegionBatch, TargetSpec, aggregate_predictions
from ilg_toolkit.models import ConductanceModel

# Known calibrated means on a log1p fitted scale: invert EACH before averaging.
target = TargetSpec("synthetic divergence", units="index", transform="log1p")
model_means = (np.log(2), np.log(10))
calibrated = {
    str(i): Prediction(np.array([[0, np.expm1(value)], [np.expm1(value), 0]]), ("a", "b"), target)
    for i, value in enumerate(model_means)
}
aggregate = aggregate_predictions(calibrated)
print("Mean original-scale prediction:", aggregate.values[0, 1], "(expected 5)")
print("Inverse of mean fitted-scale prediction:", np.expm1(np.mean(model_means)))


@final
class ConstantConductance(ConductanceModel):
    level: jax.Array
    patch_size: int = eqx.field(static=True, default=1)

    def conductance(self, features, *, patch_batch_size=None):
        return jnp.ones(features.shape[:2]) * self.level


with jax.enable_x64():
    region = RegionBatch("graph", np.ones((1, 2, 1)), ("a", "b"), np.array([[0, 0], [0, 1]]))
    target = TargetSpec("synthetic resistance dissimilarity", units="index")
    models = {
        str(i): CalibratedModel(ConstantConductance(jnp.asarray(level)), target, 1)
        for i, level in enumerate((1.0, 4.0))
    }
    predictions = {name: model.predict(region) for name, model in models.items()}
    average = aggregate_predictions(predictions)
    descriptive_surface_mean = np.mean(
        [model.conductance_surface(region) for model in models.values()], axis=0
    )
    mean_surface_model = CalibratedModel(
        ConstantConductance(jnp.asarray(descriptive_surface_mean[0, 0])), target, 1
    )
    print("Mean member resistance:", average.values[0, 1], "(expected .625)")
    print("Descriptive mean conductance:", descriptive_surface_mean)
    print(
        "Resistance of mean conductance:",
        mean_surface_model.predict(region).values[0, 1],
        "(expected .4)",
    )
    print("Member spread is descriptive; neither average identifies a unique biological surface.")
