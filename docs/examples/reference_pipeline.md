# Reference pipeline

[View `reference_pipeline.ipynb` on GitHub](https://github.com/vboussange/ilg-toolkit/blob/main/examples/reference_pipeline.ipynb)

This notebook is the reference pipeline for inverse landscape genetics (ILG) with ILG Toolkit.
It relates pairwise Fst between 12 Mountain Pygmy-possum (*Burramys parvus*) sampling units in
the Victorian Alps to land cover. Three population-holdout folds each withhold four sampling
units. In each fold, an isolation-by-distance (IBD) model and a log-linear patch encoder trained with
the maximum-likelihood population effects (MLPE) loss, each with an MLPE calibration head,
predict the pairs that involve the withheld sampling units. The three landscape genetics models
with trained patch encoders, one per fold, form an ensemble, which summarizes the learned
resistance surface and the land-cover effects.

Genetic data: pairwise Fst kindly provided by Cesar Australia. Land cover: ESA WorldCover 2021
(CC BY 4.0).


```python
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import rioxarray
import xarray as xr
from ilg_toolkit import (
    Ensemble,
    fit_calibration_head,
    LandscapeGeneticsModel,
    load_model,
    MLPELoss,
    PairwiseObservations,
    RegionBatch,
    save_model,
    TargetSpec,
    train,
    TrainingConfig,
)
from ilg_toolkit.models import IBDModel, initialize_uniform_conductance, LogLinearConductance
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# Effective resistance needs float64: enable x64 before creating any inputs.
jax.config.update("jax_enable_x64", True)
print(jax.devices())
```

    [CudaDevice(id=0)]


## Land cover and sampling units

The example data hold the ESA WorldCover 2021 land-cover raster, the projected coordinates of
the sampling units and their pairwise Fst. Land-cover pixels of 8.36 m are averaged into cells
of about 84 m, giving the fraction of each class in each cell. The fractions of grassland,
permanent water and the remaining non-tree classes are the landscape features; tree cover is
the reference class. Each vertex of the conductance grid graph covers 8 × 8 cells, about 669 m.


```python
DATA_URL = (
    "https://github.com/vboussange/ilg-toolkit/releases/download/"
    "example-data-v1/mountain-pygmy-possum.zip"
)
CACHE = Path("data") / "example-data-v1"
COARSENING = 10  # Averages 8.36 m land-cover pixels into cells of about 84 m.
PATCH_SIZE = 8  # Groups 8x8 cells into one vertex of about 669 m.
# ESA WorldCover classes merged into landscape features; tree cover is the reference.
FEATURE_CLASSES = {
    "grassland": [30],
    "permanent water": [80],
    "other": [20, 40, 50, 60, 90],  # Shrubland, cropland, built-up, bare, herbaceous wetland.
}

data = CACHE / "mountain-pygmy-possum"
if not data.is_dir():
    CACHE.mkdir(parents=True, exist_ok=True)
    archive, _ = urllib.request.urlretrieve(DATA_URL, CACHE / "mountain-pygmy-possum.zip")
    with zipfile.ZipFile(archive) as contents:
        contents.extractall(CACHE)
    Path(archive).unlink()
landcover = rioxarray.open_rasterio(data / "landcover.tif").squeeze("band", drop=True).load()
sites = pd.read_csv(data / "sites.csv", index_col="site")
fst = pd.read_csv(data / "fst.csv", index_col="site")

fractions = (
    xr.concat(
        [landcover.isin(codes) for codes in FEATURE_CLASSES.values()],
        dim=pd.Index(list(FEATURE_CLASSES), name="feature"),
    )
    .astype(np.float32)
    .coarsen(y=COARSENING, x=COARSENING, boundary="trim")
    .mean()
)
fractions = fractions.isel(
    y=slice(fractions.sizes["y"] // PATCH_SIZE * PATCH_SIZE),
    x=slice(fractions.sizes["x"] // PATCH_SIZE * PATCH_SIZE),
)
features = fractions.rename(y="row", x="column").transpose("row", "column", "feature")
region = RegionBatch.from_points(
    "victorian-alps",
    cx.from_xarray(features),
    sampling_unit_ids=tuple(sites.index),
    coordinates=sites[["x", "y"]].to_numpy(),
)
print(f"Land cover: {landcover.sizes['y']} x {landcover.sizes['x']} pixels")
print(f"Landscape features: {features.sizes['row']} x {features.sizes['column']} cells")
print(
    f"Conductance grid graph: {features.sizes['row'] // PATCH_SIZE} x "
    f"{features.sizes['column'] // PATCH_SIZE} vertices"
)
```

    Land cover: 4596 x 3052 pixels
    Landscape features: 456 x 304 cells
    Conductance grid graph: 57 x 38 vertices


## Pairwise Fst and population-holdout folds

The target is pairwise Fst, on the model scale of its odds Fst / (1 − Fst), the linearized Fst.
The 12 sampling units are assigned at random to three folds of four. In each fold, two of the
eight remaining sampling units are drawn at random as validation units, and the other six are
training units. Training pairs join two training units, validation pairs join a validation unit
to another sampling unit outside the fold, and test pairs involve at least one sampling unit of
the fold. Calibration pairs, all pairs among the eight sampling units outside the fold, fit the
MLPE calibration heads.


```python
observations = PairwiseObservations(
    tuple(fst.index),
    fst.to_numpy(),
    target=TargetSpec("pairwise Fst", units="proportion", transform="odds"),
)
folds = np.array_split(np.random.default_rng(0).permutation(sites.index), 3)
rng = np.random.default_rng(1)
validation_units, training_sets, validation_sets, calibration_sets, test_sets = [], [], [], [], []
for held_out in folds:
    calibration_ids = sites.index.difference(held_out)
    validation_ids = rng.choice(calibration_ids, 2, replace=False)
    training_ids = calibration_ids.difference(validation_ids)
    calibration = observations.select_pairs(list(combinations(calibration_ids, 2)))
    validation_units.append(validation_ids)
    training_sets.append(observations.select_pairs(list(combinations(training_ids, 2))))
    validation_sets.append(
        calibration.select_pairs(
            [pair for pair in calibration.observed_pairs if set(pair) & set(validation_ids)]
        )
    )
    calibration_sets.append(calibration)
    test_sets.append(
        observations.select_pairs(
            [pair for pair in observations.observed_pairs if set(pair) & set(held_out)]
        )
    )

fst.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Basalt Hill</th>
      <th>Bundara Scree</th>
      <th>Higginbotham East</th>
      <th>Higginbotham West</th>
      <th>Little Higginbotham</th>
      <th>Mt Bogong</th>
      <th>Mt Loch East</th>
      <th>Mt Loch North</th>
      <th>Mt McKay</th>
      <th>Pretty Valley West</th>
      <th>Spargo</th>
      <th>Timms Spur</th>
    </tr>
    <tr>
      <th>site</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Basalt Hill</th>
      <td>0.000</td>
      <td>0.176</td>
      <td>0.186</td>
      <td>0.187</td>
      <td>0.189</td>
      <td>0.216</td>
      <td>0.169</td>
      <td>0.173</td>
      <td>0.169</td>
      <td>0.186</td>
      <td>0.195</td>
      <td>0.204</td>
    </tr>
    <tr>
      <th>Bundara Scree</th>
      <td>0.176</td>
      <td>0.000</td>
      <td>0.068</td>
      <td>0.067</td>
      <td>0.065</td>
      <td>0.104</td>
      <td>0.047</td>
      <td>0.046</td>
      <td>0.054</td>
      <td>0.059</td>
      <td>0.046</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>Higginbotham East</th>
      <td>0.186</td>
      <td>0.068</td>
      <td>0.000</td>
      <td>0.009</td>
      <td>0.044</td>
      <td>0.120</td>
      <td>0.043</td>
      <td>0.046</td>
      <td>0.070</td>
      <td>0.075</td>
      <td>0.028</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>Higginbotham West</th>
      <td>0.187</td>
      <td>0.067</td>
      <td>0.009</td>
      <td>0.000</td>
      <td>0.043</td>
      <td>0.118</td>
      <td>0.043</td>
      <td>0.045</td>
      <td>0.066</td>
      <td>0.073</td>
      <td>0.029</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>Little Higginbotham</th>
      <td>0.189</td>
      <td>0.065</td>
      <td>0.044</td>
      <td>0.043</td>
      <td>0.000</td>
      <td>0.119</td>
      <td>0.041</td>
      <td>0.042</td>
      <td>0.068</td>
      <td>0.071</td>
      <td>0.021</td>
      <td>0.116</td>
    </tr>
    <tr>
      <th>Mt Bogong</th>
      <td>0.216</td>
      <td>0.104</td>
      <td>0.120</td>
      <td>0.118</td>
      <td>0.119</td>
      <td>0.000</td>
      <td>0.100</td>
      <td>0.099</td>
      <td>0.088</td>
      <td>0.102</td>
      <td>0.095</td>
      <td>0.043</td>
    </tr>
    <tr>
      <th>Mt Loch East</th>
      <td>0.169</td>
      <td>0.047</td>
      <td>0.043</td>
      <td>0.043</td>
      <td>0.041</td>
      <td>0.100</td>
      <td>0.000</td>
      <td>0.010</td>
      <td>0.052</td>
      <td>0.055</td>
      <td>0.014</td>
      <td>0.099</td>
    </tr>
    <tr>
      <th>Mt Loch North</th>
      <td>0.173</td>
      <td>0.046</td>
      <td>0.046</td>
      <td>0.045</td>
      <td>0.042</td>
      <td>0.099</td>
      <td>0.010</td>
      <td>0.000</td>
      <td>0.052</td>
      <td>0.054</td>
      <td>0.016</td>
      <td>0.098</td>
    </tr>
    <tr>
      <th>Mt McKay</th>
      <td>0.169</td>
      <td>0.054</td>
      <td>0.070</td>
      <td>0.066</td>
      <td>0.068</td>
      <td>0.088</td>
      <td>0.052</td>
      <td>0.052</td>
      <td>0.000</td>
      <td>0.046</td>
      <td>0.049</td>
      <td>0.092</td>
    </tr>
    <tr>
      <th>Pretty Valley West</th>
      <td>0.186</td>
      <td>0.059</td>
      <td>0.075</td>
      <td>0.073</td>
      <td>0.071</td>
      <td>0.102</td>
      <td>0.055</td>
      <td>0.054</td>
      <td>0.046</td>
      <td>0.000</td>
      <td>0.052</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>Spargo</th>
      <td>0.195</td>
      <td>0.046</td>
      <td>0.028</td>
      <td>0.029</td>
      <td>0.021</td>
      <td>0.095</td>
      <td>0.014</td>
      <td>0.016</td>
      <td>0.049</td>
      <td>0.052</td>
      <td>0.000</td>
      <td>0.099</td>
    </tr>
    <tr>
      <th>Timms Spur</th>
      <td>0.204</td>
      <td>0.111</td>
      <td>0.118</td>
      <td>0.115</td>
      <td>0.116</td>
      <td>0.043</td>
      <td>0.099</td>
      <td>0.098</td>
      <td>0.092</td>
      <td>0.105</td>
      <td>0.099</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(
    {
        "held-out sampling units": [", ".join(sorted(held_out)) for held_out in folds],
        "validation units": [", ".join(sorted(ids)) for ids in validation_units],
        "training pairs": [len(pairs.observed_pairs) for pairs in training_sets],
        "validation pairs": [len(pairs.observed_pairs) for pairs in validation_sets],
        "calibration pairs": [len(pairs.observed_pairs) for pairs in calibration_sets],
        "test pairs": [len(pairs.observed_pairs) for pairs in test_sets],
    },
    index=pd.RangeIndex(1, len(folds) + 1, name="fold"),
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>held-out sampling units</th>
      <th>validation units</th>
      <th>training pairs</th>
      <th>validation pairs</th>
      <th>calibration pairs</th>
      <th>test pairs</th>
    </tr>
    <tr>
      <th>fold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Higginbotham East, Little Higginbotham, Mt Loc...</td>
      <td>Mt Bogong, Mt Loch East</td>
      <td>15</td>
      <td>13</td>
      <td>28</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Basalt Hill, Higginbotham West, Mt Bogong, Tim...</td>
      <td>Bundara Scree, Pretty Valley West</td>
      <td>15</td>
      <td>13</td>
      <td>28</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bundara Scree, Mt Loch East, Mt McKay, Spargo</td>
      <td>Mt Loch North, Timms Spur</td>
      <td>15</td>
      <td>13</td>
      <td>28</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>




```python
classes = xr.zeros_like(landcover, dtype=np.uint8)
for index, codes in enumerate(FEATURE_CLASSES.values(), start=1):
    classes = classes.where(~landcover.isin(codes), index)
CLASS_COLORS = ["#006400", "#ffff4c", "#0064c8", "#b4b4b4"]
CLASS_NAMES = ["tree cover", *FEATURE_CLASSES]
FOLD_COLORS = ["#e41a1c", "#984ea3", "#ff7f00"]


def kilometers(axis):
    axis.xaxis.set_major_formatter(lambda value, _: f"{value / 1e3:.0f}")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value / 1e3:.0f}")
    axis.set(xlabel="Easting (km)", ylabel="Northing (km)", aspect="equal")


figure, axis = plt.subplots(figsize=(5.5, 7), layout="constrained")
classes.plot.imshow(
    ax=axis, cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5, add_colorbar=False
)
for fold, (held_out, color) in enumerate(zip(folds, FOLD_COLORS, strict=True), start=1):
    axis.scatter(
        *sites.loc[held_out, ["x", "y"]].T.to_numpy(),
        s=40,
        c=color,
        edgecolors="black",
        label=f"fold {fold}",
    )
handles = [Patch(color=color, label=name) for color, name in zip(CLASS_COLORS, CLASS_NAMES)]
axis.legend(handles=[*handles, *axis.get_legend_handles_labels()[0]], loc="upper left", fontsize=8)
axis.set_title("ESA WorldCover 2021 land cover and sampling units")
kilometers(axis)
plt.show()
```


    
![png](reference_pipeline_files/reference_pipeline_7_0.png)
    


## IBD model

The IBD model's ecological distance is an affine function of the straight-line distance between
the sampling units' vertices, on the same vertices as the patch encoder. Each fold's IBD model
keeps its initial slope and intercept, and `fit_calibration_head` fits its MLPE calibration head
on the calibration pairs.


```python
loss = MLPELoss()
ibd_models = []
for calibration in calibration_sets:
    ibd = IBDModel(patch_size=PATCH_SIZE)
    calibration_head = fit_calibration_head(region, calibration, ibd, loss=loss)
    ibd_models.append(LandscapeGeneticsModel(ibd, calibration_head, region.feature_names))

pd.DataFrame(
    [
        {
            "MLPE slope": mlpe_calibration.slope,
            "population variance": mlpe_calibration.unit_variance,
            "residual variance": mlpe_calibration.residual_variance,
            "log likelihood": mlpe_calibration.ml_log_likelihood,
        }
        for mlpe_calibration in (
            model.calibration_head.for_region(region.name) for model in ibd_models
        )
    ],
    index=pd.RangeIndex(1, len(folds) + 1, name="fold"),
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MLPE slope</th>
      <th>population variance</th>
      <th>residual variance</th>
      <th>log likelihood</th>
    </tr>
    <tr>
      <th>fold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.025166</td>
      <td>0.002693</td>
      <td>0.000233</td>
      <td>59.935815</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.012491</td>
      <td>0.000077</td>
      <td>0.000013</td>
      <td>102.654083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.030275</td>
      <td>0.002447</td>
      <td>0.000226</td>
      <td>60.643914</td>
    </tr>
  </tbody>
</table>
</div>



## Log-linear patch encoder

The log-linear patch encoder maps the patch means $x_k$ of the three land-cover fractions to the
conductance $C_k = \min\{e^{\beta^\top x_k + \beta_0} + 10^{-2}, 10^3\}$ of vertex $k$. In each
fold, the patch encoder is initialized at the best uniform conductance and trained on the
training pairs with the MLPE loss and AdamW (learning rate $10^{-2}$, weight decay $10^{-4}$)
for 1,000 updates. Training evaluates the model every 10 updates and selects the update with
the lowest validation loss. The selected patch encoder, frozen, receives an MLPE calibration
head fitted on the calibration pairs.


```python
config = TrainingConfig(
    loss=loss, optimizer=optax.adamw(1e-2, weight_decay=1e-4), steps=1000, eval_every=10
)
states, patch_models = [], []
for training, validation, calibration in zip(
    training_sets, validation_sets, calibration_sets, strict=True
):
    distance_model = initialize_uniform_conductance(
        LogLinearConductance(3, patch_size=PATCH_SIZE), region, training, loss
    )
    state = train(
        region,
        training,
        distance_model=distance_model,
        config=config,
        validation=(region, validation),
    )
    selected = state.selected_model.distance_model
    calibration_head = fit_calibration_head(region, calibration, selected, loss=loss)
    states.append(state)
    patch_models.append(LandscapeGeneticsModel(selected, calibration_head, region.feature_names))

pd.DataFrame(
    {
        "selected update": [state.selected_step for state in states],
        "initial validation loss": [state.history[0].validation_loss for state in states],
        "selected validation loss": [state.selected_loss for state in states],
    },
    index=pd.RangeIndex(1, len(folds) + 1, name="fold"),
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>selected update</th>
      <th>initial validation loss</th>
      <th>selected validation loss</th>
    </tr>
    <tr>
      <th>fold</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>170</td>
      <td>-1.393227</td>
      <td>-1.574191</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>-2.958546</td>
      <td>-3.039169</td>
    </tr>
    <tr>
      <th>3</th>
      <td>200</td>
      <td>-1.408699</td>
      <td>-1.651802</td>
    </tr>
  </tbody>
</table>
</div>




```python
figure, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
for fold, (axis, state) in enumerate(zip(axes, states, strict=True), start=1):
    steps = [record.step for record in state.history]
    axis.plot(steps, [record.training_loss for record in state.history], label="training")
    axis.plot(steps, [record.validation_loss for record in state.history], label="validation")
    axis.axvline(state.selected_step, color="gray", linestyle="--", linewidth=1, label="selected")
    axis.set(title=f"Fold {fold}", xlabel="update")
axes[0].set_ylabel("MLPE loss per pair")
axes[0].legend(fontsize=8)
plt.show()
```


    
![png](reference_pipeline_files/reference_pipeline_12_0.png)
    


## Population-holdout evaluation

Each model gives marginal predictions for the test pairs of its fold on the original Fst scale.
Pearson's $r$, predictive $R^2 = 1 - \sum (y - \hat y)^2 / \sum (y - \bar y)^2$ and the root
mean squared error (RMSE) compare the predictions $\hat y$ with the observed Fst $y$ in each
fold.


```python
rows, predictions = [], []
for fold, test in enumerate(test_sets, start=1):
    observed = np.asarray(test.observed_values)
    for name, model in (
        ("IBD model", ibd_models[fold - 1]),
        ("log-linear patch encoder", patch_models[fold - 1]),
    ):
        predicted = np.asarray(model.predict(region, test.observed_pairs).values)
        predictions.append((fold, name, observed, predicted))
        rows.append(
            {
                "fold": fold,
                "model": name,
                "Pearson's r": np.corrcoef(observed, predicted)[0, 1],
                "predictive R²": 1
                - np.sum((observed - predicted) ** 2) / np.sum((observed - observed.mean()) ** 2),
                "RMSE": np.sqrt(np.mean((observed - predicted) ** 2)),
            }
        )
metrics = pd.DataFrame(rows)
metrics.pivot(index="fold", columns="model").round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Pearson's r</th>
      <th colspan="2" halign="left">predictive R²</th>
      <th colspan="2" halign="left">RMSE</th>
    </tr>
    <tr>
      <th>model</th>
      <th>IBD model</th>
      <th>log-linear patch encoder</th>
      <th>IBD model</th>
      <th>log-linear patch encoder</th>
      <th>IBD model</th>
      <th>log-linear patch encoder</th>
    </tr>
    <tr>
      <th>fold</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.696</td>
      <td>0.723</td>
      <td>0.228</td>
      <td>0.333</td>
      <td>0.042</td>
      <td>0.039</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.300</td>
      <td>0.327</td>
      <td>-0.587</td>
      <td>-0.746</td>
      <td>0.067</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.571</td>
      <td>0.602</td>
      <td>-0.079</td>
      <td>-0.151</td>
      <td>0.046</td>
      <td>0.048</td>
    </tr>
  </tbody>
</table>
</div>




```python
metrics.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Pearson's r</th>
      <th colspan="2" halign="left">predictive R²</th>
      <th colspan="2" halign="left">RMSE</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IBD model</th>
      <td>0.522</td>
      <td>0.203</td>
      <td>-0.146</td>
      <td>0.412</td>
      <td>0.052</td>
      <td>0.013</td>
    </tr>
    <tr>
      <th>log-linear patch encoder</th>
      <td>0.550</td>
      <td>0.203</td>
      <td>-0.188</td>
      <td>0.541</td>
      <td>0.052</td>
      <td>0.016</td>
    </tr>
  </tbody>
</table>
</div>




```python
figure, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True, layout="constrained")
for axis, name in zip(axes, ("IBD model", "log-linear patch encoder"), strict=True):
    for fold, model_name, observed, predicted in predictions:
        if model_name == name:
            axis.scatter(
                observed, predicted, s=18, color=FOLD_COLORS[fold - 1], label=f"fold {fold}"
            )
    axis.axline((0, 0), slope=1, color="gray", linestyle="--", linewidth=1)
    axis.set(title=name, xlabel="observed Fst", aspect="equal")
axes[0].set_ylabel("predicted Fst")
axes[0].legend(fontsize=8)
plt.show()
```


    
![png](reference_pipeline_files/reference_pipeline_16_0.png)
    


## Ensemble and learned resistance surface

The three landscape genetics models with trained patch encoders, one per population-holdout
fold, form an ensemble, whose genetic prediction averages the members on the original Fst
scale. For each member, the $\log_{10}$ resistance relative to tree cover at vertex $k$ is
$\log_{10}(Q_k / Q_\mathrm{tree})$, where $Q_k = 1/C_k$ is the local resistance and
$Q_\mathrm{tree}$ the local resistance of a vertex covered by tree cover. At each vertex, its
median and interquartile range (IQR) across the members summarize the learned resistance
surface.


```python
ensemble = Ensemble(patch_models)
prediction = ensemble.predict(region, observations.observed_pairs)
pd.DataFrame(
    {
        "observed Fst": np.asarray(observations.observed_values),
        "ensemble mean": prediction.mean.values,
    }
    | {
        f"member {index}": member.values for index, member in enumerate(prediction.members, start=1)
    },
    index=pd.MultiIndex.from_tuples(prediction.mean.pairs),
).head().round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>observed Fst</th>
      <th>ensemble mean</th>
      <th>member 1</th>
      <th>member 2</th>
      <th>member 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Basalt Hill</th>
      <th>Bundara Scree</th>
      <td>0.176</td>
      <td>0.089</td>
      <td>0.098</td>
      <td>0.051</td>
      <td>0.118</td>
    </tr>
    <tr>
      <th>Higginbotham East</th>
      <td>0.186</td>
      <td>0.101</td>
      <td>0.114</td>
      <td>0.070</td>
      <td>0.119</td>
    </tr>
    <tr>
      <th>Higginbotham West</th>
      <td>0.187</td>
      <td>0.103</td>
      <td>0.116</td>
      <td>0.071</td>
      <td>0.121</td>
    </tr>
    <tr>
      <th>Little Higginbotham</th>
      <td>0.189</td>
      <td>0.100</td>
      <td>0.113</td>
      <td>0.070</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>Mt Bogong</th>
      <td>0.216</td>
      <td>0.108</td>
      <td>0.120</td>
      <td>0.070</td>
      <td>0.133</td>
    </tr>
  </tbody>
</table>
</div>




```python
tree_cover = jnp.zeros(len(region.feature_names))  # Land-cover fractions of a tree-cover vertex.
relative_log_resistance = xr.concat(
    [
        np.log10(
            float(member.distance_model.conductance(tree_cover, 0))
            / member.distance_model.conductance_surface(region).to_xarray()
        )
        for member in ensemble.members
    ],
    dim="member",
)
median = relative_log_resistance.median("member")
quartiles = relative_log_resistance.quantile([0.25, 0.75], "member")
iqr = quartiles.sel(quantile=0.75) - quartiles.sel(quantile=0.25)

figure, axes = plt.subplots(1, 3, figsize=(14, 6.4), sharex=True, sharey=True, layout="constrained")
classes.plot.imshow(
    ax=axes[0], cmap=ListedColormap(CLASS_COLORS), vmin=-0.5, vmax=3.5, add_colorbar=False
)
axes[0].legend(
    handles=[Patch(color=color, label=name) for color, name in zip(CLASS_COLORS, CLASS_NAMES)],
    loc="upper left",
    fontsize=8,
)
axes[0].set_title("Land cover")
for axis, surface, title, cmap in (
    (axes[1], median, "Median of log$_{10}$ resistance\nrelative to tree cover", "magma"),
    (axes[2], iqr, "IQR of log$_{10}$ resistance\nrelative to tree cover", "viridis"),
):
    image = surface.plot.imshow(
        ax=axis, x="column", y="row", cmap=cmap, cbar_kwargs={"shrink": 0.7, "label": title}
    )
    axis.set_title(title)
    axis.axis(image.get_extent())
for axis in axes:
    axis.scatter(sites["x"], sites["y"], s=25, c="white", edgecolors="black")
    kilometers(axis)
for axis in axes[1:]:
    axis.set_ylabel("")
plt.show()
```


    
![png](reference_pipeline_files/reference_pipeline_19_0.png)
    


## Land-cover effects

`feature_effects` returns each member's log-linear coefficients by feature name. The features
are land-cover fractions with tree cover omitted, so each effect is the log-conductance
difference between a patch covered by that class and a patch covered by tree cover. A negative
effect marks a class with a lower conductance, and a higher resistance, than tree cover.


```python
effects = pd.DataFrame(
    [
        {
            name: value
            for name, (value,) in member.distance_model.feature_effects(
                region.feature_names
            ).items()
        }
        for member in ensemble.members
    ],
    index=pd.RangeIndex(1, len(folds) + 1, name="member"),
)
axis = effects.T.plot.bar(color=FOLD_COLORS, figsize=(6, 3.8), rot=0)
axis.axhline(0, color="black", linewidth=0.8)
axis.set(ylabel="effect on log conductance\nrelative to tree cover", xlabel="")
axis.legend(title="member", fontsize=8)
plt.show()
effects.round(3)
```


    
![png](reference_pipeline_files/reference_pipeline_21_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>grassland</th>
      <th>permanent water</th>
      <th>other</th>
    </tr>
    <tr>
      <th>member</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-0.121</td>
      <td>-1.396</td>
      <td>1.476</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.507</td>
      <td>0.529</td>
      <td>-0.525</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.879</td>
      <td>-1.933</td>
      <td>1.749</td>
    </tr>
  </tbody>
</table>
</div>



## Saving and reloading the members

`save_model` writes each member to one file. `load_model` reads it into a log-linear patch
encoder built with the same settings, and the reloaded members form the same ensemble.


```python
with TemporaryDirectory() as directory:
    paths = [Path(directory) / f"member-{index}.eqx" for index in range(1, len(folds) + 1)]
    for path, member in zip(paths, ensemble.members, strict=True):
        save_model(path, member)
    template = LogLinearConductance(3, patch_size=PATCH_SIZE)
    reloaded = Ensemble([load_model(path, distance_model=template) for path in paths])

print(
    "Members equal:",
    all(
        eqx.tree_equal(member, original)
        for member, original in zip(reloaded.members, ensemble.members, strict=True)
    ),
)
reloaded_prediction = reloaded.predict(region, observations.observed_pairs)
print(
    "Predictions equal:",
    np.allclose(reloaded_prediction.mean.values, prediction.mean.values, rtol=1e-6, atol=0),
)
```

    Members equal: True
    Predictions equal: True

