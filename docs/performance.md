# Performance

Requested pairs control prediction output. Patch processing, region sampling,
and species processing control training memory and work. Resistance calculation
and maximum-likelihood population effects (MLPE) have separate numerical costs.

## Requested pairs

`model.predict(region, pairs=[("west", "east")])` returns values for those query
pairs, preserving their order and orientation. Omitting `pairs` requests every
distinct pair among the region's sampling units. A `Prediction` stores one value
per pair; `as_matrix()` allocates a sampling-unit matrix, with `NaN` on the diagonal
and for unrequested pairs. Keep the pair-aligned representation for large queries.

Embedding models compute squared embedding distances for the requested pairs
after constructing the region's embeddings. Conductance models compute the
resistance matrix among the selected species' sampling units and then select the
requested entries. Their resistance work therefore depends on the sampling-unit
count as well as the conductance graph size.

## Memory and training controls

`patch_size` changes the spatial resolution of the distance model. In contrast,
`patch_microbatch_size` changes how many raster patches are processed together,
while preserving the model's predictions and gradients. Set this field when
constructing `LogLinearConductance`, `ResNet9Conductance`, or `UNet`.

`TrainingConfig.region_batch_size` samples regions uniformly without replacement
for each optimizer update, up to the number available. `None` uses all regions.
Each selected group uses all its training pairs. Full evaluation uses all training
and validation groups at step zero, every `eval_every` updates, and the final step.
Increase `eval_every` to reduce evaluation frequency within a long fit.

For experimental multispecies training, `species_batch_size` samples species
within each selected region. Sampling weights preserve equal species influence
and equal region influence within each species. `species_microbatch_size` sets
how many selected species are processed together before their gradients form one
update. It changes memory use while retaining the objective. Both default to
`None`.

## Resistance calculation

Conductance models use JAXScape with conjugate-gradient solves in float64. Enable
`JAX_ENABLE_X64=true` before constructing inputs. `ResistanceSolverConfig` sets
relative tolerance (`rtol`), absolute tolerance (`atol`), and the iteration limit
(`max_steps`). A failed solve raises an error.

Installing the `amg` extra and setting `use_amg=True` enables algebraic multigrid
preconditioning. Preparation uses the conductance graph's shape; resistance
calculations use the model's learned conductances. Solver settings belong to the
distance model, separately from `TrainingConfig`.

## MLPE calculations

For $m$ observed pairs and $n$ sampling units, let $Z$ have two ones per row,
identifying the populations in each pair. With population variance $u$ and
residual variance plus jitter $d=e+j$,

$$V=dI+uZZ^\top,\qquad G=Z^\top Z,\qquad K=I+(u/d)G.$$

Calculations accumulate $G$ by population and use
$\log\det V=m\log d+\log\det K$. For any columns $B$, define
$A=(u/d)K^{-1}Z^\top B$ and $R=B-ZA$ using linear solves. Then

$$B^\top V^{-1}B=R^\top R/d+A^\top A/u.$$

This positive representation avoids subtracting nearly equal large quantities.
It provides the generalized least-squares coefficients, residual quadratic, and
population-effect posterior. Storage is $O(n^2+m)$ and factorization costs
$O(n^3)$. For all distinct pairs, $m=n(n-1)/2$; the landscape model and resistance
calculation may dominate the overall fit. The [statistics guide](statistics.md)
defines the likelihood and calibration procedures.

## MLPE precision limits

Standalone calibration and conditional prediction use float64 within their calls.
Joint MLPE training requires float64 to be enabled beforehand. The supported
roundoff bound is

$$\epsilon_{\rm machine}\left(1+2\max_i\deg(i)\frac{u}{e+j}\right)\le0.01,$$

where $\deg(i)$ counts observed pairs involving population $i$. This bound covers
incomplete pair sets. Unsupported variance ratios, failed factorizations,
nonfinite distances, and unidentified calibration fits raise errors. Jitter is
set by `MLPEConfig.jitter` and remains fixed throughout fitting and prediction.
Variance fitting requires at least three observed pairs, shared populations,
and a nonsingular distance design.
