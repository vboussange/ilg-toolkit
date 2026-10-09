# Experimental Wishart diagnostic contract

Status on 2026-10-09: **reviewed no-go for experimental training**. The coordinating
agent independently reviewed this contract, the implementation and its tests,
and checked the primary SciPy Wishart/matrix-normal densities and EEMS source.
That implementation review supports the strict synthetic diagnostic and confirms
the conservative no-go for joint training. It does **not** endorse empirical
genotype validity and is not a reviewed go decision. The diagnostic is available under
`ilg_toolkit.experimental`; no Wishart optimizer or stable objective is enabled.
Ticket #17 remains blocked while the requirements below are unresolved. Stable
direct regression and MLPE delivery are independent of this decision.

## Supported observations

The exact diagnostic assumes M independent, identically distributed Gaussian
marker contrast vectors with known zero mean. There are N identified populations,
and p=N-1 independent population contrast coordinates. Population centering removes
the unidentifiable common population level; it does not estimate and subtract a
mean across marker replicates. The latter operation changes degrees of freedom
and is outside this contract.

For marker measurements x, the input is explicitly either

```
D_ij = (1/M) sum_l (x_il - x_jl)^2   # marker_scaling="average"
D_ij =       sum_l (x_il - x_jl)^2   # marker_scaling="scatter"
```

The caller must declare `interpretation="squared_gaussian_marker_distance"`.
The complete matrix must be finite, nonnegative, exactly symmetric, and have a
zero diagonal. Missing pairs, generic FST summaries, relatedness, arbitrary
dissimilarities, and implicit squaring or target transformations are unsupported.
Declaring the interpretation does not establish Gaussianity or marker independence
for empirical observations. There is no adapter that silently converts ordinary
`PairwiseObservations` into this experimental input.

`marker_count` is an explicit integer count of independent Gaussian replicates,
never a default inferred from the population count. The supported ordinary
Wishart density requires M>p-1 and positive-definite empirical scatter. Fractional
effective counts, LD corrections, non-Gaussian genotype approximations, per-marker
weights, unequal marker sets across populations, and pair-dependent sample sizes
require separate justification and are currently rejected or unsupported.

EEMS also motivates its Wishart construction using independent Gaussian marker
contrasts, while explicitly treating real genotypes and marker dependence as an
approximation with a separately estimated effective count. This diagnostic does
not import that approximation or its priors. [Petkova, Novembre and Stephens,
2016, Methods](https://stephenslab.uchicago.edu/assets/papers/Petkova2016.pdf).

## Centered representation and reference density

Let H be the documented row-orthonormal Helmert basis returned by
`observations.contrast_basis`: H1=0 and HH'=I. Then

```
B = -H D H'/2
S = M B   # average input
S = B     # scatter input
S ~ Wishart_p(M, Sigma)
```

In particular, S=(Hx)(Hx)' for the corresponding Gaussian marker matrix.
`centered_scatter` returns S. The implementation checks positive definiteness in
contrast space, where the ordinary density is defined; the N-dimensional
population-centered matrix is necessarily singular. Neither missing targets nor
negative eigenvalues are repaired. No covariance clipping, nearest-positive
projection, diagonal jitter, or inferred marker scaling is applied. Matrices
numerically singular at the ordinary float64 rank threshold are rejected; symmetry
checks on computed covariance allow floating-point roundoff without modifying it.

`wishart_log_likelihood(observations, covariance)` evaluates the complete normalized
density, including the multivariate gamma and empirical determinant terms. Its
covariance is **per marker**, in the same H coordinates. For marker-average
covariance, it includes the Jacobian `p(p+1)/2 * log(M)` from S=M B. Thus average
and scatter inputs describe the same sample but have different density units.
The standard SPD Wishart density and support used as an independent reference are
documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html).

## Landscape covariance and identification

The graph diagnostic evaluates actual JAXScape resistance R from a conductance
encoder on prepared features. It proposes the fixed regional covariance

```
K_R = -H R H'/2
Sigma = a K_R + tau I
a > 0; tau >= 0; Sigma strictly positive definite
```

Here tau is a scientific population-specific noise variance, not numerical jitter.
This corresponds to expected off-diagonal marker-average squared distances
`E[D_ij]=a R_ij+2 tau`. It does not establish a biological interpretation of the
conductance surface. Repeated graph locations can make K_R singular, in which case
tau=0 may be inadmissible. Adding an empirical-data nugget to make an invalid
observed covariance pass is prohibited.

With a frozen encoder, a and tau are algebraically distinguishable only if
`vec(K_R)` and `vec(I)` are linearly independent. The report returns this design's
rank and condition number, which do not establish finite-sample statistical
identifiability. If K_R is proportional to I, the two parameters are confounded.
When fitting the encoder too, multiplying every conductance by c divides R by c;
simultaneously multiplying a by c leaves Sigma unchanged. Joint training needs a
reviewed explicit conductance normalization or a fixed scale. This diagnostic
accepts fixed a and tau and performs no fitting.

## Held-out population score: derivation and information boundary

This protocol is derived here from the stated Gaussian sampling model. It is not
an assertion that arbitrary pair holdout is Wishart-valid.

Before fitting, choose at least two training populations and a training anchor b.
Retain at least one held-out population. Let A have rows `e_i-e_b`, ordered with
the non-anchor training populations first and the held-out populations last.
Then the anchored scatter and model covariance are

```
S_A = -A D_scatter A'/2
Sigma_A = (A H') Sigma (H A')
```

The training/training principal block depends only on training genetic distances.
It has the marginal Wishart distribution with the same M and the corresponding
principal model block. Freeze the encoder, scale, nugget, and marker assumptions
after fitting that training block. The diagnostic requires the explicit declaration
`parameter_source="training_only"`, or `"fixed"` for known synthetic parameters.
This records the caller's declaration; the reference cannot audit prior access to
targets. An eventual fitter must enforce and persist the actual provenance.

The full-data genetic matrix is used **only to evaluate** the remaining blocks:

```
log p(S_remaining | S_TT) = log p_Wishart(S_A) - log p_Wishart(S_TT)
```

This is a proper conditional density, including training/held-out cross-block
information and held-out/held-out scatter, conditional on the training scatter.
Both densities use the same anchor coordinates, covariance convention, marker
count, and complete marker set. Separate full/training Helmert bases cannot be
subtracted as if they were principal blocks. Full-data population centering is
not used to form the training genetic statistic.

In anchor coordinates the isotropic orthonormal nugget becomes `tau A A'`, which
is `tau (I+11')`; replacing it with `tau I` would change the model. For average
inputs the conditional Jacobian is `(r_p-r_t) log(M)`, where
`r_d=d(d+1)/2` and t is the number of non-anchor training populations.

An independent verification uses the Gaussian regression decomposition. Write
`C=Sigma_HT Sigma_TT^-1` and `V=Sigma_HH-C Sigma_TH`. Conditional on S_TT,
`S_HT` is matrix normal with mean `C S_TT`, row covariance V, and column
covariance S_TT. Independently, the residual Schur scatter is Wishart with
degrees of freedom M-t and covariance V. The normalized sum of these densities
matches the joint/marginal ratio. The independent matrix-normal normalization is
documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html).

`HeldoutWishartScore.coordinate_measure` explicitly records anchored scatter or
marker-average covariance entries. The result is not a pair-mean loss, a density
of raw marker measurements, or a confidence interval. Comparing numerical scores
across different units, marker scalings, or population dimensions requires an
explicit evaluation design. Predictions cannot condition on the held-out targets;
scoring necessarily observes them after parameters are frozen.

## Reproducible evidence and remaining gate

Run the synthetic reference workflow after installing the toolkit:

```bash
JAX_ENABLE_X64=true python -m ilg_toolkit.experimental
```

It uses a two-feature ResNet9 encoder, an actual small resistance graph, simulated
Gaussian marker contrasts, and known fixed scale/nugget values. The tests compare
normalized densities to SciPy, conditional scores to the independent block
decomposition, and empirical scatter to raw Gaussian marker cross-products.
Changing held-out marker observations leaves the training marginal unchanged.
Invalid matrices, unsupported interpretations, missing scaling, invalid marker
counts, and inappropriate holdout declarations fail explicitly.

The independently reviewed implementation assessment remains **no-go for #17**,
despite the coherent strict synthetic reference, because:

- No empirical target adapter establishes Gaussian marker contrast validity,
  independent/effective marker counts, or validity for FST or relatedness.
- Joint encoder/regional-scale normalization, nuisance estimation, and behavior
  near scale/nugget confounding have not been specified and independently reviewed.
- The future fitter must enforce training-only provenance, fixed marker sets,
  population holdout, and target access instead of trusting a reference caller's
  declaration. Differentiable fitting and scientific validation are unimplemented.

These requirements can be resolved by evidence and an independently reviewed
go contract; this document imposes no human-only approval mechanism. The recorded
no-go review supports the present diagnostic boundary, does not resolve those
requirements, and does not enable training.
