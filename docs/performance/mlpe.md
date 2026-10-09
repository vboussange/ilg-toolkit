# Exact MLPE covariance acceleration

The JAX likelihood, signed GLS profiling, standalone calibration optimizer, and
population-effect posterior use the sampling-unit-sized endpoint system.
The Gaussian statistical model and variance parameterizations are unchanged.
Dense covariance calculations remain in independent tests and this benchmark.

For m observed pairs and n sampling units, Z has two endpoint indicators per
row. With unit variance u and residual variance d=e+j (explicit jitter j),
V=d I+u Z Z.T. Build G=Z.T Z by endpoint scatters, then factor
K=I+(u/d)G. The exact determinant is m log d+logdet K. No m*m covariance or
m*n incidence matrix is allocated in ordinary fitting.

For any right-hand-side columns B, let A=(u/d) solve(K,Z.T B) and R=B-Z A.
Then B.T V^-1 B=R.T R/d+A.T A/u. This positive representation avoids subtracting
large nearly equal Gram matrices when profiling fixed effects or evaluating the
residual quadratic. GLS solves its two-column normal system without a slope
constraint. The posterior mean is the corresponding A for the fitted residual;
posterior covariance is u K^-1 and its precision factor is chol(K)/sqrt(u).
Ordinary storage is O(n^2+m), with n-sized Cholesky rather than m-sized Cholesky.

## Validity and precision

`pair_mask` is an optional boolean vector accepted by the pure likelihood and
posterior functions. False rows are absent from sample score moments, covariance,
normalization and quadratic terms, even if their values are nonfinite or their
endpoint indices are placeholders. The mask works in eager and JIT execution.

Profiling requires at least three observed pairs, nonconstant finite scores, a
nonsingular GLS design, and endpoint overlap that distinguishes the two variance
components. For disjoint pairs, covariance is (2u+e+j)I and the two components are
unidentified. Fixed-parameter likelihood scoring remains valid for one query
pair when supplied with the stored training moments and coefficients.

The kernels preserve their explicit **nonfinite-result diagnostic contract**:
invalid input, singular/unidentified profiling, failed factorization, or an
unsupported numerical range produces nonfinite NLL and coefficients; the
posterior helper returns nonfinite arrays. These are failure sentinels, never
usable fit results. Host calibration rejects failed optimization and posterior
results; every `MLPEHead` construction rejects nonfinite scalar/posterior state.
Any optimizer using the pure kernels must check all returned likelihoods,
coefficients, gradients and posterior arrays before accepting an update/head.

The conservative supported range is
`eps * (1 + 2 * max_endpoint_degree * u/(e+j)) <= 0.01`, using the actual floating
precision. Gershgorin bounds give this upper bound on eps*cond(K), including
incomplete and bipartite endpoint graphs. Failure never changes a variance,
adds jitter, or substitutes another statistical model. Host calibration uses
float64 and reports this range in its numerical error. JAX inputs control kernel
precision; importing the package does not enable x64 globally. Use
`with jax.enable_x64():` and float64 scores/variance parameters for large ratios.
The bound is deliberately conservative for well-conditioned complete graphs.

Independent tests compare full ML likelihoods, signed coefficients, and score
and raw-variance gradients on complete/incomplete pairs in arbitrary order, with
ratios from 1e-8 to 1e8. Ordinary float64 NLL/coefficients agree at 1e-10 absolute
precision; at extreme ratios, dense-reference comparisons use 2e-7 relative
likelihood/coefficient and 2e-6 relative gradient tolerances. At u/e=1e8 the
float64 dense variance gradient itself suffers cancellation. A separate 60-digit
Decimal dense Gaussian-elimination oracle validates that gradient. Independent
central finite differences, masked rows, eager/JIT agreement, frozen R/lme4
calibration, and direct joint-Gaussian posteriors cover the additional seams.

## Reproducible host measurement

Run from this checkout after installing dependencies:

```bash
PYTHONPATH=src python benchmarks/benchmark_mlpe.py \
  --sizes 12 24 48 72 --repeats 5 --output /tmp/mlpe-host-scaling.json
```

Recorded 2026-10-09 on an AMD EPYC 7742 CPU, Python 3.12.13, NumPy 2.4.4,
SciPy 1.17.1, Linux. Each size/backend runs in a fresh process with one BLAS
thread, float64 inputs, one warm call and five timed calls. Both methods
precompute their endpoint crossproduct before timing. Times include covariance
factorization, signed GLS, and full-ML NLL at fixed variances. Data preparation,
encoder/graph solves, optimization iterations, JAX compilation, gradients, and
GPU/device allocation are outside this measurement.

Peak memory is whole-process `ru_maxrss`, including imports, native workspace,
and allocator behavior. The increase column subtracts the process high-water
mark after imports from the final high-water mark; it is not exact allocation
size. Zero increase means the calculation did not exceed the earlier high-water
mark. The raw measurements and environment are in
[mlpe-host-scaling.json](mlpe-host-scaling.json).

| Units | Pairs | Dense ms | Endpoint ms | Dense peak MiB | Endpoint peak MiB | Dense increase MiB | Endpoint increase MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 66 | 0.153 | 0.130 | 253.0 | 252.7 | 0.73 | 0.00 |
| 24 | 276 | 1.261 | 0.185 | 257.3 | 252.0 | 5.37 | 0.00 |
| 48 | 1,128 | 34.631 | 0.370 | 297.3 | 253.2 | 45.12 | 0.97 |
| 72 | 2,556 | 266.595 | 0.666 | 467.7 | 253.9 | 215.31 | 1.29 |

For 72 sampling units and 2,556 pairs, this host calculation reduced median
latency from about 267 ms to 0.666 ms and process peak RSS from 468 MiB to
254 MiB. These measurements establish the benefit of removing the dense pair
covariance on this workload; they do not predict complete-training acceleration
when encoder/solver computation dominates. Timings are not CI assertions.
