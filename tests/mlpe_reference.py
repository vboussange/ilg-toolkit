"""Frozen full-ML oracle from R 4.1.2, lme4 1.1-28, ResistanceGA2 1.1.20.

REML=FALSE, explicit two-endpoint random effects, R sample-SD scaling.
Copied numeric data from the Apache-2.0 research implementation; tests need no R.
"""

import numpy as np

_R_SCORES = np.array(
    [
        -1.0847845325555074,
        -0.7856562765597935,
        1.4076148426443598,
        -0.005894023710191141,
        -1.375475831778455,
        0.8982644456212491,
        -1.2894979474794355,
        1.5465626070493472,
        -1.0016534312119365,
        -0.07602355302299327,
        -0.9160524297215114,
        0.38800235096599806,
        -1.378941894269037,
        0.5617899637344953,
        -1.606879982245179,
        0.7412399591855078,
        -0.9872195123319644,
        1.9954955626370834,
        -0.2919254085375479,
        -0.9038371900227332,
        -0.9801366442736841,
        -0.21221830555519952,
        -0.3569980434650668,
        -0.5131156917245834,
        1.5220664939667614,
        -0.4262339398597622,
        0.4617831441001972,
        0.6577818753941715,
    ]
)

_R_TARGETS = np.array(
    [
        0.7867947762864243,
        0.5844969351554589,
        -0.19944738061299205,
        0.7820143760316058,
        1.4691315076235152,
        0.4817642650020033,
        0.9965003483084469,
        -0.6201612100865199,
        1.7103092068446286,
        1.1655587443537478,
        1.7845260919265316,
        1.0164728564848375,
        1.003964815178125,
        0.6983581279293211,
        2.5398922377512445,
        1.029933745907628,
        2.599668270650879,
        -0.6011444799317919,
        1.9785807335604428,
        2.513022681996845,
        2.4701382843858752,
        0.9982332167064097,
        2.087895648701189,
        2.5344018288602435,
        0.1677412867636987,
        2.8312101442620357,
        0.7609805324512989,
        1.2129905800176382,
    ]
)

_R_FIXED = np.array([1.2422795775895932, -0.6653657702662492])

_R_VARIANCES = np.array([0.23562746377731067, 0.02879401793899513])

_R_LOG_LIKELIHOOD = -6.137989809270536

_R_BLUPS = np.array(
    [
        -0.7520639701118881,
        -0.46501385425700786,
        -0.13777554492600885,
        0.1958548966482916,
        0.3832154152413225,
        0.4273528598272225,
        0.7313024458218655,
        -0.3828722482437748,
    ]
)


def dense_profile(scores, targets, left, right, n_populations, variances, *, jitter: float = 0):
    """Independent dense full-covariance oracle retained for acceleration checks."""
    scores = np.asarray(scores, dtype=np.float64)
    z = np.zeros((len(scores), n_populations))
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        z[row, a] = z[row, b] = 1
    x = np.column_stack([np.ones(len(scores)), (scores - scores.mean()) / scores.std(ddof=1)])
    v = variances[0] * (z @ z.T) + (variances[1] + jitter) * np.eye(len(scores))
    beta = np.linalg.lstsq(
        x.T @ np.linalg.solve(v, x), x.T @ np.linalg.solve(v, targets), rcond=None
    )[0]
    r = np.asarray(targets) - x @ beta
    nll = 0.5 * (
        len(scores) * np.log(2 * np.pi) + np.linalg.slogdet(v)[1] + r @ np.linalg.solve(v, r)
    )
    return nll, beta


def dense_jax_profile(scores, targets, left, right, n_populations, raw_variances):
    """Dense autodiff oracle independent of endpoint-system arithmetic."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import cho_solve

    z = np.zeros((len(left), n_populations))
    for row, (a, b) in enumerate(zip(left, right, strict=True)):
        z[row, a] = z[row, b] = 1
    z = jnp.asarray(z, dtype=scores.dtype)
    variances = jax.nn.softplus(raw_variances) + 1e-10
    x = jnp.column_stack((jnp.ones_like(scores), (scores - scores.mean()) / scores.std(ddof=1)))
    v = variances[0] * (z @ z.T) + variances[1] * jnp.eye(len(scores))
    factor = jnp.linalg.cholesky(v)
    beta = jnp.linalg.solve(
        x.T @ cho_solve((factor, True), x), x.T @ cho_solve((factor, True), targets)
    )
    r = targets - x @ beta
    return 0.5 * (
        len(scores) * np.log(2 * np.pi)
        + 2 * jnp.log(jnp.diag(factor)).sum()
        + r @ cho_solve((factor, True), r)
    ), beta


def decimal_dense_profile_and_gradients(scores, targets, left, right, raw_variances):
    """60-digit dense Gaussian oracle when float64 covariance gradients cancel.

    Uses standard-library Decimal Gaussian elimination, not the Woodbury or
    positive-bilinear implementation. It is intentionally tiny test machinery.
    """
    import math
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 60

        def d(value):
            return Decimal(str(float(value)))

        size = len(scores)
        s, y = list(map(d, scores)), list(map(d, targets))
        raw = list(map(d, raw_variances))
        u, e = [(Decimal(1) + value.exp()).ln() + Decimal("1e-10") for value in raw]
        overlap = [
            [len({left[i], right[i]} & {left[j], right[j]}) for j in range(size)]
            for i in range(size)
        ]
        # Gauss-Jordan inversion of the full pair covariance, with determinant.
        a = [
            [u * overlap[i][j] + (e if i == j else 0) for j in range(size)]
            + [Decimal(i == j) for j in range(size)]
            for i in range(size)
        ]
        determinant = Decimal(1)
        for i in range(size):
            pivot = max(range(i, size), key=lambda row: abs(a[row][i]))
            if pivot != i:
                a[i], a[pivot] = a[pivot], a[i]
                determinant = -determinant
            value = a[i][i]
            determinant *= value
            a[i] = [entry / value for entry in a[i]]
            for j in range(size):
                if j != i:
                    coefficient = a[j][i]
                    a[j] = [v - coefficient * b for v, b in zip(a[j], a[i], strict=True)]
        inverse = [row[size:] for row in a]
        center = sum(s) / size
        scale = (sum((value - center) ** 2 for value in s) / (size - 1)).sqrt()
        z = [(value - center) / scale for value in s]
        x = [[Decimal(1), value] for value in z]
        vinv_y = [sum(v * value for v, value in zip(row, y, strict=True)) for row in inverse]
        vinv_x = [
            [sum(inverse[i][j] * x[j][k] for j in range(size)) for k in range(2)]
            for i in range(size)
        ]
        normal = [
            [sum(x[i][j] * vinv_x[i][k] for i in range(size)) for k in range(2)] for j in range(2)
        ]
        rhs = [sum(x[i][j] * vinv_y[i] for i in range(size)) for j in range(2)]
        denominator = normal[0][0] * normal[1][1] - normal[0][1] * normal[1][0]
        beta = [
            (rhs[0] * normal[1][1] - rhs[1] * normal[0][1]) / denominator,
            (rhs[1] * normal[0][0] - rhs[0] * normal[1][0]) / denominator,
        ]
        residual = [y[i] - beta[0] - beta[1] * z[i] for i in range(size)]
        w = [sum(v * value for v, value in zip(row, residual, strict=True)) for row in inverse]
        nll = (
            size * d(math.log(2 * math.pi))
            + determinant.ln()
            + sum(a * b for a, b in zip(residual, w, strict=True))
        ) / 2
        wz = sum(a * b for a, b in zip(w, z, strict=True))
        score_gradient = [
            -beta[1] * (w[i] - sum(w) / size - z[i] * wz / (size - 1)) / scale for i in range(size)
        ]
        trace_unit = sum(inverse[i][j] * overlap[i][j] for i in range(size) for j in range(size))
        quadratic_unit = sum(w[i] * w[j] * overlap[i][j] for i in range(size) for j in range(size))
        variance_gradient = [
            (trace_unit - quadratic_unit) / (2 * (1 + (-raw[0]).exp())),
            (sum(inverse[i][i] for i in range(size)) - sum(value**2 for value in w))
            / (2 * (1 + (-raw[1]).exp())),
        ]
        return (
            float(nll),
            np.array(list(map(float, beta))),
            (
                np.array(list(map(float, score_gradient))),
                np.array(list(map(float, variance_gradient))),
            ),
        )
