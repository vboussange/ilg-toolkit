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


def dense_profile(scores, targets, left, right, n_populations, variances, *, jitter=0):
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
