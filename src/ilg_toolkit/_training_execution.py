"""Bounded numeric species tails and one regional encoder VJP.

No host identities enter these kernels. A chunk reduces its scalar objective
before differentiation and completes its backward pass before the next chunk.
The encoder parameter tree is absent from every tail gradient.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from .mlpe import finalize_mlpe_fit, mlpe_ml_negative_log_likelihood, profiled_mlpe_ml_fit


class LossSettings(NamedTuple):
    objective: str
    mlpe_variance_floor: float
    mlpe_jitter: float


def settings(config):
    # Scheduling/seed/registry identity must not specialize numerical kernels.
    return LossSettings(config.objective, config.mlpe_variance_floor, config.mlpe_jitter)


def supports_pooled(model):
    return callable(getattr(model, "_split_pooled", None))


def join_backbone(tail, backbone):
    return eqx.tree_at(lambda model: model.backbone, tail, backbone, is_leaf=lambda x: x is None)


def _encode(backbone, tail, features, *, inference, key):
    model = tail if backbone is None else join_backbone(tail, backbone)
    return model._pooled_features(features, inference=inference, key=key)


_compiled_encode = eqx.filter_jit(_encode)


def pooled_region(model, features, *, jit, inference=True, key=None):
    backbone, tail = model._split_pooled()
    encode = _compiled_encode if jit else _encode
    pullback = None
    if backbone is not None and not inference:
        pooled, pullback = eqx.filter_vjp(
            lambda b: encode(b, tail, features, inference=False, key=key), backbone
        )
    else:
        pooled = encode(backbone, tail, features, inference=inference, key=key)
    jax.block_until_ready(pooled)
    return tail, pooled, pullback


# Cache construction of the encoder VJP as well as its numerical forward pass.
# Returning the pullback PyTree keeps the existing tail/backward arithmetic cuts.
compiled_pooled_region = eqx.filter_jit(pooled_region)


def shape_chunks(groups, size):
    buckets = {}
    for group in groups:
        payload = group.payload
        shape = tuple((leaf.shape, leaf.dtype) for leaf in jax.tree.leaves(payload[1:]))
        buckets.setdefault(shape, []).append(group)
    for bucket in buckets.values():
        for start in range(0, len(bucket), size):
            yield tuple(bucket[start : start + size])


def stack_payload(chunk):
    # Keep a shared raster out of the batched payload, including device copies.
    payloads = tuple(
        group.payload._replace(features=jnp.empty((0,), dtype=group.payload.features.dtype))
        for group in chunk
    )
    return jax.tree.map(lambda *values: jnp.stack(values), *payloads)


def add_gradients(total, value):
    return value if total is None else jax.tree.map(lambda left, right: left + right, total, value)


def _distances(tail, pooled, payload, context):
    values = tail._pairs_from_pooled(
        pooled, payload.distance_pairs, payload.species_index, context=context
    )
    if values.shape != payload.targets.shape:
        raise ValueError("Distance model must return one value per prepared pair")
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values) | (values < 0)),
        "Distance model produced nonfinite or negative distances",
    )


def _loss(distances, raw, payload, config):
    if config.objective == "direct_log1p":
        return jnp.mean((jnp.log1p(distances) - jnp.log1p(payload.targets)) ** 2)
    nll, _ = profiled_mlpe_ml_fit(
        distances.astype(jnp.float64),
        payload.targets,
        *payload.pairs,
        n_populations=len(payload.nodes),
        raw_variances=raw[payload.group_index],
        variance_floor=config.mlpe_variance_floor,
        jitter=config.mlpe_jitter,
    )
    return nll / len(distances)


def _chunk_loss(parameters, payloads, *, config, context):
    tail, pooled, raw = parameters

    def group(payload):
        return payload.weight * _loss(
            _distances(tail, pooled, payload, context), raw, payload, config
        )

    return jnp.sum(jax.vmap(group)(payloads))


def chunk_gradient(parameters, payloads, *, config, context):
    value, gradient = eqx.filter_value_and_grad(_chunk_loss)(
        parameters, payloads, config=config, context=context
    )
    bad = ~jnp.isfinite(value)
    for leaf in jax.tree.leaves(gradient):
        bad |= jnp.any(~jnp.isfinite(leaf))
    return eqx.error_if(
        (value, gradient),
        bad,
        "Nonfinite objective or gradient; check nonconstant distances, "
        "identifiable endpoints and variance conditioning",
    )


compiled_chunk_gradient = eqx.filter_jit(chunk_gradient)


def group_gradient(model, raw, payloads, *, config, context=None, key=None):
    """Compile one compatible bounded chunk and its single encoder pullback."""
    backbone, tail = model._split_pooled()
    features = payloads[0].features
    pullback = None
    if backbone is None:
        pooled = _compiled_encode(backbone, tail, features, inference=False, key=key)
    else:
        pooled, pullback = eqx.filter_vjp(
            lambda value: _compiled_encode(value, tail, features, inference=False, key=key),
            backbone,
        )
    numeric = tuple(
        payload._replace(features=jnp.empty((0,), dtype=features.dtype)) for payload in payloads
    )
    batched = jax.tree.map(lambda *values: jnp.stack(values), *numeric)
    value, (tail_gradient, feature_gradient, raw_gradient) = compiled_chunk_gradient(
        (tail, pooled, raw), batched, config=config, context=context
    )
    if pullback is not None:
        (backbone_gradient,) = pullback(feature_gradient)
        model_gradient = join_backbone(tail_gradient, backbone_gradient)
    else:
        model_gradient = tail_gradient
    return value, model_gradient if raw is None else (model_gradient, raw_gradient)


compiled_group_gradient = eqx.filter_jit(group_gradient)


def chunk_evaluate(tail, pooled, raw, payloads, moments, *, config, context, mode):
    def group(payload, fixed):
        distances = _distances(tail, pooled, payload, context)
        if mode == "direct":
            return _loss(distances, None, payload, config)
        distances = distances.astype(jnp.float64)
        if mode == "refresh":
            return finalize_mlpe_fit(
                distances,
                payload.targets,
                *payload.pairs,
                n_populations=len(payload.nodes),
                raw_variances=raw[payload.group_index],
                variance_floor=config.mlpe_variance_floor,
                jitter=config.mlpe_jitter,
            )
        center, scale, intercept, slope = fixed
        return mlpe_ml_negative_log_likelihood(
            distances,
            payload.targets,
            *payload.pairs,
            n_populations=len(payload.nodes),
            fixed_effects=jnp.stack((intercept, slope)),
            raw_variances=raw[payload.group_index],
            distance_center=center,
            distance_scale=scale,
            variance_floor=config.mlpe_variance_floor,
            jitter=config.mlpe_jitter,
        ) / len(distances)

    return jax.vmap(group)(payloads, moments)


compiled_chunk_evaluate = eqx.filter_jit(chunk_evaluate)
