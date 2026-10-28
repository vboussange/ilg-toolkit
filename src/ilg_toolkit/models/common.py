"""Shared raster-to-patch utilities."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from .._checks import _positive_integer


class _PatchModel(Protocol):
    patch_size: int
    patch_microbatch_size: int | None


def encode_patches(
    model: _PatchModel, features: jax.Array, encode: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Patch features computed from non-overlapping raster patches."""
    patches, shape = patchify(features, model.patch_size)
    batch_size = model.patch_microbatch_size
    if batch_size is None:
        values = eqx.filter_vmap(encode)(patches)
    else:
        values = jax.lax.map(eqx.filter_checkpoint(encode), patches, batch_size=batch_size)
    return values.reshape(*shape, -1)


def _is_stateful(value: object) -> bool:
    return isinstance(value, eqx.nn.StateIndex)


def _reject_stateful_layers(model: object) -> None:
    """Raises a ``TypeError`` when the distance model holds Equinox state indices."""
    if any(_is_stateful(leaf) for leaf in jax.tree.leaves(model, is_leaf=_is_stateful)):
        raise TypeError(
            "Distance models with stateful layers (Equinox state indices, such as "
            "eqx.nn.BatchNorm) are not supported; use stateless layers such as eqx.nn.GroupNorm"
        )


def _inference_model[Model](model: Model) -> Model:
    """Stateless distance model with every ``inference`` flag set for frozen evaluation."""
    _reject_stateful_layers(model)
    return eqx.nn.inference_mode(model, value=True)


def _takes_key(value: object) -> bool:
    if not isinstance(value, eqx.Module) or not hasattr(value, "inference"):
        return False
    call = getattr(type(value), "__call__", None)
    return call is not None and "key" in inspect.signature(call).parameters


def _training_model[Model](model: Model, key: jax.Array) -> Model:
    """Distance model in training mode, with random keys bound to its stochastic layers.

    Every ``inference`` flag is cleared. Each inference-dependent layer whose call
    accepts a ``key`` receives its own split of ``key``, in tree order.
    """
    model = eqx.nn.inference_mode(model, value=False)

    def is_layer(value: object) -> bool:
        return value is not model and _takes_key(value)

    count = sum(map(is_layer, jax.tree.leaves(model, is_leaf=is_layer)))
    if not count:
        return model
    keys = iter(jax.random.split(key, count))
    return jax.tree.map(
        lambda layer: eqx.Partial(layer, key=next(keys)) if is_layer(layer) else layer,
        model,
        is_leaf=is_layer,
    )


def patchify(features: ArrayLike, patch_size: int) -> tuple[jax.Array, tuple[int, int]]:
    """Non-overlapping feature-first patches from row, column, feature data."""
    raster = jnp.asarray(features)
    if raster.ndim != 3:
        raise ValueError(f"Expected features with shape (row, column, feature), got {raster.shape}")
    patch_size = _positive_integer("patch_size", patch_size)
    height, width, channels = raster.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Raster shape {(height, width)} is not divisible by patch_size={patch_size}"
        )
    rows, columns = height // patch_size, width // patch_size
    patches = raster.reshape(rows, patch_size, columns, patch_size, channels)
    patches = patches.transpose(0, 2, 4, 1, 3)
    return patches.reshape(rows * columns, channels, patch_size, patch_size), (rows, columns)


def pixel_to_patch_nodes(
    pixel_nodes: ArrayLike,
    *,
    raster_shape: tuple[int, int],
    patch_size: int,
) -> jax.Array:
    """Row-major patch vertices corresponding to row-major raster pixel indices."""
    height, width = raster_shape
    if height % patch_size or width % patch_size:
        raise ValueError("Raster dimensions must be divisible by patch_size")
    nodes = jnp.asarray(pixel_nodes, dtype=jnp.int32)
    rows = (nodes // width) // patch_size
    columns = (nodes % width) // patch_size
    return rows * (width // patch_size) + columns
