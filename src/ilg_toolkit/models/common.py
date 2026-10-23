"""Shared raster-to-patch utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


class _PatchModel(Protocol):
    patch_size: int
    patch_microbatch_size: int | None


def encode_patches(
    model: _PatchModel, features: jax.Array, encode: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Pooled landscape features computed from raster patches."""
    patches, shape = patchify(features, model.patch_size)
    batch_size = model.patch_microbatch_size
    if batch_size is None:
        values = eqx.filter_vmap(encode)(patches)
    else:
        values = jax.lax.map(eqx.filter_checkpoint(encode), patches, batch_size=batch_size)
    return values.reshape(*shape, -1)


def with_dropout_mode[Model](model: Model, *, inference: bool, key: jax.Array | None) -> Model:
    """Distance model with dropout set for training or prediction."""

    def is_dropout(value):
        return isinstance(value, eqx.nn.Dropout)

    layers = jax.tree.leaves(model, is_leaf=is_dropout)
    count = sum(is_dropout(layer) for layer in layers)
    if count:
        if not inference and key is None:
            raise ValueError("Training with dropout requires a random key")
        keys = iter([None] * count if key is None else jax.random.split(key, count))
        model = jax.tree.map(
            lambda layer: (
                eqx.Partial(layer, inference=inference, key=next(keys))
                if is_dropout(layer)
                else layer
            ),
            model,
            is_leaf=is_dropout,
        )
    return model


def patchify(features: ArrayLike, patch_size: int) -> tuple[jax.Array, tuple[int, int]]:
    """Partition row, column, feature data into non-overlapping feature-first patches."""
    raster = jnp.asarray(features)
    if raster.ndim != 3:
        raise ValueError(f"Expected features with shape (row, column, feature), got {raster.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
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
    """Map row-major raster pixel indices to row-major patch nodes."""
    height, width = raster_shape
    if height % patch_size or width % patch_size:
        raise ValueError("Raster dimensions must be divisible by patch_size")
    nodes = jnp.asarray(pixel_nodes, dtype=jnp.int32)
    rows = (nodes // width) // patch_size
    columns = (nodes % width) // patch_size
    return rows * (width // patch_size) + columns
