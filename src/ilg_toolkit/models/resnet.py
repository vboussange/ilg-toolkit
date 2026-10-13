"""Pooled ResNet-9 patch features with species-specific positive conductances."""

from __future__ import annotations

import math
from numbers import Integral
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from .._pairs import PairSelection
from ..config import ResistanceSolverConfig, validate_patch_batch_size
from ..resistance import _effective_resistance_pairs
from .base import ConductanceModel, validate_species
from .common import patchify


@eqx.filter_checkpoint
def _encode_patch(model: _ResNetBackbone, patch: jax.Array) -> jax.Array:
    """Recompute CNN activations in the backward pass instead of retaining them."""
    return model(patch)


@final
class ResidualBlock(eqx.Module):
    """Two convolutions with GroupNorm and an identity skip connection."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, channels: int, *, key: jax.Array):
        first, second = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(channels, channels, 3, padding=1, dtype=jnp.float32, key=first)
        self.conv2 = eqx.nn.Conv2d(channels, channels, 3, padding=1, dtype=jnp.float32, key=second)
        self.norm1 = eqx.nn.GroupNorm(min(8, channels), channels, dtype=jnp.float32)
        self.norm2 = eqx.nn.GroupNorm(min(8, channels), channels, dtype=jnp.float32)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Encode one channel-first feature map."""
        residual = inputs
        outputs = jax.nn.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return jax.nn.relu(outputs + residual)


@final
class _ResNetBackbone(eqx.Module):
    """Shared patch encoder before the species-specific scalar projection."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm3: eqx.nn.GroupNorm
    residual1: ResidualBlock
    residual2: ResidualBlock

    def __init__(self, in_channels: int, *, key: jax.Array):
        keys = jax.random.split(key, 8)
        self.conv1 = eqx.nn.Conv2d(in_channels, 64, 3, padding=1, dtype=jnp.float32, key=keys[0])
        self.norm1 = eqx.nn.GroupNorm(8, 64, dtype=jnp.float32)
        self.residual1 = ResidualBlock(64, key=keys[1])
        self.conv2 = eqx.nn.Conv2d(64, 128, 3, padding=1, dtype=jnp.float32, key=keys[2])
        self.norm2 = eqx.nn.GroupNorm(8, 128, dtype=jnp.float32)
        self.residual2 = ResidualBlock(128, key=keys[3])
        self.conv3 = eqx.nn.Conv2d(128, 256, 3, padding=1, dtype=jnp.float32, key=keys[4])
        self.norm3 = eqx.nn.GroupNorm(8, 256, dtype=jnp.float32)

    def __call__(self, patch: jax.Array) -> jax.Array:
        patch = jnp.asarray(patch, dtype=self.conv1.weight.dtype)
        values = jax.nn.relu(self.norm1(self.conv1(patch)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual1(values)
        values = jax.nn.relu(self.norm2(self.conv2(values)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual2(values)
        values = jax.nn.relu(self.norm3(self.conv3(values)))
        return jnp.mean(values, axis=(1, 2))


@final
class ResNet9Conductance(ConductanceModel):
    """One pooled patch backbone with a positive scalar projection per species."""

    species: tuple[str, ...] = eqx.field(static=True)
    backbone: _ResNetBackbone
    projection_weight: jax.Array
    projection_bias: jax.Array
    patch_size: int = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    patch_batch_size: int | None = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        species,
        patch_size: int = 4,
        min_conductance: float = 1e-6,
        patch_batch_size: int | None = None,
        solver_config: ResistanceSolverConfig | None = None,
        key: jax.Array,
    ):
        self.species = validate_species(species)
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, Integral)
            or in_channels < 1
        ):
            raise ValueError("in_channels must be a positive integer")
        if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size < 4:
            raise ValueError("ResNet9Conductance requires integer patch_size >= 4")
        if not math.isfinite(min_conductance) or min_conductance <= 0:
            raise ValueError("min_conductance must be finite and positive")
        validate_patch_batch_size(patch_batch_size)
        solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        if not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        self.backbone = _ResNetBackbone(in_channels, key=key)
        head_key = jax.random.split(key, 8)[5]
        head_keys = jax.vmap(
            lambda index: jax.lax.cond(
                index == 0, lambda _: head_key, lambda i: jax.random.fold_in(head_key, i), index
            )
        )(jnp.arange(len(self.species)))
        heads = eqx.filter_vmap(
            lambda head_key: eqx.nn.Linear(256, 1, dtype=jnp.float32, key=head_key)
        )(head_keys)
        self.projection_weight = heads.weight[:, 0]
        assert heads.bias is not None
        self.projection_bias = heads.bias[:, 0]
        self.patch_size = patch_size
        self.min_conductance = min_conductance
        self.patch_batch_size = patch_batch_size
        self.solver_config = solver_config

    def _split_pooled(self):
        """Private encoder/tail partition for one regional VJP and bounded tails."""
        return self.backbone, eqx.tree_at(lambda model: model.backbone, self, None)

    def _pooled_features(self, features, *, inference=True, key=None, patch_batch_size=None):
        """Return patch features, checkpointing every configured encoder chunk.

        A missing override uses the constructor setting. None at construction
        encodes the whole region without checkpointing. A positive chunk size
        recomputes CNN activations during differentiation, including when all
        patches fit in one chunk. The scalar projection is applied afterwards.
        """
        if patch_batch_size is None:
            patch_batch_size = self.patch_batch_size
        validate_patch_batch_size(patch_batch_size)
        features = jnp.asarray(features)
        channels = self.backbone.conv1.weight.shape[1]
        if features.ndim != 3 or features.shape[-1] != channels:
            raise ValueError(f"Expected row, column, feature data with {channels} features")
        patches, grid_shape = patchify(features, self.patch_size)
        if patch_batch_size is None:
            pooled = eqx.filter_vmap(self.backbone)(patches)
        else:
            # Bounded map includes a ragged final chunk without unrolling one
            # CNN graph per chunk under JIT. Only pooled patch vectors survive.
            pooled = jax.lax.map(
                lambda patch: _encode_patch(self.backbone, patch),
                patches,
                batch_size=patch_batch_size,
            )
        return pooled.reshape(*grid_shape, -1)

    def _conductance_from_pooled(self, pooled, species_index):
        logits = (
            pooled @ self.projection_weight[species_index] + self.projection_bias[species_index]
        )
        return jax.nn.softplus(logits) + self.min_conductance

    def _conductance(self, features, species_index, *, patch_batch_size=None):
        pooled = self._pooled_features(features, patch_batch_size=patch_batch_size)
        return self._conductance_from_pooled(pooled, species_index)

    def _pairs_from_pooled(self, pooled, selection: PairSelection, species_index, *, context=None):
        if context is not None and context.config != self.solver_config:
            raise ValueError("Solver context policy does not match the conductance model settings")
        surface = self._conductance_from_pooled(pooled, species_index)
        if context is None:
            return _effective_resistance_pairs(surface, selection, config=self.solver_config)
        return _effective_resistance_pairs(surface, selection, context=context)

    def _pair_distances(
        self,
        features,
        selection,
        species_index,
        *,
        context=None,
        inference=True,
        key=None,
        patch_batch_size=None,
    ):
        pooled = self._pooled_features(
            features, inference=inference, key=key, patch_batch_size=patch_batch_size
        )
        return self._pairs_from_pooled(pooled, selection, species_index, context=context)
