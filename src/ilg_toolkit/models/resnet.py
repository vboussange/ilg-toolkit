"""ResNet-9 patch encoder for positive graph conductances."""

from __future__ import annotations

import math
from numbers import Integral
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from ..config import ResistanceSolverConfig, validate_patch_microbatch_size
from .base import ConductanceModel
from .common import encode_patches


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
class ResNet9Conductance(ConductanceModel):
    """Map each native-resolution raster patch to one positive conductance."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    norm3: eqx.nn.GroupNorm
    residual1: ResidualBlock
    residual2: ResidualBlock
    output: eqx.nn.Linear
    num_species: int = eqx.field(static=True, default=1, kw_only=True)
    patch_size: int = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    patch_microbatch_size: int | None = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        patch_size: int = 4,
        min_conductance: float = 1e-6,
        patch_microbatch_size: int | None = None,
        solver_config: ResistanceSolverConfig | None = None,
        key: jax.Array,
    ):
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
        validate_patch_microbatch_size(patch_microbatch_size)
        solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        if not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        keys = jax.random.split(key, 8)
        self.conv1 = eqx.nn.Conv2d(in_channels, 64, 3, padding=1, dtype=jnp.float32, key=keys[0])
        self.norm1 = eqx.nn.GroupNorm(8, 64, dtype=jnp.float32)
        self.residual1 = ResidualBlock(64, key=keys[1])
        self.conv2 = eqx.nn.Conv2d(64, 128, 3, padding=1, dtype=jnp.float32, key=keys[2])
        self.norm2 = eqx.nn.GroupNorm(8, 128, dtype=jnp.float32)
        self.residual2 = ResidualBlock(128, key=keys[3])
        self.conv3 = eqx.nn.Conv2d(128, 256, 3, padding=1, dtype=jnp.float32, key=keys[4])
        self.norm3 = eqx.nn.GroupNorm(8, 256, dtype=jnp.float32)
        self.output = eqx.nn.Linear(256, 1, dtype=jnp.float32, key=keys[5])
        self.patch_size = patch_size
        self.min_conductance = min_conductance
        self.patch_microbatch_size = patch_microbatch_size
        self.solver_config = solver_config

    def _encode_patch(self, patch: jax.Array) -> jax.Array:
        """Pooled landscape features for one raster patch."""
        patch = jnp.asarray(patch, dtype=self.conv1.weight.dtype)
        values = jax.nn.relu(self.norm1(self.conv1(patch)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual1(values)
        values = jax.nn.relu(self.norm2(self.conv2(values)))
        values = eqx.nn.MaxPool2d(2, stride=2)(values)
        values = self.residual2(values)
        values = jax.nn.relu(self.norm3(self.conv3(values)))
        values = jnp.mean(values, axis=(1, 2))
        return values

    def backbone(self, features: jax.Array) -> jax.Array:
        """Pooled landscape features for each raster patch."""
        if features.ndim != 3 or features.shape[-1] != self.conv1.weight.shape[1]:
            raise ValueError(
                f"Expected row, column, feature data with {self.conv1.weight.shape[1]} features"
            )
        return encode_patches(self, features, self._encode_patch)

    def conductance(self, latent: jax.Array, species_index: int) -> jax.Array:
        """Positive conductance surface from pooled landscape features."""
        logits = eqx.filter_vmap(eqx.filter_vmap(self.output))(latent).squeeze(-1)
        return jax.nn.softplus(logits) + self.min_conductance
