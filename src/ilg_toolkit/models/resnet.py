"""A residual-network conductance model with a pooled backbone over raster patches."""

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
        """Residual convolutional features for one channel-first feature map."""
        residual = inputs
        outputs = jax.nn.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        return jax.nn.relu(outputs + residual)


@final
class ResNet9Conductance(ConductanceModel):
    """A conductance model using a pooled residual-network backbone over raster patches.

    Species-specific linear coefficients map each patch's features to positive
    conductance through softplus and ``min_conductance``.

    Args:
        in_channels:
            Number of raster covariates in the retained feature order.
        num_species: Number of species-specific output rows; default 1. The
            convolutional
            backbone is pooled across all regions and species.
        patch_size: Square patch side length, at least 4 (default 4). Raster dimensions
            must
            be divisible by it; output shape is ``(rows // patch_size, columns // patch_size)``.
            Sampling units in the same patch share a graph node.
        channels: Widths of the three convolutional stages, default ``(64, 128, 256)``
            as in the paper. Each width must be a positive multiple of 8 because
            GroupNorm uses 8 groups; narrower widths suit small studies and CPU runs.
        min_conductance: Positive conductance floor added after softplus, default 1e-6.
        patch_microbatch_size: Maximum patches processed together; ``None`` processes
            all patches.
            Lower values trade throughput for peak activation memory without changing
            the patch grid or the model's receptive field.
        solver_config: Resistance tolerances, iteration limit, and optional algebraic
            multigrid (AMG)
            preconditioner.
        key: JAX random key for reproducible parameter initialization.

    Notes:
        Patch convolutions and GroupNorm use float32 parameters; effective resistance
        uses float64, which must be enabled before constructing inputs. The network
        encodes each patch through two residual blocks and spatial pooling, followed
        by a species-specific linear map and softplus. It does not mix features across
        neighboring patches; the resistance graph connects their conductances.

        See the maximum-likelihood population effects (MLPE)
        [workflow](../examples/mlpe_workflow.md) for pooled regional training,
        solver choices, and practical patch/cropping guidance.
    """

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
    channels: tuple[int, int, int] = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    min_conductance: float = eqx.field(static=True)
    patch_microbatch_size: int | None = eqx.field(static=True)
    solver_config: ResistanceSolverConfig = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        num_species: int = 1,
        patch_size: int = 4,
        channels: tuple[int, int, int] = (64, 128, 256),
        min_conductance: float = 1e-6,
        patch_microbatch_size: int | None = None,
        solver_config: ResistanceSolverConfig | None = None,
        key: jax.Array,
    ):
        if isinstance(num_species, bool) or not isinstance(num_species, int) or num_species < 1:
            raise ValueError("num_species must be a positive integer")
        if (
            isinstance(in_channels, bool)
            or not isinstance(in_channels, Integral)
            or in_channels < 1
        ):
            raise ValueError("in_channels must be a positive integer")
        if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size < 4:
            raise ValueError("ResNet9Conductance requires integer patch_size >= 4")
        widths = tuple(channels)
        if len(widths) != 3 or any(
            isinstance(width, bool) or not isinstance(width, Integral) or width < 8 or width % 8
            for width in widths
        ):
            raise ValueError("ResNet9Conductance channels must be three positive multiples of 8")
        if not math.isfinite(min_conductance) or min_conductance <= 0:
            raise ValueError("min_conductance must be finite and positive")
        validate_patch_microbatch_size(patch_microbatch_size)
        solver_config = ResistanceSolverConfig() if solver_config is None else solver_config
        if not isinstance(solver_config, ResistanceSolverConfig):
            raise TypeError("solver_config must be a ResistanceSolverConfig")
        first, second, third = (int(width) for width in widths)
        keys = jax.random.split(key, 8)
        self.conv1 = eqx.nn.Conv2d(in_channels, first, 3, padding=1, dtype=jnp.float32, key=keys[0])
        self.norm1 = eqx.nn.GroupNorm(8, first, dtype=jnp.float32)
        self.residual1 = ResidualBlock(first, key=keys[1])
        self.conv2 = eqx.nn.Conv2d(first, second, 3, padding=1, dtype=jnp.float32, key=keys[2])
        self.norm2 = eqx.nn.GroupNorm(8, second, dtype=jnp.float32)
        self.residual2 = ResidualBlock(second, key=keys[3])
        self.conv3 = eqx.nn.Conv2d(second, third, 3, padding=1, dtype=jnp.float32, key=keys[4])
        self.norm3 = eqx.nn.GroupNorm(8, third, dtype=jnp.float32)
        self.output = eqx.nn.Linear(third, num_species, dtype=jnp.float32, key=keys[5])
        self.num_species = num_species
        self.channels = (first, second, third)
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

    def conductance(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Positive conductance surface from pooled landscape features."""
        assert self.output.bias is not None
        logits = latent @ self.output.weight[species_index] + self.output.bias[species_index]
        return jax.nn.softplus(logits) + self.min_conductance
