"""A U-Net distance model deriving squared embedding distances from landscape features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp

from .._checks import _positive_integer
from .base import _species_names, EmbeddingDistanceModel
from .common import encode_patches


def _match_shape(values: jax.Array, reference: jax.Array) -> jax.Array:
    """A channel, height, width array cropped or zero-padded to a reference spatial shape."""
    target_height, target_width = reference.shape[1:]
    height, width = values.shape[1:]
    if height > target_height:
        start = (height - target_height) // 2
        values = values[:, start : start + target_height, :]
    elif height < target_height:
        before = (target_height - height) // 2
        after = target_height - height - before
        values = jnp.pad(values, ((0, 0), (before, after), (0, 0)))
    if width > target_width:
        start = (width - target_width) // 2
        values = values[:, :, start : start + target_width]
    elif width < target_width:
        before = (target_width - width) // 2
        after = target_width - width - before
        values = jnp.pad(values, ((0, 0), (0, 0), (before, after)))
    return values


@final
class ResidualConvBlock(eqx.Module):
    """Residual convolutional block used throughout the U-Net."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    projection: eqx.nn.Conv2d | None
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    dropout: eqx.nn.Dropout

    def __init__(self, inputs: int, outputs: int, *, dropout: float, key: jax.Array):
        first, second, projection = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(inputs, outputs, 3, padding=1, dtype=jnp.float32, key=first)
        self.conv2 = eqx.nn.Conv2d(outputs, outputs, 3, padding=1, dtype=jnp.float32, key=second)
        self.projection = (
            None
            if inputs == outputs
            else eqx.nn.Conv2d(inputs, outputs, 1, dtype=jnp.float32, key=projection)
        )
        self.norm1 = eqx.nn.GroupNorm(min(8, outputs), outputs, dtype=jnp.float32)
        self.norm2 = eqx.nn.GroupNorm(min(8, outputs), outputs, dtype=jnp.float32)
        self.dropout = eqx.nn.Dropout(dropout, inference=True)

    def __call__(self, values: jax.Array) -> jax.Array:
        """Residual convolutional features for one channel-first feature map."""
        residual = values if self.projection is None else self.projection(values)
        values = jax.nn.gelu(self.norm1(self.conv1(values)))
        values = self.norm2(self.conv2(values))
        values = self.dropout(values)
        return jax.nn.gelu(values + residual)


@final
class UNetEmbeddingDistance(EmbeddingDistanceModel):
    """A U-Net embedding-distance model with species-specific embeddings.

    A U-Net combines patch features across the region's patch grid, and squared
    embedding distances provide ecological distances between sampling units.

    Args:
        in_channels:
            Number of raster features, with meanings and order preserved across regions.
        species: Species names, one embedding projection each in parameter order;
            ``None`` (default) models one unnamed species.
        patch_size: Square raster patch side length, default 32. Both raster dimensions
            must be divisible by it; sampling units use their containing patch.
        base_channels: Width of the first U-Net stage, default 32. Later stages expand
            this width.
        embedding_dim: Dimension of each species' embedding at a patch, default 16.
        dropout: Dropout probability during training, default 0.1; prediction disables
            it.
        patch_microbatch_size: Maximum patches embedded together; ``None`` processes all
            patches. It limits patch-embedding work but does not microbatch the whole
            U-Net.
        key: JAX initialization key.

    Notes:
        The U-Net combines neighboring patch features through two downsampling
        stages, a bottleneck, and skip-connected upsampling. The output grid has one
        embedding per patch and species. Ecological distance is the sum of squared
        coordinate differences between the two sampled embeddings, without a resistance
        solve. Cropping changes the spatial context available to the U-Net and can
        therefore change predictions even at retained sampling locations.
    """

    patch_embedding: eqx.nn.Linear
    patch_norm: eqx.nn.GroupNorm
    encoder1: ResidualConvBlock
    down1: eqx.nn.Conv2d
    down1_norm: eqx.nn.GroupNorm
    encoder2: ResidualConvBlock
    down2: eqx.nn.Conv2d
    down2_norm: eqx.nn.GroupNorm
    bottleneck: ResidualConvBlock
    up2: eqx.nn.ConvTranspose2d
    merge2: eqx.nn.Conv2d
    merge2_norm: eqx.nn.GroupNorm
    decoder2: ResidualConvBlock
    up1: eqx.nn.ConvTranspose2d
    merge1: eqx.nn.Conv2d
    merge1_norm: eqx.nn.GroupNorm
    decoder1: ResidualConvBlock
    embedding_head: eqx.nn.Conv2d
    species: tuple[str, ...] | None = eqx.field(static=True, default=None, kw_only=True)
    patch_microbatch_size: int | None = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)
    base_channels: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        *,
        species: Sequence[str] | None = None,
        patch_size: int = 32,
        base_channels: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        patch_microbatch_size: int | None = None,
        key: jax.Array,
    ):
        in_channels = _positive_integer("in_channels", in_channels)
        species = _species_names(species)
        patch_size = _positive_integer("patch_size", patch_size)
        base_channels = _positive_integer("base_channels", base_channels)
        embedding_dim = _positive_integer("embedding_dim", embedding_dim)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        patch_microbatch_size = _positive_integer(
            "patch_microbatch_size", patch_microbatch_size, optional=True
        )
        keys = jax.random.split(key, 13)
        stage2, bottleneck = 2 * base_channels, 4 * base_channels
        self.patch_embedding = eqx.nn.Linear(
            in_channels * patch_size * patch_size,
            base_channels,
            dtype=jnp.float32,
            key=keys[0],
        )
        self.patch_norm = eqx.nn.GroupNorm(min(8, base_channels), base_channels, dtype=jnp.float32)
        self.encoder1 = ResidualConvBlock(
            base_channels, base_channels, dropout=dropout, key=keys[1]
        )
        self.down1 = eqx.nn.Conv2d(
            base_channels,
            stage2,
            3,
            stride=2,
            padding=1,
            dtype=jnp.float32,
            key=keys[2],
        )
        self.down1_norm = eqx.nn.GroupNorm(min(8, stage2), stage2, dtype=jnp.float32)
        self.encoder2 = ResidualConvBlock(stage2, stage2, dropout=dropout, key=keys[3])
        self.down2 = eqx.nn.Conv2d(
            stage2,
            bottleneck,
            3,
            stride=2,
            padding=1,
            dtype=jnp.float32,
            key=keys[4],
        )
        self.down2_norm = eqx.nn.GroupNorm(min(8, bottleneck), bottleneck, dtype=jnp.float32)
        self.bottleneck = ResidualConvBlock(bottleneck, bottleneck, dropout=dropout, key=keys[5])
        self.up2 = eqx.nn.ConvTranspose2d(
            bottleneck, stage2, 2, stride=2, dtype=jnp.float32, key=keys[6]
        )
        self.merge2 = eqx.nn.Conv2d(
            2 * stage2, stage2, 3, padding=1, dtype=jnp.float32, key=keys[7]
        )
        self.merge2_norm = eqx.nn.GroupNorm(min(8, stage2), stage2, dtype=jnp.float32)
        self.decoder2 = ResidualConvBlock(stage2, stage2, dropout=dropout, key=keys[8])
        self.up1 = eqx.nn.ConvTranspose2d(
            stage2, base_channels, 2, stride=2, dtype=jnp.float32, key=keys[9]
        )
        self.merge1 = eqx.nn.Conv2d(
            2 * base_channels,
            base_channels,
            3,
            padding=1,
            dtype=jnp.float32,
            key=keys[10],
        )
        self.merge1_norm = eqx.nn.GroupNorm(min(8, base_channels), base_channels, dtype=jnp.float32)
        self.decoder1 = ResidualConvBlock(
            base_channels, base_channels, dropout=dropout, key=keys[11]
        )
        self.embedding_head = eqx.nn.Conv2d(
            base_channels,
            (1 if species is None else len(species)) * embedding_dim,
            1,
            dtype=jnp.float32,
            key=keys[12],
        )
        self.species = species
        self.patch_microbatch_size = patch_microbatch_size
        self.patch_size = patch_size
        self.base_channels = base_channels
        self.embedding_dim = embedding_dim

    def backbone(self, features: jax.Array) -> jax.Array:
        """Patch features with a region-wide receptive field."""
        values = encode_patches(
            self,
            features,
            lambda patch: self.patch_embedding(
                patch.reshape(-1).astype(self.patch_embedding.weight.dtype)
            ),
        )
        values = jnp.moveaxis(values, -1, 0)
        values = jax.nn.gelu(self.patch_norm(values))

        skip1 = self.encoder1(values)
        values = jax.nn.gelu(self.down1_norm(self.down1(skip1)))
        skip2 = self.encoder2(values)
        values = jax.nn.gelu(self.down2_norm(self.down2(skip2)))
        values = self.bottleneck(values)
        values = _match_shape(self.up2(values), skip2)
        values = self.merge2_norm(self.merge2(jnp.concatenate([values, skip2], axis=0)))
        values = self.decoder2(jax.nn.gelu(values))
        values = _match_shape(self.up1(values), skip1)
        values = self.merge1_norm(self.merge1(jnp.concatenate([values, skip1], axis=0)))
        values = self.decoder1(jax.nn.gelu(values))
        return jnp.moveaxis(values, 0, -1)

    def embedding(self, latent: jax.Array, species_index: int | jax.Array) -> jax.Array:
        """Embedding grid from patch features."""
        start = species_index * self.embedding_dim
        weights = jax.lax.dynamic_slice_in_dim(
            self.embedding_head.weight, start, self.embedding_dim, axis=0
        )
        assert self.embedding_head.bias is not None
        bias = jax.lax.dynamic_slice_in_dim(
            self.embedding_head.bias, start, self.embedding_dim, axis=0
        )
        return latent @ weights[:, :, 0, 0].T + bias[:, 0, 0]
