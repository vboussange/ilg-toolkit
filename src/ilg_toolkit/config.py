"""Explicit optimization options without study configuration."""

import math
from dataclasses import dataclass


def validate_patch_batch_size(value: int | None) -> None:
    """Validate the optional encoder chunk size."""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("patch_batch_size must be a positive integer or None")


@dataclass(frozen=True)
class FitConfig:
    """Fixed training budget; validation only changes predictor selection.

    A fixed Adam learning rate is used. Epoch zero is included in the reported
    history and is eligible for validation selection. No test data is accepted.
    """

    epochs: int = 100
    learning_rate: float = 0.001
    seed: int = 0
    jit: bool = True

    def __post_init__(self):
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs < 0:
            raise ValueError("epochs must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
