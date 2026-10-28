"""Argument checks shared by the public constructors and entry points."""

from numbers import Integral
from typing import Literal, overload

import jax


@overload
def _positive_integer(
    name: str, value: object, *, optional: Literal[False] = False, error: type[Exception] = ...
) -> int: ...


@overload
def _positive_integer(
    name: str, value: object, *, optional: Literal[True], error: type[Exception] = ...
) -> int | None: ...


def _positive_integer(
    name: str, value: object, *, optional: bool = False, error: type[Exception] = ValueError
) -> int | None:
    """Returns ``value`` as an ``int`` after checking that it is a positive integer.

    Booleans are rejected. With ``optional=True``, ``None`` is accepted and returned.

    Args:
        name: Argument name used in the error message.
        value: Value to check.
        optional: Accept ``None``.
        error: Exception type raised for an invalid value.
    """
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise error(f"{name} must be a positive integer" + (" or None" if optional else ""))
    return int(value)


def _require_float64() -> None:
    """Raises a ``RuntimeError`` unless JAX float64 is enabled.

    Training, calibration fitting, prediction, conductance surfaces and loading
    call this check before any computation.
    """
    if not jax.enable_x64.value:
        raise RuntimeError(
            "ILG Toolkit requires float64: call "
            'jax.config.update("jax_enable_x64", True) before creating arrays'
        )
