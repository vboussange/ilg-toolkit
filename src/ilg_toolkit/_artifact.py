"""Deterministic descriptions for caller-supplied native artifact templates."""

import dataclasses
import functools
import json
import math
import types
from enum import Enum
from typing import Any

import jax
import numpy as np


class ArtifactError(ValueError):
    """Unsupported, incomplete or structurally incompatible native artifact."""


def _symbol(value):
    return f"{value.__module__}.{value.__qualname__}"


def describe(value, *, static=False) -> Any:
    """Describe structure and configuration without recording learned array values."""
    if isinstance(value, (jax.Array, np.ndarray)):
        result = {
            "array": "jax" if isinstance(value, jax.Array) else "numpy",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if static:
            result["values"] = np.asarray(value).tolist()
        return result
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactError("Nonfinite model configuration cannot be persisted")
        return {"float": value}
    if isinstance(value, np.generic):
        return {"scalar_dtype": str(value.dtype), "value": describe(value.item())}
    if isinstance(value, complex):
        return {"complex": [describe(value.real), describe(value.imag)]}
    if isinstance(value, Enum):
        return {"enum": _symbol(type(value)), "name": value.name}
    if isinstance(value, np.dtype):
        return {"dtype": str(value)}
    if isinstance(value, type):
        return {"type": _symbol(value)}
    if dataclasses.is_dataclass(value):
        return {
            "class": _symbol(type(value)),
            "fields": {
                field.name: {
                    "static": bool(field.metadata.get("static", False)),
                    "value": describe(
                        getattr(value, field.name),
                        static=static or bool(field.metadata.get("static", False)),
                    ),
                }
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, (tuple, list)):
        return {
            "sequence": _symbol(type(value)),
            "items": [describe(item, static=static) for item in value],
        }
    if isinstance(value, dict):
        entries = [
            [describe(key, static=True), describe(item, static=static)]
            for key, item in value.items()
        ]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True))
        return {"mapping": _symbol(type(value)), "items": entries}
    if isinstance(value, functools.partial):
        return {
            "partial": describe(value.func),
            "args": describe(value.args, static=True),
            "keywords": describe(value.keywords, static=True),
        }
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        result: dict[str, Any] = {"callable": _symbol(value)}
        bound = getattr(value, "__self__", None)
        if bound is not None and not isinstance(bound, types.ModuleType):
            result["bound"] = describe(bound, static=True)
        if isinstance(value, types.FunctionType):
            if value.__name__ == "<lambda>":
                raise ArtifactError("Use a named callable or callable module in model templates")
            result["defaults"] = describe(value.__defaults__, static=True)
            result["kwdefaults"] = describe(value.__kwdefaults__, static=True)
            result["closure"] = [
                describe(cell.cell_contents, static=True) for cell in value.__closure__ or ()
            ]
        return result
    raise ArtifactError(f"Unsupported model configuration type {_symbol(type(value))}")


def require_equal(saved, supplied, path="distance_model"):
    """Report the first incompatible identity, configuration or numerical leaf shape."""
    if type(saved) is not type(supplied):
        raise ArtifactError(f"Incompatible {path}: changed type")
    if isinstance(saved, dict):
        if saved.keys() != supplied.keys():
            raise ArtifactError(f"Incompatible {path}: changed structure")
        for name in saved:
            require_equal(saved[name], supplied[name], f"{path}.{name}")
    elif isinstance(saved, list):
        if len(saved) != len(supplied):
            raise ArtifactError(f"Incompatible {path}: changed length/shape")
        for index, (first, second) in enumerate(zip(saved, supplied, strict=True)):
            require_equal(first, second, f"{path}[{index}]")
    elif saved != supplied:
        raise ArtifactError(f"Incompatible {path}: saved {saved!r}, supplied {supplied!r}")


def pack(value):
    """Preserve host tuple/list/dictionary structure across JSON metadata."""
    if isinstance(value, dict):
        return {"dict": [[key, pack(value[key])] for key in sorted(value)]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [pack(item) for item in value]}
    return value


def unpack(value) -> Any:
    """Decode only the host containers used by the native input/state contract."""
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ArtifactError("Invalid checkpoint container metadata")
    kind, entries = next(iter(value.items()))
    if not isinstance(entries, list):
        raise ArtifactError("Invalid checkpoint container entries")
    if kind == "dict":
        result = dict((key, unpack(item)) for key, item in entries)
        if len(result) != len(entries) or any(not isinstance(key, str) for key in result):
            raise ArtifactError("Invalid checkpoint mapping keys")
        return result
    if kind == "tuple":
        return tuple(unpack(item) for item in entries)
    if kind == "list":
        return [unpack(item) for item in entries]
    raise ArtifactError("Invalid checkpoint container kind")
