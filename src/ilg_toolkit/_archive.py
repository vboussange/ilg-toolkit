"""Versioned numeric archives shared by inference and training checkpoints."""

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import tempfile
import zipfile
from pathlib import Path

import jax
import numpy as np


class ArtifactError(ValueError):
    """An unsupported, incomplete or invalid saved artifact."""


def runtime_versions():
    result = {"python": platform.python_version(), "platform": jax.default_backend()}
    for name in ("ilg-toolkit", "jax", "jaxlib", "equinox", "optax", "numpy", "scipy", "jaxscape"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "0.1.0" if name == "ilg-toolkit" else "unknown"
    return result


def require_fields(value, fields, context):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ArtifactError(f"{context}: missing or unknown metadata fields")


def _numeric(value):
    array = np.asarray(value)
    if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
        raise ArtifactError("Numeric payload must have a finite non-object numeric dtype")
    return array


class ArrayWriter:
    """Collect numbered numeric payloads without serializing arbitrary objects."""

    def __init__(self):
        self.values = {}

    def add(self, value):
        name = f"arrays/{len(self.values):06d}.npy"
        self.values[name] = _numeric(value).copy()
        return {"array": name}


def write_archive(path, *, kind, payload, arrays):
    """Atomically replace one complete archive; failures retain the prior file."""
    encoded, table = {}, {}
    for name, array in arrays.values.items():
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        encoded[name] = data
        table[name] = {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = {
        "format": "ilg-toolkit",
        "schema": 1,
        "kind": kind,
        "runtime": runtime_versions(),
        "payload": payload,
        "arrays": table,
    }
    try:
        content = json.dumps(manifest, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactError("Artifact metadata must be finite standard JSON") from error
    destination = Path(path)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", content)
                for name, data in encoded.items():
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError("Manifest contains duplicate JSON keys")
        result[key] = value
    return result


def _constant(value):
    raise ArtifactError(f"Manifest contains nonfinite JSON constant {value}")


def _float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ArtifactError("Manifest contains an out-of-range JSON number")
    return parsed


class ArchiveReader:
    """Context-managed strict manifest and lazy checked numeric payload reader."""

    def __init__(self, path, expected_kind):
        self._zip = None
        try:
            self._zip = zipfile.ZipFile(path)
            names = self._zip.namelist()
            if len(names) != len(set(names)):
                raise ArtifactError("Archive contains duplicate entry names")
            if self._zip.getinfo("manifest.json").file_size > 16 * 1024 * 1024:
                raise ArtifactError("Artifact manifest exceeds the supported metadata size")
            manifest = json.loads(
                self._zip.read("manifest.json"),
                object_pairs_hook=_object,
                parse_constant=_constant,
                parse_float=_float,
            )
            require_fields(
                manifest,
                {"format", "schema", "kind", "runtime", "payload", "arrays"},
                "Artifact manifest",
            )
            if (
                manifest["format"] != "ilg-toolkit"
                or type(manifest["schema"]) is not int
                or manifest["schema"] != 1
            ):
                raise ArtifactError("Unsupported artifact format or schema version")
            if manifest["kind"] != expected_kind:
                raise ArtifactError(
                    f"Expected {expected_kind} artifact, found {manifest['kind']!r}"
                )
            self.payload, self.runtime = manifest["payload"], manifest["runtime"]
            self._table, self._used = manifest["arrays"], set()
            if not isinstance(self.runtime, dict) or not isinstance(self._table, dict):
                raise ArtifactError("Artifact runtime and array table must be mappings")
            require_fields(
                self.runtime,
                {
                    "python",
                    "platform",
                    "ilg-toolkit",
                    "jax",
                    "jaxlib",
                    "equinox",
                    "optax",
                    "numpy",
                    "scipy",
                    "jaxscape",
                },
                "Artifact runtime",
            )
            if any(not isinstance(value, str) or not value for value in self.runtime.values()):
                raise ArtifactError("Artifact runtime versions must be nonempty strings")
            if set(names) != {"manifest.json", *self._table}:
                raise ArtifactError("Archive has missing or undeclared numeric payload entries")
            for name, description in self._table.items():
                if re.fullmatch(r"arrays/[0-9]{6}\.npy", name) is None:
                    raise ArtifactError("Unsupported numeric payload name")
                require_fields(description, {"shape", "dtype", "sha256"}, "Array descriptor")
                shape = description["shape"]
                dtype = np.dtype(description["dtype"])
                if (
                    not isinstance(shape, list)
                    or any(type(n) is not int or n < 0 for n in shape)
                    or dtype.kind not in "biufc"
                ):
                    raise ArtifactError("Invalid numeric array shape or dtype")
                expected = int(np.prod(shape, dtype=object)) * dtype.itemsize
                if not expected <= self._zip.getinfo(name).file_size <= expected + 65536:
                    raise ArtifactError("Numeric payload size does not match declared shape")
        except Exception as error:
            if self._zip is not None:
                self._zip.close()
            if isinstance(error, ArtifactError):
                raise
            raise ArtifactError(f"Cannot read artifact: {error}") from error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._zip.close()

    def array(self, reference, *, shape=None, dtype=None):
        require_fields(reference, {"array"}, "Array reference")
        name = reference["array"]
        if not isinstance(name, str) or name not in self._table:
            raise ArtifactError("Numeric payload reference is missing")
        description = self._table[name]
        if shape is not None and tuple(description["shape"]) != tuple(shape):
            raise ArtifactError(
                f"Numeric payload {name} shape does not match its model/state template"
            )
        if dtype is not None and np.dtype(description["dtype"]) != np.dtype(dtype):
            raise ArtifactError(
                f"Numeric payload {name} dtype does not match its declared component"
            )
        try:
            data = self._zip.read(name)
            if hashlib.sha256(data).hexdigest() != description["sha256"]:
                raise ArtifactError(f"Numeric payload {name} failed its integrity check")
            buffer = io.BytesIO(data)
            array = _numeric(np.load(buffer, allow_pickle=False))
            if buffer.tell() != len(data):
                raise ArtifactError("Numeric payload contains trailing data")
            if list(array.shape) != description["shape"] or array.dtype.str != description["dtype"]:
                raise ArtifactError("Numeric payload header does not match its descriptor")
        except Exception as error:
            if isinstance(error, ArtifactError):
                raise
            raise ArtifactError(f"Cannot read numeric payload {name}: {error}") from error
        self._used.add(name)
        return array

    def finish(self):
        if self._used != set(self._table):
            raise ArtifactError("Artifact contains unused or incomplete numeric payloads")


def read_archive(path, *, expected_kind):
    return ArchiveReader(path, expected_kind)
