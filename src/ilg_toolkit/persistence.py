"""Reloadable inference artifacts, separate from resumable training checkpoints."""

from ._archive import ArrayWriter, ArtifactError, read_archive, write_archive
from ._codecs import decode_model, encode_model
from .model import CalibratedModel


def save_model(path, model: CalibratedModel) -> None:
    """Atomically save a complete model for a shipped encoder architecture."""
    if not isinstance(model, CalibratedModel):
        raise ArtifactError("save_model requires a fitted CalibratedModel")
    try:
        arrays = ArrayWriter()
        payload = encode_model(model, arrays)
        # Schema one uses this historical artifact tag independently of API names.
        write_archive(path, kind="predictor", payload=payload, arrays=arrays)
    except (TypeError, KeyError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"Cannot save model: {error}") from error


def load_model(path) -> CalibratedModel:
    """Load without training data or caller template; refuse incomplete artifacts."""
    try:
        with read_archive(path, expected_kind="predictor") as archive:
            model = decode_model(archive.payload, archive)
            archive.finish()
            return model
    except (TypeError, KeyError, ValueError, OverflowError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"Invalid model artifact: {error}") from error
