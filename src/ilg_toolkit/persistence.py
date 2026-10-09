"""Reloadable inference artifacts, separate from resumable training checkpoints."""

from ._archive import ArrayWriter, ArtifactError, read_archive, write_archive
from ._codecs import decode_predictor, encode_predictor
from .predictor import Predictor


def save_predictor(path, predictor: Predictor) -> None:
    """Atomically save a complete predictor for a shipped encoder architecture."""
    if not isinstance(predictor, Predictor):
        raise ArtifactError("save_predictor requires a fitted Predictor")
    try:
        arrays = ArrayWriter()
        payload = encode_predictor(predictor, arrays)
        write_archive(path, kind="predictor", payload=payload, arrays=arrays)
    except (TypeError, KeyError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"Cannot save predictor: {error}") from error


def load_predictor(path) -> Predictor:
    """Load without training data or caller template; refuse incomplete artifacts."""
    try:
        with read_archive(path, expected_kind="predictor") as archive:
            predictor = decode_predictor(archive.payload, archive)
            archive.finish()
            return predictor
    except (TypeError, KeyError, ValueError, OverflowError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(f"Invalid predictor artifact: {error}") from error
