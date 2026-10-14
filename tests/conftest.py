"""Keep compilation caches scoped to independent test modules."""

from collections.abc import Iterator

import jax
import pytest


@pytest.fixture(scope="module", autouse=True)
def compilation_cache_lifetime() -> Iterator[None]:
    # Preserve reuse within each module while avoiding accumulated executables
    # from unrelated test shapes and architectures across the complete suite.
    yield
    jax.clear_caches()
