"""Operation-local numerical pair selections, prepared before JAX transforms."""

from typing import NamedTuple

import jax


class PairSelection(NamedTuple):
    """Unique required distance nodes and pair indices into that node vector."""

    nodes: jax.Array
    left: jax.Array
    right: jax.Array
