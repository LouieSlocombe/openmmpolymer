"""Deriving reproducible random seeds for the several streams a run needs.

A run has one master seed. The thermostat, the initial velocities, the barostat,
packmol and every chain conformer each need their own, and they must not
collide. Deriving them from labels keeps a run reproducible while letting stages
be added without renumbering anything.

The one trap this module exists to avoid: OpenMM reads a seed of ``0`` as "pick
a random one", on both integrators and barostats. A derived seed of zero would
silently destroy the reproducibility the master seed is there to provide, so
:func:`derive_seed` never returns it.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: OpenMM seeds are C ``int``; 0 means "randomise", so the usable range starts
#: at 1 and stops below the 32-bit signed maximum.
_MAX_SEED = 2**31 - 2


def derive_seed(master: int, *labels: str) -> int:
    """Return a stable, non-zero seed for *labels* under *master*.

    Args:
        master: The run's master seed.
        *labels: Names identifying the stream, e.g. ``"thermostat"`` or
            ``"conformer", "7"``.

    Returns:
        An integer in ``[1, 2**31 - 2]``, stable across processes and platforms.
    """
    payload = "\0".join((str(master), *labels)).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest, "big") % _MAX_SEED + 1


def seed_random_stream(obj: Any, seed: int) -> None:
    """Seed an OpenMM object that owns a random stream.

    Args:
        obj: Anything with ``setRandomNumberSeed``, i.e. an integrator or a
            barostat.
        seed: The seed to set. Must be non-zero; :func:`derive_seed` guarantees
            that.

    Raises:
        ValueError: *seed* is zero, which OpenMM would read as "randomise".
    """
    if seed == 0:
        raise ValueError(
            "seed=0 tells OpenMM to choose its own seed, which makes the run "
            "irreproducible. Use derive_seed() to produce one."
        )
    obj.setRandomNumberSeed(seed)
