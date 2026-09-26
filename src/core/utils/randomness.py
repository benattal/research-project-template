"""Named random domains derived from one base seed.

Every stochastic consumer (training data, initialization, evaluation,
diagnostics) draws from a stream identified by a name. A stream's state
depends only on the base seed and its name, never on how many other streams
exist or the order in which they were created. Adding a diagnostic stream
therefore cannot shift a training stream, and a repeated run with the same
base seed reproduces every stream.

A stream persists for the life of its ``RandomDomains``: consecutive draws
from it are fresh samples, never a restart at the same seed.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    # Stable across processes, unlike hash(), which is salted per process.
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")


class RandomDomains:
    """The set of named random streams of one run."""

    def __init__(self, base_seed: int) -> None:
        if base_seed < 0:
            raise ValueError("base_seed must be non-negative")
        self.base_seed = base_seed
        self._streams: dict[str, np.random.Generator] = {}

    def _sequence(self, name: str, replicate: int | None = None) -> np.random.SeedSequence:
        entropy = [self.base_seed, _name_key(name)]
        if replicate is not None:
            entropy.append(replicate)
        return np.random.SeedSequence(entropy)

    def stream(self, name: str) -> np.random.Generator:
        """Return the persistent generator of domain ``name``."""
        if name not in self._streams:
            self._streams[name] = np.random.Generator(
                np.random.PCG64(self._sequence(name)))
        return self._streams[name]

    def integer_seed(self, name: str) -> int:
        """A 63-bit seed for a framework that takes an integer.

        Use it once per domain to seed, for example, a ``torch.Generator``;
        the framework's generator then owns the stream.
        """
        return int(self._sequence(name).generate_state(1, np.uint64)[0] >> np.uint64(1))

    def replay(self, name: str, replicate: int) -> np.random.Generator:
        """A new generator that repeats the same draws for the same replicate.

        Control-only: finite differences and replay diagnostics use it to
        reproduce a draw on purpose. Production code uses ``stream``.
        """
        return np.random.Generator(np.random.PCG64(self._sequence(name, replicate)))
