"""Command ``python -m src.apps.example <preset> [overrides]``.

The first positional argument selects a preset from ``EXAMPLE_PRESETS``;
dotted flags override its fields, for example ``--training.iterations 50``.
``--help`` lists the presets.
"""

from __future__ import annotations

from collections.abc import Sequence

import tyro

from src.apps.example.presets import EXAMPLE_PRESETS
from src.apps.example.trainer import train


def main(args: Sequence[str] | None = None) -> None:
    """Parse a preset and its overrides, then train."""
    config = tyro.extras.overridable_config_cli(
        EXAMPLE_PRESETS, args=args,
        description="Fit the example polynomial model.")
    train(config)


if __name__ == "__main__":
    main()
