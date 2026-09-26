"""Run directories and the provenance every run saves.

A ``RunContext`` is the resolved runtime state derived from an immutable
experiment configuration: where the run writes, which job produced it, and
which code it ran. The requested configuration is never mutated to record
these values; they live here and are serialized beside it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments], cwd=REPOSITORY_ROOT, check=True,
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclass(frozen=True)
class RunContext:
    """The output directory and identity of one run.

    Attributes:
        experiment_name: The preset name; also the directory prefix.
        output_dir: ``<output_root>/<experiment>_<YYYYmmdd_HHMMSS>_<job>``.
        job_id: ``SLURM_JOB_ID`` inside a scheduler job, else the process ID.
    """

    experiment_name: str
    output_dir: Path
    job_id: str

    @classmethod
    def create(cls, experiment_name: str,
               output_root: str | Path = "outputs") -> "RunContext":
        """Create a fresh run directory; an existing directory is an error."""
        job_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_root) / f"{experiment_name}_{timestamp}_{job_id}"
        output_dir.mkdir(parents=True, exist_ok=False)
        return cls(experiment_name, output_dir, job_id)

    def write_json(self, name: str, payload: Any) -> Path:
        """Write ``payload`` as indented JSON to ``output_dir / name``."""
        path = self.output_dir / name
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default) + "\n",
            encoding="utf-8")
        return path

    def save_provenance(self, config: Any) -> None:
        """Save the resolved configuration and the code and host that ran it.

        Writes ``config.json`` (the dataclass as executed, including command
        line overrides) and ``provenance.json`` (commit, dirty state, host,
        interpreter, argv, job).
        """
        self.write_json("config.json", dataclasses.asdict(config))
        status = _git("status", "--porcelain")
        self.write_json("provenance.json", {
            "experiment_name": self.experiment_name,
            "job_id": self.job_id,
            "commit": _git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "argv": sys.argv,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })
