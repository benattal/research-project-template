"""Audit source ownership, naming, and nesting without importing the project.

Usage:

    python scripts/infra/check_source_tree.py [--json]

Exits non-zero when it reports findings. A zero count is necessary, not
sufficient: naming and ownership still need human review
(``docs/style/code.md``).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"

# TODO(template): the project's shared concerns and applications. Adding one
# is a deliberate ownership decision; record it in
# docs/implementation/architecture.md in the same change.
CORE_CONCERNS = {"training", "utils"}
APPLICATIONS = {"example"}

# Subdirectories of an application that may hold a single module, because
# the vertical slice names them even before they grow.
APPLICATION_SLICES = {"config", "entries", "methods", "models", "training"}

GENERIC_FILES = {
    "_common.py", "base.py", "common.py", "core.py", "helpers.py", "misc.py",
    "utils.py",
}

# Import strings retired by a clean break, mapped to their replacement. Add an
# entry when a module moves so stale callers fail this audit.
STALE_IMPORTS: dict[str, str] = {}

MAX_CORE_DEPTH = 4   # core/<concern>/<domain>/file.py
MAX_APP_DEPTH = 5    # apps/<app>/<concern>/<domain>/file.py


def _python_children(directory: Path) -> tuple[list[Path], list[Path]]:
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix == ".py" and path.name != "__init__.py"
    )
    directories = sorted(
        path for path in directory.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    )
    return files, directories


def audit() -> list[dict[str, str]]:
    """Return one finding per violated rule, as ``kind``/``path``/``detail``."""
    findings: list[dict[str, str]] = []

    def add(kind: str, path: Path, detail: str) -> None:
        findings.append({
            "kind": kind,
            "path": path.relative_to(ROOT).as_posix(),
            "detail": detail,
        })

    for path in sorted(SOURCE.iterdir()):
        if path.name in {"core", "apps", "__init__.py", "__pycache__"}:
            continue
        add("unexpected-root", path, "functionality lives in src/core or src/apps")

    core = SOURCE / "core"
    apps = SOURCE / "apps"
    for parent, expected, kind in (
            (core, CORE_CONCERNS, "unexpected-core-concern"),
            (apps, APPLICATIONS, "unexpected-application")):
        actual = {
            path.name for path in parent.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        }
        for name in sorted(actual - expected):
            add(kind, parent / name, f"expected only {sorted(expected)}")

    for path in sorted(SOURCE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.name in GENERIC_FILES:
            add("generic-file-name", path, "rename to the concern it owns")
        source = path.read_text(encoding="utf-8")
        for token, detail in STALE_IMPORTS.items():
            if token in source:
                add("stale-import", path, detail)

        relative = path.relative_to(SOURCE)
        if relative.parts[0] == "core" and len(relative.parts) > MAX_CORE_DEPTH:
            add("deep-core-nesting", path,
                "default maximum is core/<concern>/<domain>/file.py")
        if relative.parts[0] == "apps" and len(relative.parts) > MAX_APP_DEPTH:
            add("deep-app-nesting", path,
                "default maximum is apps/<app>/<concern>/<domain>/file.py")

    for directory in sorted(
        path for path in SOURCE.rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    ):
        files, directories = _python_children(directory)
        init = directory / "__init__.py"
        relative_directory = directory.relative_to(SOURCE)
        application_slice = (
            len(relative_directory.parts) == 3
            and relative_directory.parts[0] == "apps"
            and relative_directory.parts[2] in APPLICATION_SLICES
        )
        if not init.exists() and (files or directories):
            add("missing-package-init", directory,
                "add an __init__.py whose docstring names what the package owns")
        if init.exists() and not files and not directories:
            if len(relative_directory.parts) > 1:
                add("empty-package", directory, "remove the package shell")
        elif (init.exists() and len(files) == 1 and not directories
              and not application_slice):
            add("one-file-package", directory,
                f"fold {files[0].name} into the parent with a descriptive name")

        child_names = defaultdict(list)
        for child in [*files, *directories]:
            stem = child.stem if child.is_file() else child.name
            normalized = stem[:-1] if stem.endswith("s") and len(stem) > 3 else stem
            child_names[normalized].append(child)
        for normalized, children in child_names.items():
            if len(children) > 1:
                names = ", ".join(child.name for child in children)
                add("ambiguous-sibling-names", directory,
                    f"normalized stem {normalized!r}: {names}")

        file_stems = {path.stem for path in files}
        directory_names = {path.name for path in directories}
        for collision in sorted(file_stems & directory_names):
            add("file-directory-name-collision", directory,
                f"both {collision}.py and {collision}/ exist")

        prefix_groups = defaultdict(list)
        for path in files:
            if "_" in path.stem:
                prefix_groups[path.stem.split("_", 1)[0]].append(path.name)
        for prefix, names in sorted(prefix_groups.items()):
            if len(names) > 1:
                add("shared-prefix-siblings", directory,
                    f"combine {', '.join(names)} under {prefix}.py or record "
                    "independent public contracts")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    findings = audit()
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        for finding in findings:
            print(f"{finding['kind']}: {finding['path']}: {finding['detail']}")
        print(f"source-tree findings: {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
