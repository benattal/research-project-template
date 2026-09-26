"""The source-tree audit reports no findings."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_source_tree_audit_is_clean() -> None:
    checker = runpy.run_path(str(ROOT / "scripts/infra/check_source_tree.py"))
    findings = checker["audit"]()
    assert not findings, "\n".join(
        f"{finding['kind']}: {finding['path']}: {finding['detail']}"
        for finding in findings)
