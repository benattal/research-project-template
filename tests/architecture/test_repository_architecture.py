"""Repository-wide structural boundaries that must not silently regress."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROLES = {
    "archive", "figures", "infra", "install", "launchers", "lib", "pages",
    "probes", "profiling", "regression", "scoring", "verify",
}


def _repository_files(*patterns: str) -> list[Path]:
    """Tracked plus untracked-but-not-ignored files matching ``patterns``."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard",
         "--", *patterns],
        cwd=ROOT, check=True, capture_output=True, text=True)
    return [Path(line) for line in result.stdout.splitlines()]


def _ignored(relative: str) -> bool:
    return subprocess.run(
        ["git", "check-ignore", "-q", relative], cwd=ROOT).returncode == 0


def test_tests_are_grouped_by_verification_scope() -> None:
    tests = ROOT / "tests"
    assert not list(tests.glob("test_*.py"))
    assert {
        path.name for path in tests.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    } == {"architecture", "unit", "integration"}


def test_scripts_are_filed_by_role() -> None:
    loose = [
        path for path in _repository_files("scripts/*")
        if path.parent == Path("scripts") and path.name != "README.md"
    ]
    assert not loose, "\n".join(map(str, loose))
    roles = {
        path.name for path in (ROOT / "scripts").iterdir() if path.is_dir()
        and path.name != "__pycache__"
    }
    assert roles <= SCRIPT_ROLES, sorted(roles - SCRIPT_ROLES)


def test_generated_state_cannot_be_staged_as_source() -> None:
    for relative in (
            "outputs/run/config.json", "logs/job_1.out", "docs/pages/site/index.html",
            "data/resources/weights.pt", "results.npy", "arrays.npz", ".data-credentials"):
        assert _ignored(relative), relative
    arrays = [path for path in _repository_files("*.npy", "*.npz") if (ROOT / path).exists()]
    assert not arrays, "\n".join(map(str, arrays))


def test_log_directory_placeholder_is_tracked() -> None:
    assert (ROOT / "logs/README.md").is_file()
    assert not _ignored("logs/README.md")


def test_every_source_module_describes_its_owner() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
        and not ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    ]
    assert not missing, "\n".join(missing)


def test_production_source_never_imports_scripts() -> None:
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if re.search(r"^\s*(from|import)\s+scripts\b",
                     path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert not offenders, "\n".join(offenders)


def test_documented_repository_paths_resolve() -> None:
    documents = [ROOT / name for name in ("AGENTS.md", "CLAUDE.md", "README.md", "DATA.md")]
    documents += [
        path for path in (ROOT / "docs").rglob("*.md")
        if "pages" not in path.relative_to(ROOT / "docs").parts
    ]
    pattern = re.compile(r"`((?:src|scripts|data|docs|tests|third_party)/[^`\s,:;)]+)")
    missing = []
    for document in documents:
        for raw in pattern.findall(document.read_text(encoding="utf-8")):
            relative = raw.rstrip(".,").split("#", 1)[0]
            if (any(char in relative for char in "*{}<>")
                    or relative.startswith(("docs/pages/", "data/resources/"))
                    or relative.endswith("...")):
                continue
            if not (ROOT / relative).exists():
                missing.append(f"{document.relative_to(ROOT)}: {relative}")
    assert not missing, "\n".join(missing)


def test_agent_guidance_is_durable_and_synchronized() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for source in (agents, claude):
        assert not re.search(r"\b(?:job|SLURM)\s+\d{5,}\b", source)
        assert not re.search(r"outputs/[^\s`]*_\d{8}_\d{6}", source)
        assert "docs/style/code.md" in source
        assert "docs/environments/index.md" in source

    shared_start = "Do not put live job status"
    operating = "## Operating in this repository"
    claude_common = claude[claude.index(shared_start):]
    agents_common = agents[agents.index(shared_start):agents.index(operating)]
    assert agents_common.rstrip() == claude_common.rstrip()


def test_job_cache_cleanup_accepts_every_directory_it_creates() -> None:
    source = (ROOT / "scripts/lib/job_cache.sh").read_text(encoding="utf-8")
    created = re.findall(r'\$\{root\}/([a-z_]+)_\$\{tag\}', source)
    assert created
    for prefix in created:
        assert f"{prefix}_*" in source, prefix
