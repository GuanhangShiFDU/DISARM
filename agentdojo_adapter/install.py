#!/usr/bin/env python3
"""Install the DISARM overlay into a clean AgentDojo checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_AGENTDOJO_COMMIT = "5cea5891fa8e6b13c4299a94691e1ec64d445fcd"
PATCH_NAME = "agentdojo-v0.1.35.patch"


def _run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_output(target: Path, *args: str) -> str:
    return _run(["git", *args], cwd=target).stdout.strip()


def _patch_check(target: Path, patch: Path, *, reverse: bool = False) -> bool:
    command = ["git", "apply", "--check"]
    if reverse:
        command.append("--reverse")
    command.append(str(patch))
    return _run(command, cwd=target, check=False).returncode == 0


def _same_file(left: Path, right: Path) -> bool:
    return left.is_file() and right.is_file() and left.read_bytes() == right.read_bytes()


def _overlay_is_current(source_dir: Path, destination_dir: Path, metrics_source: Path, metrics_target: Path) -> bool:
    source_files = sorted(source_dir.glob("*.py"))
    return bool(source_files) and all(
        _same_file(source, destination_dir / source.name) for source in source_files
    ) and _same_file(metrics_source, metrics_target)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy DISARM into AgentDojo and apply the minimal integration patch."
    )
    parser.add_argument("agentdojo", type=Path, help="Path to a clean AgentDojo checkout")
    parser.add_argument("--dry-run", action="store_true", help="Validate without changing files")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Allow a different AgentDojo commit when the patch still applies cleanly",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow installation into a checkout with existing local changes",
    )
    args = parser.parse_args()

    target = args.agentdojo.expanduser().resolve()
    release_root = Path(__file__).resolve().parents[1]
    source_dir = release_root / "disarm"
    patch = Path(__file__).resolve().with_name(PATCH_NAME)
    metrics_source = Path(__file__).resolve().parent / "files" / "metrics.py"
    destination_dir = target / "src" / "agentdojo" / "agent_pipeline" / "disarm"
    metrics_target = target / "src" / "agentdojo" / "agent_pipeline" / "metrics.py"

    required = [
        target / ".git",
        target / "src" / "agentdojo" / "agent_pipeline" / "agent_pipeline.py",
        target / "src" / "agentdojo" / "benchmark.py",
        target / "src" / "agentdojo" / "task_suite" / "task_suite.py",
        source_dir,
        patch,
        metrics_source,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("required paths are missing:\n  " + "\n  ".join(missing))

    head = _git_output(target, "rev-parse", "HEAD")
    if head != SUPPORTED_AGENTDOJO_COMMIT and not args.allow_unsupported:
        parser.error(
            "unsupported AgentDojo commit "
            f"{head}. Checkout {SUPPORTED_AGENTDOJO_COMMIT}, or rerun with "
            "--allow-unsupported after reviewing the patch."
        )

    dirty = _git_output(target, "status", "--porcelain")
    already_patched = _patch_check(target, patch, reverse=True)
    already_current = already_patched and _overlay_is_current(
        source_dir, destination_dir, metrics_source, metrics_target
    )
    if already_current:
        print("DISARM is already installed and matches this release.")
        return 0
    if dirty and not args.allow_dirty:
        parser.error(
            "the AgentDojo checkout has local changes. Use a clean checkout or "
            "review them and rerun with --allow-dirty."
        )
    if already_patched:
        parser.error(
            "the integration patch is already present but the overlay differs. "
            "Review the checkout before replacing files."
        )
    if not _patch_check(target, patch):
        parser.error(
            "the integration patch does not apply cleanly. Confirm the AgentDojo "
            "commit and inspect agentdojo_adapter/agentdojo-v0.1.35.patch."
        )

    planned = [destination_dir / path.name for path in sorted(source_dir.glob("*.py"))]
    planned.append(metrics_target)
    if args.dry_run:
        print(f"AgentDojo commit: {head}")
        print(f"Patch check: OK ({patch.name})")
        print("Files to install:")
        for path in planned:
            print(f"  {path.relative_to(target)}")
        return 0

    _run(["git", "apply", str(patch)], cwd=target)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*.py")):
        shutil.copy2(source, destination_dir / source.name)
    shutil.copy2(metrics_source, metrics_target)

    print(f"Installed DISARM into {target}")
    print("Next: uv sync")
    print(
        "Smoke test: uv run python -m agentdojo.scripts.benchmark "
        "--model gpt-4o-2024-05-13 --attack important_instructions "
        "--defense disarm --suite slack --user-task user_task_0 "
        "--injection-task injection_task_0"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
