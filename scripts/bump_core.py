#!/usr/bin/env python3
"""
bump_core.py — Update candlelab-core version pin in both downstream repos.

Usage:
    python scripts/bump_core.py v1.0.1

Steps:
    1. Validate tag argument format (must start with 'v')
    2. Verify tag exists on remote GitHub repo
    3. Update candlelab/requirements.txt
    4. Git add/commit/push candlelab
    5. Update oanda-trading/requirements.txt
    6. Git add/commit/push oanda-trading
    7. Print success summary
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CORE_REPO = "https://github.com/theonlykk/candlelab-core.git"
CANDLELAB_REQ = Path(r"D:\candlelab\requirements.txt")
OANDA_REQ = Path(r"D:\oanda-trading\requirements.txt")
CANDLELAB_DIR = Path(r"D:\candlelab")
OANDA_DIR = Path(r"D:\oanda-trading")


def _parse_and_validate() -> str:
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/bump_core.py <tag>",
            file=sys.stderr,
        )
        print("  <tag> must start with 'v' (e.g. v1.0.1)", file=sys.stderr)
        sys.exit(1)
    tag = sys.argv[1].strip()
    if not tag.startswith("v"):
        print(
            "Usage: python scripts/bump_core.py <tag>",
            file=sys.stderr,
        )
        print("  Error: tag must start with 'v'", file=sys.stderr)
        sys.exit(1)
    return tag


def _check_paths_exist() -> None:
    missing = [p for p in (CANDLELAB_REQ, OANDA_REQ) if not p.is_file()]
    if missing:
        for p in missing:
            print(f"ERROR: requirements file not found: {p}", file=sys.stderr)
        sys.exit(1)


def _verify_remote_tag(tag: str) -> None:
    proc = subprocess.run(
        ["git", "ls-remote", "--tags", CORE_REPO, tag],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end="")
        sys.exit(1)
    if not proc.stdout.strip():
        print(
            f"ERROR: Tag '{tag}' not found on remote. "
            "Did you forget to run 'git push --tags'?",
            file=sys.stderr,
        )
        sys.exit(1)


def _update_requirements(req_path: Path, tag: str) -> None:
    with open(req_path, encoding="utf-8") as f:
        lines = f.readlines()
    new_line = (
        f"candlelab-core @ git+https://github.com/theonlykk/candlelab-core.git@{tag}\n"
    )
    found = False
    out: list[str] = []
    for line in lines:
        if line.strip().lower().startswith("candlelab-core"):
            out.append(new_line)
            found = True
        else:
            out.append(line)
    if not found:
        print(f"Warning: no candlelab-core line in {req_path} — skipped")
        return
    with open(req_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"Updated {req_path}")


def _git_push(repo_dir: Path, tag: str) -> None:
    commit_msg = f"chore: bump candlelab-core to {tag}"
    for argv in (
        ["git", "add", "requirements.txt"],
        ["git", "commit", "-m", commit_msg],
        ["git", "push"],
    ):
        try:
            subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                cwd=repo_dir,
            )
        except subprocess.CalledProcessError as exc:
            err = exc.stderr or ""
            if err:
                print(err, file=sys.stderr, end="")
            sys.exit(1)
    print(f"Pushed {repo_dir}")


if __name__ == "__main__":
    tag = _parse_and_validate()
    _check_paths_exist()
    _verify_remote_tag(tag)
    _update_requirements(CANDLELAB_REQ, tag)
    _git_push(CANDLELAB_DIR, tag)
    _update_requirements(OANDA_REQ, tag)
    _git_push(OANDA_DIR, tag)
    print(f"\n✓ candlelab-core {tag} deployed to both repos.")
    print(f"  candlelab:     pinned to {tag}")
    print(f"  oanda-trading: pinned to {tag}")
    print("  Railway builds triggered. Watch both dashboards.")
