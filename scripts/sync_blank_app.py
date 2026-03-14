#!/usr/bin/env python3
"""Merge files from a source repo into this repo with a 3-month freshness filter.

Features:
- Copies only files updated in last N days (default 90).
- Protects sensitive files by skipping secret-like paths/extensions.
- Creates backup copies before overwriting existing files.
- Detects unresolved merge markers after sync.
- Optional deletion of destination files older than N days.
"""

from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import os
from pathlib import Path
import re
import shutil
import subprocess

SKIP_DIRS = {".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv"}
SKIP_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".crt"}
SKIP_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
CONFLICT_RE = re.compile(r"^(<{7}|={7}|>{7})( .+)?$", re.MULTILINE)


def is_sensitive(path: Path) -> bool:
    return path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES


def run(cmd: list[str], cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def recent_files_from_git(source: Path, since_days: int) -> set[Path]:
    since_expr = f"{since_days} days ago"
    out = run(["git", "log", "--since", since_expr, "--name-only", "--pretty=format:"], source)
    paths = {Path(line) for line in out.splitlines() if line.strip()}
    return {p for p in paths if p.exists() and p.is_file()}


def recent_files_from_mtime(source: Path, since_days: int) -> set[Path]:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=since_days)
    files: set[Path] = set()
    for root, dirs, names in os.walk(source):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in names:
            p = Path(root) / n
            try:
                mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
            except FileNotFoundError:
                continue
            if mtime >= cutoff and p.is_file():
                files.add(p.relative_to(source))
    return files


def detect_conflict_markers(dest: Path) -> list[Path]:
    bad: list[Path] = []
    for root, dirs, files in os.walk(dest):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for n in files:
            p = Path(root) / n
            rel = p.relative_to(dest)
            if rel.parts and rel.parts[0] == ".git":
                continue
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ipynb", ".lock"}:
                continue
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            if CONFLICT_RE.search(text):
                bad.append(rel)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Path to blank-app source folder")
    ap.add_argument("--dest", default=".", help="Destination repo root")
    ap.add_argument("--days", type=int, default=90, help="Keep files updated within this many days")
    ap.add_argument("--delete-older-in-dest", action="store_true", help="Delete destination files older than --days")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    source = Path(args.source).resolve()
    dest = Path(args.dest).resolve()

    if not source.exists() or not source.is_dir():
        print(f"ERROR: source folder not found: {source}")
        return 2

    if (source / ".git").exists():
        try:
            recent = recent_files_from_git(source, args.days)
            recent = {p for p in recent if (source / p).is_file()}
        except Exception:
            recent = recent_files_from_mtime(source, args.days)
    else:
        recent = recent_files_from_mtime(source, args.days)

    recent = {p for p in recent if all(part not in SKIP_DIRS for part in p.parts)}

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dest / ".merge_backups" / ts

    copied = 0
    sensitive_skipped = 0
    for rel in sorted(recent):
        src = source / rel
        dst = dest / rel
        if not src.is_file():
            continue
        if is_sensitive(rel):
            sensitive_skipped += 1
            continue
        if dst.exists() and dst.is_file() and not filecmp.cmp(src, dst, shallow=False):
            bpath = backup_root / rel
            if not args.dry_run:
                bpath.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, bpath)
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1

    deleted = 0
    if args.delete_older_in_dest:
        cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
        keep = set(recent) | {Path("README.md"), Path("kaggle/dataset_page.md"), Path("kaggle/benchmark_notebook.ipynb")}
        for root, dirs, names in os.walk(dest):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for n in names:
                p = Path(root) / n
                rel = p.relative_to(dest)
                if rel.parts and rel.parts[0] == ".git":
                    continue
                if rel in keep or is_sensitive(rel):
                    continue
                mtime = dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)
                if mtime < cutoff:
                    if not args.dry_run:
                        p.unlink(missing_ok=True)
                    deleted += 1

    bad = [] if args.dry_run else detect_conflict_markers(dest)

    print(f"Recent files considered: {len(recent)}")
    print(f"Copied/updated: {copied}")
    print(f"Sensitive skipped: {sensitive_skipped}")
    print(f"Deleted older files: {deleted}")
    if bad:
        print("Conflict markers detected in:")
        for p in bad:
            print(f" - {p}")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
