#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Tuple


DEFAULT_LOG_GLOBS = [
    "*.o", "*.e", "*.out", "*.err", "*.log",
    "slurm-*.out", "slurm-*.err",
]
DEFAULT_SCRATCH_GLOBS = [
    "*_run.sh", "*.sh",
]


def _collect_files(base: Path, subdir: str, globs: List[str]) -> List[Path]:
    p = base / subdir
    if not p.exists() or not p.is_dir():
        return []
    files: List[Path] = []
    for g in globs:
        files.extend(sorted(p.glob(g)))
    # de-dupe while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen and f.is_file():
            seen.add(f)
            uniq.append(f)
    return uniq


def _delete(files: Iterable[Path], dry_run: bool) -> Tuple[int, int]:
    deleted = 0
    failed = 0
    for f in files:
        try:
            if dry_run:
                print(f"[dry-run] rm {f}")
            else:
                f.unlink()
                print(f"rm {f}")
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"[WARN] failed to delete {f}: {e}")
    return deleted, failed


def main():
    ap = argparse.ArgumentParser(description="Remove HPC scratch .sh scripts and log files under a writepath.")
    ap.add_argument("--w", dest="writepath", type=str, required=True,
                    help="Writepath containing scratch/ and logs/ (e.g., /project/.../output_shared/ripper_gecco)")
    ap.add_argument("--dry_run", action="store_true", help="Print what would be deleted without deleting.")
    ap.add_argument("--yes", action="store_true", help="Do not prompt (non-interactive).")

    ap.add_argument("--no_logs", action="store_true", help="Skip deleting logs/")
    ap.add_argument("--no_scratch", action="store_true", help="Skip deleting scratch/")

    args = ap.parse_args()

    writepath = Path(args.writepath).expanduser().resolve()
    if not writepath.exists():
        raise FileNotFoundError(f"writepath not found: {writepath}")

    log_files = [] if args.no_logs else _collect_files(writepath, "logs", DEFAULT_LOG_GLOBS)
    scratch_files = [] if args.no_scratch else _collect_files(writepath, "scratch", DEFAULT_SCRATCH_GLOBS)

    total = len(log_files) + len(scratch_files)
    print(f"Target writepath: {writepath}")
    print(f"Found {len(scratch_files)} scratch file(s) to delete in {writepath/'scratch'}")
    print(f"Found {len(log_files)} log file(s) to delete in {writepath/'logs'}")
    print(f"Total: {total}")

    if total == 0:
        print("Nothing to delete.")
        return 0

    if not args.yes and not args.dry_run:
        resp = input("Proceed with deletion? (y/N): ").strip().lower()
        if resp not in ("y", "yes"):
            print("Cancelled.")
            return 0

    d1, f1 = _delete(scratch_files, args.dry_run)
    d2, f2 = _delete(log_files, args.dry_run)

    print(f"Deleted: {d1 + d2}, Failed: {f1 + f2}")
    return 0 if (f1 + f2) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
