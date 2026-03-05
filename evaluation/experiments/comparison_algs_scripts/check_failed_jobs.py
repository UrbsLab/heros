#!/usr/bin/env python3
"""
check_failed_jobs.py

Scan an HPC log folder for failed jobs by inspecting stdout/stderr log files.

Typical layout:
  logs/
    jobA.o
    jobA.e
    jobB.out
    jobB.err
    slurm-12345.out

Heuristics:
  - If stderr is non-empty and contains error keywords => failed
  - If stdout/stderr contain fatal keywords => failed
  - If stdout lacks any "success marker" (optional) but contains crash keywords => failed
  - Detect common scheduler failure messages

Outputs:
  - Human-readable summary to stdout
  - Optional JSON report for downstream tooling

Usage:
  python check_failed_jobs.py --log-dir /path/to/logs --json report.json
  python check_failed_jobs.py --log-dir logs --pair-suffixes .o .e
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# --- Tune these to your environment ---
DEFAULT_ERROR_KEYWORDS = [
    # Python / general
    r"Traceback \(most recent call last\):",
    r"\bException\b",
    r"\bERROR\b",
    r"\bFATAL\b",
    r"\bFatal\b",
    r"\bSegmentation fault\b",
    r"\bcore dumped\b",
    r"\bout of memory\b",
    r"\bOOM\b",
    r"\bKilled\b",
    r"\bkilled by signal\b",
    r"\bCUDA error\b",
    r"\bCUBLAS\b",
    r"\bCUDNN\b",
    r"\bnvcc\b.*error",
    # LSF
    r"\bExited with exit code\b",
    r"\bTERM_(MEMLIMIT|RUNLIMIT|CPULIMIT)\b",
    r"\bMEMLIMIT\b",
    r"\bRun time exceeded\b",
    r"\bJob killed\b",
    # Slurm
    r"\bslurmstepd:\b.*error",
    r"\bCANCELLED\b",
    r"\bOUT_OF_MEMORY\b",
    r"\bTIMEOUT\b",
    r"\bNODE_FAIL\b",
    r"\bFAILED\b",
    # PBS/Torque
    r"\bPBS:\b.*Killed",
    r"\bjob killed\b",
]

DEFAULT_SUCCESS_MARKERS = [
    # Put something your jobs print on success (recommended)
    r"\bTRAINING COMPLETE\b",
    r"\bDONE\b",
    r"\bCompleted successfully\b",
]


@dataclass
class LogFinding:
    job_key: str
    stdout_path: Optional[str]
    stderr_path: Optional[str]
    status: str  # "failed" | "suspect" | "ok"
    reasons: List[str]


def read_tail(path: Path, max_bytes: int = 200000) -> str:
    """
    Read up to max_bytes from the end of a file (fast for big logs).
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
            data = f.read()
        # decode with replacement to avoid crashes on binary junk
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"<<ERROR reading file {path}: {e}>>"


def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE | re.MULTILINE) for p in patterns]


def matches_any(text: str, pats: List[re.Pattern]) -> List[str]:
    hits = []
    for p in pats:
        if p.search(text):
            hits.append(p.pattern)
    return hits


def group_logs(
    log_dir: Path,
    pair_suffixes: Tuple[str, str],
    extra_stdout_suffixes: Tuple[str, ...] = (".out", ".o", ".stdout"),
    extra_stderr_suffixes: Tuple[str, ...] = (".err", ".e", ".stderr"),
) -> Dict[str, Dict[str, Path]]:
    """
    Group logs by a "job_key" derived from filename minus suffix.

    Examples:
      foo.o + foo.e => job_key="foo"
      slurm-12345.out (no matching err) => job_key="slurm-12345"
    """
    groups: Dict[str, Dict[str, Path]] = {}

    # Suffix sets for detection
    stdout_sufs = set(extra_stdout_suffixes)
    stderr_sufs = set(extra_stderr_suffixes)

    for p in log_dir.iterdir():
        if not p.is_file():
            continue

        suf = p.suffix
        name = p.name

        # Handle double suffix like .o123? (rare) - skip complexity unless needed
        if suf in stdout_sufs:
            key = name[: -len(suf)]
            groups.setdefault(key, {})["stdout"] = p
        elif suf in stderr_sufs:
            key = name[: -len(suf)]
            groups.setdefault(key, {})["stderr"] = p
        else:
            # Also accept common scheduler default names like "slurm-12345.out"
            # which already matches ".out" above; anything else ignored.
            continue

    # If user gave explicit pair suffixes, ensure those are preferred when present
    out_suf, err_suf = pair_suffixes
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith(out_suf):
            key = p.name[: -len(out_suf)]
            groups.setdefault(key, {})["stdout"] = p
        if p.name.endswith(err_suf):
            key = p.name[: -len(err_suf)]
            groups.setdefault(key, {})["stderr"] = p

    return groups


def classify_job(
    job_key: str,
    stdout_path: Optional[Path],
    stderr_path: Optional[Path],
    error_pats: List[re.Pattern],
    success_pats: List[re.Pattern],
    max_bytes: int,
    treat_nonempty_stderr_as_suspect: bool = True,
) -> LogFinding:
    reasons: List[str] = []

    stdout_txt = read_tail(stdout_path, max_bytes=max_bytes) if stdout_path else ""
    stderr_txt = read_tail(stderr_path, max_bytes=max_bytes) if stderr_path else ""

    # Keyword hits
    err_hits = matches_any(stdout_txt + "\n" + stderr_txt, error_pats)
    if err_hits:
        reasons.append(f"Matched error patterns: {', '.join(err_hits[:8])}" + (" ..." if len(err_hits) > 8 else ""))

    # Success markers (optional)
    success_hits = matches_any(stdout_txt, success_pats) if success_pats else []
    has_success = bool(success_hits)

    # stderr non-empty heuristic
    stderr_nonempty = bool(stderr_txt.strip()) if stderr_path else False
    if stderr_nonempty and treat_nonempty_stderr_as_suspect:
        # Avoid false positives from harmless warnings; keep it as "suspect" unless error keywords hit.
        reasons.append("stderr is non-empty")

    # Determine status
    if err_hits:
        status = "failed"
    elif stderr_nonempty and treat_nonempty_stderr_as_suspect and not has_success:
        status = "suspect"
        reasons.append("no success marker found in stdout")
    else:
        status = "ok"
        if has_success:
            reasons.append(f"Found success marker(s): {', '.join(success_hits[:5])}")

    return LogFinding(
        job_key=job_key,
        stdout_path=str(stdout_path) if stdout_path else None,
        stderr_path=str(stderr_path) if stderr_path else None,
        status=status,
        reasons=reasons,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan HPC log directory for failed jobs.")
    ap.add_argument("--log-dir", required=True, help="Folder containing job logs")
    ap.add_argument("--pair-suffixes", nargs=2, default=[".o", ".e"], help="Preferred stdout/stderr suffixes (e.g. .out .err or .o .e)")
    ap.add_argument("--max-bytes", type=int, default=200000, help="Read up to this many bytes from end of each log file")
    ap.add_argument("--json", dest="json_path", default=None, help="Write JSON report to this path")
    ap.add_argument("--success-marker", action="append", default=None, help="Regex for success marker in stdout (repeatable)")
    ap.add_argument("--error-keyword", action="append", default=None, help="Regex for error keyword (repeatable)")
    ap.add_argument("--strict-stderr", action="store_true", help="Treat any non-empty stderr as failed (instead of suspect)")
    args = ap.parse_args()

    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists() or not log_dir.is_dir():
        raise SystemExit(f"Log dir does not exist or is not a directory: {log_dir}")

    error_patterns = args.error_keyword if args.error_keyword else DEFAULT_ERROR_KEYWORDS
    success_patterns = args.success_marker if args.success_marker is not None else DEFAULT_SUCCESS_MARKERS

    error_pats = compile_patterns(error_patterns)
    success_pats = compile_patterns(success_patterns) if success_patterns else []

    groups = group_logs(log_dir, pair_suffixes=(args.pair_suffixes[0], args.pair_suffixes[1]))

    findings: List[LogFinding] = []
    for job_key, paths in sorted(groups.items()):
        stdout_p = paths.get("stdout")
        stderr_p = paths.get("stderr")
        finding = classify_job(
            job_key=job_key,
            stdout_path=stdout_p,
            stderr_path=stderr_p,
            error_pats=error_pats,
            success_pats=success_pats,
            max_bytes=args.max_bytes,
            treat_nonempty_stderr_as_suspect=not args.strict_stderr,
        )
        findings.append(finding)

    failed = [f for f in findings if f.status == "failed"]
    suspect = [f for f in findings if f.status == "suspect"]
    ok = [f for f in findings if f.status == "ok"]

    print(f"Scanned: {log_dir}")
    print(f"Jobs found: {len(findings)} | failed: {len(failed)} | suspect: {len(suspect)} | ok: {len(ok)}")
    print()

    def print_block(title: str, items: List[LogFinding]) -> None:
        if not items:
            return
        print(title)
        print("-" * len(title))
        for f in items:
            print(f"* {f.job_key}")
            if f.stdout_path:
                print(f"  stdout: {f.stdout_path}")
            if f.stderr_path:
                print(f"  stderr: {f.stderr_path}")
            for r in f.reasons[:6]:
                print(f"  - {r}")
            if len(f.reasons) > 6:
                print("  - ...")
        print()

    print_block("FAILED", failed)
    print_block("SUSPECT (review)", suspect)

    if args.json_path:
        out = {
            "log_dir": str(log_dir),
            "counts": {"total": len(findings), "failed": len(failed), "suspect": len(suspect), "ok": len(ok)},
            "findings": [asdict(f) for f in findings],
        }
        json_path = Path(args.json_path).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON report: {json_path}")

    # exit code for CI-style usage
    return 2 if failed else (1 if suspect else 0)


if __name__ == "__main__":
    raise SystemExit(main())
