#!/usr/bin/env python3
"""Audit the exact Git index; optionally compare known local credentials privately."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(
    r"(^|/)(?:\.env(?:\..*)?|data|private|\.private|\.venv|__pycache__|\.pi|plans|publication-audit)(/|$)|\.(?:sqlite3(?:-.*)?|pem|key|log)$"
)
PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    "provider-token": re.compile(
        rb"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|rpa_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
    "local-user-path": re.compile(rb"/(?:Users|home)/[a-zA-Z0-9_.-]+/"),
    "credential-url": re.compile(rb"https?://[^\s/\"']+:[^\s/@\"']+@"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Compare local .env values and model token without printing them"
    )
    args = parser.parse_args()
    known = {}
    for key, value in os.environ.items():
        if re.search(r"(?:TOKEN|SECRET|API_KEY|PASSWORD)", key, re.I) and len(value) >= 12:
            known[key] = value.encode()
    if args.source:
        for path in args.source.glob(".env*"):
            if path.is_file():
                for line in path.read_text().splitlines():
                    match = re.match(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
                    if match:
                        key, value = match.groups()
                        value = value.strip().strip("\"'")
                        if len(value) >= 12:
                            known[f"local-env:{key}"] = value.encode()
        token = args.source / "benchmarks/data/private/hf_token"
        if token.is_file() and len(token.read_bytes().strip()) >= 12:
            known["local-model-token"] = token.read_bytes().strip()
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    findings = []
    count = 0
    for name in filter(None, names):
        count += 1
        if FORBIDDEN.search(name):
            findings.append((name, "forbidden-path"))
        content = subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append((name, label))
        for label, secret in known.items():
            if secret in content:
                findings.append((name, f"known-local-value:{label}"))
    # Report labels/paths only, never matched credential values.
    print(
        json.dumps(
            {"files_checked": count, "known_local_values_checked": len(known), "findings": findings}, indent=2
        )
    )
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
