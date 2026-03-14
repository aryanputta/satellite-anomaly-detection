#!/usr/bin/env python3
"""Check accessibility of project public links.

Uses stdlib only. Returns non-zero if any link is unreachable.
"""

from __future__ import annotations

import argparse
import urllib.request

DEFAULT_URLS = [
    "https://kaggle.com/datasets/aryanputta/hybrid-satellite-telemetry",
    "https://github.com/aryanputta/satellite-anomaly-detection",
]


def check(url: str, timeout: int = 20) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, f"HTTP {resp.status}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--url", action="append", default=[])
    args = ap.parse_args()

    urls = args.url or DEFAULT_URLS
    failures = 0
    for url in urls:
        ok, msg = check(url, timeout=args.timeout)
        print(f"{url} -> {'OK' if ok else 'FAIL'} ({msg})")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
