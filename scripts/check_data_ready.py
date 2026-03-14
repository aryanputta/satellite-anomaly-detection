#!/usr/bin/env python3
"""Lightweight CSV readiness check without external dependencies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED_COLS = [
    'satellite_id',
    'timestamp',
    'power_w',
    'temp_c',
    'voltage_v',
    'wheel_rpm',
    'anomaly',
    'fault_type',
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/hybrid_satellite_telemetry.csv')
    ap.add_argument('--min-rows', type=int, default=100000)
    args = ap.parse_args()

    p = Path(args.csv)
    if not p.exists():
        print(f'ERROR: missing dataset CSV: {p}')
        return 1

    with p.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in header]
        if missing:
            print(f'ERROR: missing required columns: {missing}')
            return 2

        row_count = 0
        sat_values = set()
        for row in reader:
            row_count += 1
            sat_values.add(row.get('satellite_id', ''))

    print(f'CSV: {p}')
    print(f'Rows: {row_count:,}')
    print(f'Unique satellites: {len([s for s in sat_values if s])}')
    if row_count < args.min_rows:
        print(f'WARNING: row count below expected minimum {args.min_rows:,}')
        return 3

    print('Dataset readiness check passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
