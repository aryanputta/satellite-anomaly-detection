#!/usr/bin/env python3
"""Audit that core paper claims are represented in project artifacts."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "dataset_page": ROOT / "kaggle" / "dataset_page.md",
    "report": ROOT / "kaggle" / "research_report.md",
    "notebook": ROOT / "notebooks" / "satellite_anomaly_detection.ipynb",
    "eval": ROOT / "evaluation" / "fault_type_eval.py",
    "generator": ROOT / "data" / "generate_dataset.py",
}

REQUIRED = {
    "dataset_page": [
        "100,000",
        "POWER_SPIKE",
        "THERMAL_DRIFT",
        "VOLTAGE_DROP",
        "WHEEL_OSCILLATION",
        "SENSOR_DROPOUT",
        "SAT_09",
        "SAT_10",
        "0.91",
        "0.84",
        "6.5%",
        "CC BY 4.0",
    ],
    "report": [
        "Recurrence-Plot Computer Vision",
        "SAT_01",
        "SAT_10",
        "Isolation Forest",
        "Standard Autoencoder",
        "LSTM Autoencoder",
        "CNN Autoencoder",
        "aryanputta",
    ],
    "notebook": [
        "find_dataset_root",
        "Model 1: Isolation Forest",
        "Model 2: Standard Autoencoder",
        "Model 3: LSTM Autoencoder",
        "Model 4: CNN Autoencoder on recurrence plots",
        "WHEEL_OSCILLATION",
    ],
    "eval": [
        "--all-models",
        "TRAIN_SATS",
        "TEST_SATS",
        "TABLE II",
        "TABLE III",
        "metrics.json",
    ],
    "generator": [
        "NASA-HDBK-7004C",
        "N_SATELLITES",
        "hybrid_satellite_telemetry.csv",
        "noise_log.csv",
    ],
}


def main() -> int:
    failed = False
    for key, path in FILES.items():
        if not path.exists():
            print(f"ERROR [{key}] missing file: {path}")
            failed = True
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        missing = [token for token in REQUIRED[key] if token not in text]
        if missing:
            failed = True
            print(f"ERROR [{key}] missing tokens: {missing}")
        else:
            print(f"OK [{key}] all required paper-coverage tokens found")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
