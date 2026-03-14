#!/usr/bin/env python3
"""Local validation checks before Kaggle upload.

Checks:
- required files exist
- notebook is valid JSON and notebook schema basics
- required research sections/keywords are present
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / "kaggle" / "dataset_page.md",
    ROOT / "kaggle" / "benchmark_notebook.ipynb",
    ROOT / "kaggle" / "research_report.md",
]

REQUIRED_NOTEBOOK_TEXT = [
    "find_dataset_root",
    "Model 1: Isolation Forest",
    "Model 2: Standard Autoencoder",
    "Model 3: LSTM Autoencoder",
    "Model 4: CNN Autoencoder on recurrence plots",
    "per-satellite",
]

REQUIRED_DATASET_PAGE_TEXT = [
    "Hybrid Satellite Telemetry Dataset for Fault Type Anomaly Detection",
    "Dataset Structure",
    "Evaluation Protocol",
    "Benchmark Results",
    "SAT_09",
    "SAT_10",
]


def fail(msg: str) -> int:
    print(f"ERROR: {msg}")
    return 1


def main() -> int:
    for f in REQUIRED_FILES:
        if not f.exists():
            return fail(f"Missing required file: {f}")

    nb_path = ROOT / "kaggle" / "benchmark_notebook.ipynb"
    try:
        nb = json.loads(nb_path.read_text())
    except Exception as e:
        return fail(f"Notebook is not valid JSON: {e}")

    if nb.get("nbformat") != 4:
        return fail("Notebook nbformat must be 4")
    if nb.get("nbformat_minor", 0) < 5:
        return fail("Notebook nbformat_minor must be >= 5")

    cells = nb.get("cells", [])
    if not cells:
        return fail("Notebook has no cells")

    text = "\n".join("".join(c.get("source", [])) for c in cells)
    for needle in REQUIRED_NOTEBOOK_TEXT:
        if needle not in text:
            return fail(f"Notebook missing expected content: {needle}")

    ds = (ROOT / "kaggle" / "dataset_page.md").read_text()
    for needle in REQUIRED_DATASET_PAGE_TEXT:
        if needle not in ds:
            return fail(f"Dataset page missing expected content: {needle}")

    report = (ROOT / "kaggle" / "research_report.md").read_text()
    if "aryanputta" not in report:
        return fail("Research report missing owner name aryanputta")

    print("Kaggle artifact validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
