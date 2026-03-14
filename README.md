# satellite-anomaly-detection

This repository is organized to match your requested project layout and to support dataset generation, recurrence-plot encoding, fault-type evaluation, and demo workflows.

## Project Structure

```text
satellite-anomaly-detection/
├── data/
│   └── generate_dataset.py
├── encoding/
│   └── recurrence_plot.py
├── evaluation/
│   └── fault_type_eval.py
├── demo/
│   └── live_demo.py
├── noise/
│   └── arduino_noise_capture/
│       ├── noise_capture.ino
│       └── noise_log.csv
├── models/
├── notebooks/
│   └── satellite_anomaly_detection.ipynb
├── kaggle/
│   ├── dataset_page.md
│   ├── benchmark_notebook.ipynb
│   └── research_report.md
├── scripts/
│   ├── sync_blank_app.py
│   └── validate_kaggle_artifacts.py
├── requirements.txt
└── README.md
```
This repo now keeps only the two Kaggle upload artifacts:

- `kaggle/dataset_page.md` (dataset page content)
- `kaggle/benchmark_notebook.ipynb` (notebook upload file)

Owner/account name: **aryanputta**
- Dataset: https://kaggle.com/datasets/aryanputta/hybrid-satellite-telemetry
- Code: https://github.com/aryanputta/satellite-anomaly-detection

## How to import your own code/files here

### Option A: Copy local files directly
From your machine, copy your files into matching folders above.

### Option B: Use git add + commit
1. Put files in the right folders (for example `models/`, `notebooks/`, `noise/arduino_noise_capture/`).
2. Run:
```bash
git add .
git commit -m "Add my models and experiment files"
```

### Option C: Replace notebook with your own
If you want your exact notebook:
1. Save your notebook as `satellite_anomaly_detection.ipynb`
2. Place it at `notebooks/satellite_anomaly_detection.ipynb`
3. Run validation:
```bash
python scripts/validate_kaggle_artifacts.py
python -m json.tool notebooks/satellite_anomaly_detection.ipynb > /tmp/notebook_ok.json
```


### Option D: Paste your `.py` files directly
If you paste code in chat, I can place it directly into the correct file path for you.
For example:
- `evaluation/fault_type_eval.py`
- `data/generate_dataset.py`
- `demo/live_demo.py`
- `encoding/recurrence_plot.py`

I already used this flow for the latest update.

## Quick start
Install dependencies:
```bash
pip install -r requirements.txt
```

Generate dataset:
```bash
python data/generate_dataset.py --output hybrid_satellite_dataset
```

Run validator:
```bash
python scripts/validate_kaggle_artifacts.py
```
Kaggle release assets for:

**Hybrid Satellite Telemetry Anomaly Detection: A Novel Dataset, Fault Taxonomy, Recurrence-Plot Computer Vision, and Comparative Machine Learning Study**

- Dataset: https://kaggle.com/datasets/aryantputta/hybrid-satellite-telemetry
- Code: https://github.com/aryanputta/satellite-anomaly-detection
- `kaggle/dataset_page.md`: Kaggle dataset page draft
- `kaggle/benchmark_notebook.ipynb`: Kaggle benchmark notebook draft
