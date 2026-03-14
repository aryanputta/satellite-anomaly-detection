# Hybrid Satellite Telemetry Anomaly Detection Research Report

## Paper Alignment
I prepared this report to keep my dataset narrative and notebook narrative in one place for Kaggle readers. This report aligns with my paper:

**Hybrid Satellite Telemetry Anomaly Detection: A Novel Dataset, Fault Taxonomy, Recurrence-Plot Computer Vision, and Comparative Machine Learning Study**.

## What I publish on Kaggle
I publish:
1. Dataset page content with generation context, taxonomy, split protocol, and benchmark summary.
2. Research notebook that demonstrates my full modeling story, including the four model families in my paper.

## Dataset Summary
- Approximate scale: 100,000 timestamped samples
- Ten satellites (`SAT_01` to `SAT_10`)
- Four channels: `power_w`, `temp_c`, `voltage_v`, `wheel_rpm`
- Fault labels: `POWER_SPIKE`, `THERMAL_DRIFT`, `VOLTAGE_DROP`, `WHEEL_OSCILLATION`, `SENSOR_DROPOUT`

## Evaluation Protocol
I use per-satellite generalization:
- Train: `SAT_01` to `SAT_08`
- Test: `SAT_09` and `SAT_10`

This prevents leakage across runs and measures generalization to unseen satellites.

## Model Coverage in Notebook
I explicitly include all model families I discuss in my research:
- Isolation Forest
- Standard Autoencoder
- LSTM Autoencoder
- CNN Autoencoder on recurrence plots

This addresses the gap where only one or two baseline models are shown.

## Why recurrence-plot computer vision is included
I encode telemetry windows into recurrence-plot images so oscillatory dynamics become spatial textures. This supports the CNN analysis for `WHEEL_OSCILLATION` and connects directly to my paper’s CV contribution.

## Kaggle loading reliability fix
To resolve `not found` import/path issues in Kaggle, my notebook now scans `/kaggle/input/*` for `satellite_runs` automatically instead of hardcoding one mount path.


## What is explicitly shown in my notebook
I make sure my own work is visible and testable in one place:
- my recurrence-plot computer vision encoding workflow
- my per-satellite generalization protocol
- my four-model comparison setup
- my dataset fault taxonomy and class distribution context

## Model comparison snapshot from my paper
| Model | Role in my study | Headline result context |
|---|---|---|
| Isolation Forest | Unsupervised tree baseline | Useful for sharp anomalies |
| Standard Autoencoder | Tabular reconstruction baseline | Mid-tier overall performance |
| LSTM Autoencoder | Sequence reconstruction model | Best overall F1 around 0.84 |
| CNN on Recurrence Plots | Computer vision model on telemetry images | Best on `WHEEL_OSCILLATION` around 0.91 F1 |

## Account and links
Owner: `aryanputta`
- Dataset: https://kaggle.com/datasets/aryanputta/hybrid-satellite-telemetry
- Code: https://github.com/aryanputta/satellite-anomaly-detection
