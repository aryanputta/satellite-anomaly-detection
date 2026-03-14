# Hybrid Satellite Telemetry Dataset for Fault Type Anomaly Detection

## Research Motivation
I built this dataset to address a gap in satellite telemetry anomaly detection research. Satellite systems continuously stream health data, but most public benchmarks only provide binary anomaly labels. In operations, binary labels are not enough. I need fault-type labels to diagnose subsystem behavior and prioritize response actions.

This dataset supports fault-type anomaly detection and reproducible benchmarking. I also evaluate whether injecting real hardware sensor noise improves model generalization compared with purely simulated telemetry.

## Dataset Overview
I provide approximately 100,000 timestamped telemetry records from ten simulated satellite runs.

Each record includes four telemetry channels:
- `power_w`
- `temp_c`
- `voltage_v`
- `wheel_rpm`

Each telemetry window is labeled with one of five anomaly classes:
- `POWER_SPIKE`
- `THERMAL_DRIFT`
- `VOLTAGE_DROP`
- `WHEEL_OSCILLATION`
- `SENSOR_DROPOUT`

I designed this dataset to pair with my paper and open-source code release so other researchers can reproduce the generation and evaluation pipeline end to end.

## Hybrid Data Generation
I generated baseline telemetry using orbital dynamics and subsystem simulation for solar power output, internal thermal behavior, battery bus voltage, and reaction wheel speed.

To improve physical realism, I captured real sensor noise on an Arduino Uno using:
- DHT22 temperature sensor
- MPU6050 inertial measurement unit
- Voltage divider circuit for voltage measurements

I observed non-Gaussian properties in captured hardware noise:
- DHT22 thermal signal autocorrelation is approximately 0.72 at lag-1.
- MPU6050 vibration includes a resonance peak near 5 Hz.

I inject these measured noise components into the simulation pipeline. In my reported experiments, models trained with hybrid simulation plus hardware noise improved mean F1 by approximately 6.5% compared with simulation-only training.

## Fault Taxonomy
### `POWER_SPIKE`
A sudden increase in power output that may indicate instability in the solar array or power subsystem.

### `THERMAL_DRIFT`
A gradual temperature rise suggesting thermal regulation degradation.

### `VOLTAGE_DROP`
A rapid battery bus voltage decrease indicating possible power system malfunction.

### `WHEEL_OSCILLATION`
Oscillatory reaction wheel RPM behavior representing instability in attitude control.

### `SENSOR_DROPOUT`
Temporary telemetry loss caused by sensor malfunction or readout interruption.

I model `WHEEL_OSCILLATION` using oscillatory behavior consistent with pre-failure reaction wheel telemetry signatures discussed in mission literature, including Kepler failure context.


## My Core Contributions
I highlight the following contributions in this dataset release:
- A fault-typed telemetry dataset for satellite anomaly diagnosis, not only binary anomaly tags.
- A hybrid simulation plus hardware-noise data design grounded in measured Arduino sensor behavior.
- A recurrence-plot computer vision framing for telemetry windows.
- A comparative four-model evaluation setup (Isolation Forest, Standard AE, LSTM AE, CNN on RP).

## Kaggle Upload Checklist
Before publishing, I verify:
1. Dataset files are uploaded with the exact `hybrid_satellite_dataset/` directory layout.
2. `satellite_runs/sat_01.csv` to `sat_10.csv` are present.
3. `hardware_noise/*.csv` and `metadata/*.csv` are present.
4. Notebook input points to this dataset and runs end to end.
5. Dataset description includes split protocol and benchmark summary.

## Dataset Structure
```text
hybrid_satellite_dataset/

satellite_runs/
  sat_01.csv
  sat_02.csv
  sat_03.csv
  sat_04.csv
  sat_05.csv
  sat_06.csv
  sat_07.csv
  sat_08.csv
  sat_09.csv
  sat_10.csv

hardware_noise/
  dht22_noise.csv
  imu_noise.csv
  voltage_noise.csv

metadata/
  data_dictionary.csv
  fault_definitions.csv
```

## Data Dictionary
- `power_w`: Solar array output power in watts
- `temp_c`: Internal spacecraft temperature in degrees Celsius
- `voltage_v`: Battery bus voltage in volts
- `wheel_rpm`: Reaction wheel speed in revolutions per minute
- `fault_type`: Fault label for the telemetry window

## Evaluation Protocol
I use a strict per-satellite generalization split to avoid leakage across simulated runs.

Training satellites:
- `SAT_01`
- `SAT_02`
- `SAT_03`
- `SAT_04`
- `SAT_05`
- `SAT_06`
- `SAT_07`
- `SAT_08`

Testing satellites:
- `SAT_09`
- `SAT_10`

This split evaluates whether a model trained on one set of satellites can generalize to unseen satellites.

## Benchmark Results
I report the following headline results from the paper:
- CNN autoencoder on recurrence plots achieves about 0.91 F1 on `WHEEL_OSCILLATION`.
- LSTM autoencoder provides the best overall multi-fault performance at about 0.84 F1.
- Hybrid hardware-noise training improves performance by about 6.5% F1 on average across architectures.



## Research Notebook Scope
My Kaggle notebook is designed as a research companion, not only a minimal benchmark. I include sections for all four model families discussed in my paper:
- Isolation Forest
- Standard Autoencoder
- LSTM Autoencoder
- CNN Autoencoder on recurrence plots

I also use automatic `/kaggle/input` path detection in the notebook to avoid dataset mount path errors that can cause `file not found` issues.

## Repository Link
I publish the full pipeline, models, recurrence plot scripts, and evaluation code in this repository:

https://github.com/aryanputta/satellite-anomaly-detection

Dataset page:
https://kaggle.com/datasets/aryanputta/hybrid-satellite-telemetry

## Dataset License
I release this dataset and accompanying benchmark assets under the Creative Commons Attribution 4.0 International license (CC BY 4.0).
