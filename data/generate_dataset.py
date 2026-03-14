"""
generate_dataset.py
--------------------
Generates the Hybrid Satellite Telemetry Anomaly Dataset.

Pipeline:
  1. Simulate 10 satellite runs using orbital physics sinusoidal models.
  2. Inject 5 fault types per run based on NASA-HDBK-7004C failure taxonomy.
  3. Load real Arduino hardware noise and add it to each channel.
  4. Save the combined dataset to data/hybrid_satellite_telemetry.csv.

Usage:
    python data/generate_dataset.py

Reproduces the exact dataset published on Kaggle (CC BY 4.0):
    kaggle.com/datasets/aryanputta/hybrid-satellite-telemetry

Author: Aryan Putta, Rutgers University (2026)
Global seed: 42 throughout for full reproducibility.
"""

import os
from math import pi

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

N_POINTS = 10000
ORBITAL_PERIOD = 90
N_SATELLITES = 10


def simulate_satellite(n_points=N_POINTS, orbital_period=ORBITAL_PERIOD,
                       amp_jitter=0.0, phase_jitter=0.0):
    """Simulate one satellite run with optional inter-satellite variability."""
    t = np.linspace(0, 100, n_points)
    power = (28 + amp_jitter * 4) + 4 * np.sin(2 * pi * t / orbital_period + phase_jitter)
    temp = (20 + amp_jitter * 15) + 15 * np.sin(2 * pi * t / orbital_period + pi / 4 + phase_jitter)
    voltage = (28.5 + amp_jitter) + 1.5 * np.sin(2 * pi * t / orbital_period - pi / 3 + phase_jitter)
    wheel = (3000 + amp_jitter * 200) + 200 * np.sin(2 * pi * t / (orbital_period * 2))
    return pd.DataFrame({
        'timestamp': range(n_points),
        'power_w': power,
        'temp_c': temp,
        'voltage_v': voltage,
        'wheel_rpm': wheel,
        'anomaly': 0,
        'fault_type': 'NORMAL'
    })


def inject_faults(df, seed=None):
    """Inject five fault types based on spacecraft failure classes."""
    if seed is not None:
        np.random.seed(seed)
    df = df.copy()
    faults = {
        'POWER_SPIKE': ('power_w', 30, np.linspace(0, 15, 30)),
        'THERMAL_DRIFT': ('temp_c', 100, np.linspace(0, 25, 100)),
        'VOLTAGE_DROP': ('voltage_v', 50, -np.linspace(0, 8, 50)),
        'WHEEL_OSCILLATION': ('wheel_rpm', 80, 500 * np.sin(np.linspace(0, 4 * pi, 80))),
        'SENSOR_DROPOUT': ('power_w', 20, np.zeros(20)),
    }
    for name, (ch, dur, signal) in faults.items():
        idx = np.random.randint(200, len(df) - dur - 200)
        df.loc[idx:idx + dur - 1, ch] += signal
        df.loc[idx:idx + dur - 1, ['anomaly', 'fault_type']] = [1, name]
    return df


def load_or_generate_noise(n_points, noise_path=None):
    """Load recorded Arduino noise, or synthesize with measured properties."""
    if noise_path and os.path.exists(noise_path):
        raw = pd.read_csv(noise_path).values
        reps = int(np.ceil(n_points / len(raw)))
        raw = np.tile(raw, (reps, 1))[:n_points]
        return raw[:, 0], raw[:, 1], raw[:, 2]

    t_vec = np.arange(n_points) / 10.0
    phi = 0.72
    thermal_noise = np.zeros(n_points)
    for i in range(1, n_points):
        thermal_noise[i] = phi * thermal_noise[i - 1] + np.random.normal(0, 0.24)
    accel_noise = (0.08 * np.sin(2 * pi * 5.0 * t_vec)
                   + np.random.normal(0, 0.04, n_points))
    volt_noise = np.random.normal(0, 0.015, n_points)
    return thermal_noise, accel_noise, volt_noise


def add_hardware_noise(df, noise_path=None):
    """Add hardware-grounded noise to each sensor channel."""
    t_n, a_n, v_n = load_or_generate_noise(len(df), noise_path)
    df = df.copy()
    df['temp_c'] += t_n * 0.30
    df['wheel_rpm'] += a_n * 15.0
    df['voltage_v'] += v_n * 0.05
    df['power_w'] += np.random.normal(0, 0.12, len(df))
    return df


def generate(noise_path=None, output_path='data/hybrid_satellite_telemetry.csv'):
    """Generate the full dataset and save to CSV."""
    np.random.seed(SEED)
    runs = []

    for i in range(1, N_SATELLITES + 1):
        sat_id = f'SAT_{i:02d}'
        amp_j = np.random.uniform(-0.10, 0.10)
        phase_j = np.random.uniform(-0.10, 0.10)

        df = simulate_satellite(amp_jitter=amp_j, phase_jitter=phase_j)
        df = inject_faults(df, seed=SEED + i)
        df = add_hardware_noise(df, noise_path=noise_path)
        df.insert(0, 'satellite_id', sat_id)
        runs.append(df)
        print(f'  {sat_id}: {df["anomaly"].sum()} anomalous rows '
              f'({df["anomaly"].mean() * 100:.2f}%)')

    full = pd.concat(runs, ignore_index=True)

    print(f'\nTotal rows:    {len(full):,}')
    print(f'Anomaly rate:  {full["anomaly"].mean() * 100:.2f}%')
    print('Fault distribution:')
    print(full[full['anomaly'] == 1]['fault_type'].value_counts())

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    full.to_csv(output_path, index=False)
    print(f'\nDataset saved to {output_path}')
    return full


if __name__ == '__main__':
    print('Generating Hybrid Satellite Telemetry Dataset...')
    print(f'Global seed: {SEED}')
    noise_csv = 'noise/arduino_noise_capture/noise_log.csv'
    generate(
        noise_path=noise_csv if os.path.exists(noise_csv) else None,
        output_path='data/hybrid_satellite_telemetry.csv'
    )
