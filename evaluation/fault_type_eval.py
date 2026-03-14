"""
fault_type_eval.py
-------------------
Trains all four model architectures and evaluates per-fault-type F1.
Reproduces Tables II and III from Putta (2026).

Usage:
    python evaluation/fault_type_eval.py --all-models
    python evaluation/fault_type_eval.py --model lstm
    python evaluation/fault_type_eval.py --all-models --simulate-only

Author: Aryan Putta, Rutgers University (2026)
Seed: 42
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from encoding.recurrence_plot import batch_encode, extract_windows

SEED = 42
WIN = 50
CHANNELS = ['power_w', 'temp_c', 'voltage_v', 'wheel_rpm']
FAULTS = ['POWER_SPIKE', 'THERMAL_DRIFT', 'VOLTAGE_DROP',
          'WHEEL_OSCILLATION', 'SENSOR_DROPOUT']

TRAIN_SATS = [f'SAT_{i:02d}' for i in range(1, 9)]
TEST_SATS = ['SAT_09', 'SAT_10']


def compute_threshold(model, x_normal_val, pct=99):
    """99th percentile threshold protocol."""
    if hasattr(model, 'predict'):
        recon = model.predict(x_normal_val, verbose=0)
        errors = np.mean(np.square(recon - x_normal_val),
                         axis=tuple(range(1, recon.ndim)))
    else:
        errors = -model.decision_function(x_normal_val)
    return float(np.percentile(errors, pct))


def score_model(model, x_test, threshold, use_rp=False, is_sklearn=False):
    if use_rp:
        rp = batch_encode(x_test, verbose=False)
        recon = model.predict(rp, verbose=0)
        scores = np.mean(np.square(recon - rp), axis=(1, 2, 3))
    elif is_sklearn:
        flat = x_test.reshape(len(x_test), -1)
        scores = -model.decision_function(flat)
    else:
        recon = model.predict(x_test, verbose=0)
        scores = np.mean(np.square(recon - x_test), axis=(1, 2))
    return (scores > threshold).astype(int), scores


def per_fault_f1(y_pred, y_true, fault_labels):
    results = {}
    for fault in FAULTS:
        mask = np.array([f == fault or f == 'NORMAL' for f in fault_labels])
        yt = y_true[mask]
        yp = y_pred[mask]
        results[fault] = round(f1_score(yt, yp, zero_division=0), 4)
    return results


def overall_metrics(y_pred, y_true, scores):
    try:
        auc = round(roc_auc_score(y_true, scores), 4)
    except ValueError:
        auc = None
    return {
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc': auc,
    }


def build_lstm_ae(win=WIN, n_ch=4):
    from tensorflow.keras.layers import Dense, Input, LSTM, RepeatVector, TimeDistributed
    from tensorflow.keras.models import Model

    inp = Input(shape=(win, n_ch))
    x = LSTM(128, activation='relu')(inp)
    x = RepeatVector(win)(x)
    x = LSTM(128, return_sequences=True, activation='relu')(x)
    out = TimeDistributed(Dense(n_ch))(x)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m


def build_std_ae(win=WIN, n_ch=4):
    from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
    from tensorflow.keras.models import Model

    inp = Input(shape=(win, n_ch))
    flat = Flatten()(inp)
    enc = Dense(64, activation='relu')(flat)
    dec = Dense(win * n_ch, activation='linear')(enc)
    out = Reshape((win, n_ch))(dec)
    m = Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m


def build_cnn_ae(win=WIN, n_ch=4):
    from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten,
                                         Input, MaxPooling2D, Reshape)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(win, win, n_ch))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = Dense(64, activation='relu')(Flatten()(x))
    x = Dense(128 * 12 * 12, activation='relu')(encoded)
    x = Reshape((12, 12, 128))(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    out = Conv2D(n_ch, (3, 3), padding='same', activation='sigmoid')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss='mse')
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-models', action='store_true')
    parser.add_argument('--model', choices=['lstm', 'cnn', 'std', 'isoforest'])
    parser.add_argument('--simulate-only', action='store_true',
                        help='Use Gaussian noise only (noise generalization ablation)')
    parser.add_argument('--data', default='data/hybrid_satellite_telemetry.csv')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print(f'Loading dataset from {args.data}...')
    df = pd.read_csv(args.data)

    if args.simulate_only:
        print('Simulate-only mode: replacing hardware noise with Gaussian.')
        for col, std in [('temp_c', 0.3), ('wheel_rpm', 5.0), ('voltage_v', 0.02), ('power_w', 0.12)]:
            df[col] += np.random.normal(0, std, len(df))

    train_df = df[df['satellite_id'].isin(TRAIN_SATS)].reset_index(drop=True)
    test_df = df[df['satellite_id'].isin(TEST_SATS)].reset_index(drop=True)

    x_train_raw, y_train, _ = extract_windows(train_df, CHANNELS, WIN)
    x_test_raw, y_test, faults_test = extract_windows(test_df, CHANNELS, WIN)

    x_train_normal = x_train_raw[y_train == 0]

    if args.all_models:
        models_to_run = ['lstm', 'cnn', 'std', 'isoforest']
    elif args.model:
        models_to_run = [args.model]
    else:
        print('Specify --all-models or --model <name>')
        return

    all_results = {}
    y_test_arr = np.array(y_test)

    for mname in models_to_run:
        print(f'\n=== Training {mname.upper()} ===')
        np.random.seed(SEED)

        if mname == 'lstm':
            model = build_lstm_ae()
            model.fit(x_train_normal, x_train_normal, epochs=args.epochs, batch_size=64,
                      validation_split=0.1, verbose=0)
            thresh = compute_threshold(model, x_train_normal[:2000])
            y_pred, scores = score_model(model, x_test_raw, thresh)

        elif mname == 'std':
            model = build_std_ae()
            model.fit(x_train_normal, x_train_normal, epochs=args.epochs, batch_size=64,
                      validation_split=0.1, verbose=0)
            thresh = compute_threshold(model, x_train_normal[:2000])
            y_pred, scores = score_model(model, x_test_raw, thresh)

        elif mname == 'cnn':
            rp_train = batch_encode(x_train_normal, verbose=True)
            rp_test = batch_encode(x_test_raw, verbose=True)
            model = build_cnn_ae()
            model.fit(rp_train, rp_train, epochs=args.epochs, batch_size=32,
                      validation_split=0.1, verbose=0)
            recon = model.predict(rp_test, verbose=0)
            scores = np.mean(np.square(recon - rp_test), axis=(1, 2, 3))
            thresh = np.percentile(
                np.mean(np.square(model.predict(rp_train[:2000], verbose=0) - rp_train[:2000]),
                        axis=(1, 2, 3)), 99)
            y_pred = (scores > thresh).astype(int)

        elif mname == 'isoforest':
            flat_train = x_train_normal.reshape(len(x_train_normal), -1)
            flat_test = x_test_raw.reshape(len(x_test_raw), -1)
            model = IsolationForest(contamination=0.05, n_estimators=100, random_state=SEED)
            model.fit(flat_train)
            scores = -model.decision_function(flat_test)
            thresh = np.percentile(-model.decision_function(flat_train), 99)
            y_pred = (scores > thresh).astype(int)

        overall = overall_metrics(y_pred, y_test_arr, scores)
        pf_f1 = per_fault_f1(y_pred, y_test_arr, faults_test)
        all_results[mname] = {'overall': overall, 'per_fault_f1': pf_f1}
        print(f'  Overall: {overall}')
        print(f'  Per-fault F1: {pf_f1}')

    print('\n\n=== TABLE II: Overall Performance ===')
    print(f"{'Model':<15} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC-ROC':>10}")
    for m, r in all_results.items():
        o = r['overall']
        auc = 'N/A' if o['auc_roc'] is None else f"{o['auc_roc']:.4f}"
        print(f"{m:<15} {o['precision']:>10.4f} {o['recall']:>8.4f} {o['f1']:>8.4f} {auc:>10}")

    print('\n=== TABLE III: F1 by Fault Type ===')
    header = f"{'Model':<15}" + ''.join(f'{f[:10]:>12}' for f in FAULTS)
    print(header)
    for m, r in all_results.items():
        row = f'{m:<15}' + ''.join(f"{r['per_fault_f1'].get(f, 0):>12.4f}" for f in FAULTS)
        print(row)

    os.makedirs('evaluation/results', exist_ok=True)
    with open('evaluation/results/metrics.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nResults saved to evaluation/results/metrics.json')


if __name__ == '__main__':
    main()
