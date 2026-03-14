"""
live_demo.py
-------------
Raspberry Pi 4 live classification pipeline.
Reads real Arduino sensor data over USB serial, encodes each 50-timestep
window as a recurrence plot image, classifies with Isolation Forest,
and displays results.

Usage:
    python demo/live_demo.py --port /dev/ttyUSB0 --model models/isolation_forest.pkl

Author: Aryan Putta, Rutgers University (2026)
"""

import argparse
import os
import sys
import time
from collections import deque

import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from encoding.recurrence_plot import window_to_rp_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--port', default='/dev/ttyUSB0')
    p.add_argument('--baud', default=9600, type=int)
    p.add_argument('--model', default='models/isolation_forest.pkl')
    p.add_argument('--threshold', default=None, type=float,
                   help='Override threshold (default: read from threshold_meta.json)')
    p.add_argument('--eps', default=0.10, type=float)
    p.add_argument('--window', default=50, type=int)
    p.add_argument('--no-display', action='store_true')
    return p.parse_args()


def load_threshold(model_path, override=None):
    if override is not None:
        return override
    meta_path = model_path.replace('.pkl', '_threshold.json')
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            return json.load(f)['threshold']
    return -0.12


def classify_window(win_arr, model, threshold, eps):
    rp_flat = window_to_rp_image(win_arr, eps=eps).reshape(1, -1)
    score = -float(model.decision_function(rp_flat)[0])
    return score > threshold, score


def run_demo(args):
    import serial

    print(f'Loading model from {args.model}...')
    model = joblib.load(args.model)
    threshold = load_threshold(args.model, args.threshold)
    print(f'Threshold: {threshold:.4f}')

    window = deque(maxlen=args.window)

    display_ok = False
    if not args.no_display:
        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((480, 320))
            pygame.display.set_caption('Satellite Anomaly Detector')
            font_big = pygame.font.SysFont('monospace', 32, bold=True)
            font_sm = pygame.font.SysFont('monospace', 18)
            display_ok = True
            print('Touchscreen display initialized.')
        except Exception as e:
            print(f'Display unavailable ({e}), running in terminal mode.')

    print(f'Opening serial port {args.port} at {args.baud} baud...')
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2.0)
    print('Connected. Running at 10 Hz. Ctrl+C to stop.\n')

    try:
        while True:
            t0 = time.perf_counter()
            raw = ser.readline().decode('utf-8', errors='ignore').strip()
            if not raw:
                continue
            parts = raw.split(',')
            if len(parts) < 3:
                continue
            try:
                t_n = float(parts[0])
                a_n = float(parts[1])
                v_n = float(parts[2])
            except ValueError:
                continue

            window.append([t_n, a_n, v_n, 0.0])

            label = 'NORMAL'
            score = 0.0
            anomaly = False

            if len(window) == args.window:
                win_arr = np.array(window, dtype=np.float32)
                anomaly, score = classify_window(win_arr, model, threshold, args.eps)
                label = 'ANOMALY' if anomaly else 'NORMAL'

            elapsed_ms = (time.perf_counter() - t0) * 1000
            color = '\033[91m' if anomaly else '\033[92m'
            print(f'{color}{label}\033[0m  score={score:+.4f}  '
                  f'{elapsed_ms:.1f}ms  buf={len(window)}/{args.window}',
                  flush=True)

            if display_ok:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                screen.fill((30, 30, 30))
                clr = (220, 50, 50) if anomaly else (50, 200, 80)
                text_lbl = font_big.render(label, True, clr)
                text_sc = font_sm.render(f'score: {score:+.4f}', True, (200, 200, 200))
                text_lat = font_sm.render(f'latency: {elapsed_ms:.1f} ms', True, (150, 150, 150))
                screen.blit(text_lbl, (20, 100))
                screen.blit(text_sc, (20, 170))
                screen.blit(text_lat, (20, 210))
                pygame.display.flip()

            sleep_t = max(0, 0.10 - (time.perf_counter() - t0))
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        ser.close()
        if display_ok:
            pygame.quit()


if __name__ == '__main__':
    run_demo(parse_args())
