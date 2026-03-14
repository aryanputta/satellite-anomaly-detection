"""
recurrence_plot.py
-------------------
Recurrence plot encoding for satellite telemetry windows.

Based on: Eckmann, Kamphorst, and Ruelle (1987).
Applied to time-series CV by: Wang and Oates (2015).
Applied to satellite telemetry by: Putta (2026).

Author: Aryan Putta, Rutgers University (2026)
"""

import matplotlib.pyplot as plt
import numpy as np


def window_to_rp_image(window, eps=0.10):
    """
    Convert a (w, C) telemetry window to a (w, w, C) binary recurrence plot image.
    """
    rp_channels = []
    for ch in range(window.shape[1]):
        sig = window[:, ch].astype(np.float32)
        sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-8)
        diff = sig[:, None] - sig[None, :]
        rp_channels.append((np.abs(diff) < eps).astype(np.float32))
    return np.stack(rp_channels, axis=-1)


def batch_encode(windows, eps=0.10, verbose=True):
    """Encode a batch of windows as recurrence plot images."""
    n, w, c = windows.shape
    images = np.zeros((n, w, w, c), dtype=np.float32)
    for i in range(n):
        images[i] = window_to_rp_image(windows[i], eps=eps)
        if verbose and (i + 1) % 1000 == 0:
            print(f'  Encoded {i + 1:,}/{n:,} windows')
    return images


def extract_windows(df, channels, window_size=50, step=1):
    """Extract sliding windows and labels from dataframe."""
    x, y, faults = [], [], []
    vals = df[channels].values
    labels = df['anomaly'].values
    ftypes = df['fault_type'].values

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        x.append(vals[start:end])
        y.append(labels[end - 1])
        faults.append(ftypes[end - 1])

    return np.array(x, dtype=np.float32), np.array(y), faults


def visualize_rp(rp_image, channel_names=None, title=None, figsize=(8, 2.2)):
    """Visualize all channels of a recurrence plot image."""
    c = rp_image.shape[2]
    if channel_names is None:
        channel_names = [f'Ch {i}' for i in range(c)]

    fig, axes = plt.subplots(1, c, figsize=figsize)
    for i, (ax, name) in enumerate(zip(np.atleast_1d(axes), channel_names)):
        ax.imshow(rp_image[:, :, i], cmap='binary', origin='lower',
                  aspect='auto', interpolation='none')
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.8)

    if title:
        fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout(pad=0.3, w_pad=0.5)
    return fig
