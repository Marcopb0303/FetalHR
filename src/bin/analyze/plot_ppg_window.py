#!/usr/bin/env python3
"""Zoomed-in diagnostic plot of one patient's raw PPG (pvs.npy) over a specific
time window: raw, detrended, and bandpass-filtered-with-detected-beats panels.

Unlike analyze_waveforms.py (which plots the whole recording and gets clamped
to matplotlib's pixel-width limit on long sessions), this windows first, so a
narrow slice can be inspected at full detail -- e.g. to see whether a low
beat count is a genuine weak signal or a single artifact spike corrupting
detect_ppg_beats' global normalization over the rest of the recording.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import detrend

SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from analyze.data import Audio
from analyze.filters import bp_filter
from analyze.sot import detect_ppg_beats
from analyze.util import normalize_path
from constants import MATERNAL_BPM_RANGE, PVS_FILE


def plot_ppg_window(patient_dir: Path, start: float, end: float, out_dir: Path) -> Path:
    path = normalize_path(str(patient_dir))
    name = Path(path.rstrip("/")).name

    pvs = np.load(path + PVS_FILE)
    t_ppg = pvs[:, 0].astype(float)
    x_ppg_raw = pvs[:, 1].astype(float)
    hz_ppg = round(1.0 / float(np.median(np.diff(t_ppg))))

    mask = (t_ppg >= start) & (t_ppg <= end)
    if not mask.any():
        raise ValueError(f"No pvs.npy samples in [{start}, {end}]s for {name}")
    t_w, x_w_raw = t_ppg[mask], x_ppg_raw[mask]

    x_w = detrend(x_w_raw)
    ppg_filt = bp_filter(Audio(t_w, hz_ppg, x_w), 0.7, 4.0, filter_type='butter')
    beats = detect_ppg_beats(ppg_filt, MATERNAL_BPM_RANGE)

    print(f"{name}: window {start}-{end}s, {len(t_w)} samples @ {hz_ppg}Hz, "
          f"{len(beats['times'])} beats detected")

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

    axes[0].plot(t_w, x_w_raw, lw=0.5, color="steelblue")
    axes[0].set_title(f"{name} — raw PPG (pvs.npy, column 1), {start}-{end}s", fontsize=10)
    axes[0].set_ylabel("Raw amplitude")

    axes[1].plot(t_w, x_w, lw=0.5, color="darkorange")
    axes[1].set_title("Detrended", fontsize=10)
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(ppg_filt.time, ppg_filt.data, lw=0.6, color="seagreen")
    if len(beats["times"]):
        y_at_beats = np.interp(beats["times"], ppg_filt.time, ppg_filt.data)
        axes[2].plot(beats["times"], y_at_beats, 'o', color='red', ms=7, zorder=5,
                     label=f"{len(beats['times'])} detected beats")
        axes[2].legend()
    axes[2].set_title("Bandpass 0.7-4.0 Hz + detected beats", fontsize=10)
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ppg_diagnostic_{start:g}-{end:g}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patient_dir", type=Path, help="BannerData-style patient directory")
    parser.add_argument("--start", type=float, default=0.0, help="Window start, seconds (default: 0)")
    parser.add_argument("--end", type=float, required=True, help="Window end, seconds")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for the figure")
    args = parser.parse_args()

    plot_ppg_window(args.patient_dir, args.start, args.end, args.out_dir)


if __name__ == "__main__":
    main()
