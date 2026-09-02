#!/usr/bin/env python3
"""Interactive HTML analysis of a persistent acoustic carrier and the transients
that disturb it.

The pipeline reads one PicoScope BIN/JSON pair, cuts a time interval out of it,
and walks the signal through four stages:

    raw x(t)
      -> bandpass to the carrier band            x_BP(t)
      -> analytic signal via Hilbert             z(t) = x_BP + j·H{x_BP}
      -> mix down by the carrier and low-pass    u(t) = LPF{z·e^(-j2πf_c t)}
      -> rotate by a slow local phase reference  u_local(t) = u·conj(p)

Stage 3 turns "a 43.6 kHz tone that wobbles" into a slowly varying complex
number; stage 4 removes the background wobble so that fast disturbances stand
out. Alongside that, two independent detectors run on the untouched signal --
framed transient energy and a continuous wavelet transform -- so an event can be
cross-checked instead of trusted from one representation.

Figures land in bubble_raw_plots/ numbered in reading order (01-11).

Run with no arguments to use the hardcoded same-folder BIN/JSON pair:
    python3 plot_raw_bubble_event.py

Zoom individual candidate events (absolute recording seconds):
    python3 plot_raw_bubble_event.py --targeted-cwt 116.0775 116.5344

Study the local phase reference that rotates the demodulated baseband:
    python3 plot_raw_bubble_event.py --compare-local-phase-windows 5 10 20 50 \
        --local-phase-focus 116.344 --validate-local-phase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Plotly is required for interactive HTML output. Install dependencies with:\n"
        "    python3 -m pip install numpy scipy plotly PyWavelets"
    ) from exc
try:
    import pywt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyWavelets is required for the CWT figures. Install dependencies with:\n"
        "    python3 -m pip install numpy scipy plotly PyWavelets"
    ) from exc
from scipy import ndimage, signal


# ==========================================================================
#  Constants
# ==========================================================================

# Which recording, and which slice of it. The CLI can override the interval.
RECORDING_STEM = (
    "pouch200mAh_LFP_YSpcell33_formation_fiber11_AEgel_tegaderm_noFC__2"
)
START_TIME = 115.0
END_TIME = 117.0

# Transient energy: how finely the signal is chopped up, and how far out the
# "what counts as normal here" baseline looks. The baseline must be far longer
# than a frame or it would simply track the events it is meant to measure.
FRAME_DURATION_S = 256e-6
HOP_DURATION_S = 32e-6
BASELINE_DURATION_S = 20e-3

# Complex demodulation. The low-pass sets how much modulation around the
# carrier survives the mix-down; 5 kHz comfortably spans an 8 kHz-wide band
# once it has been shifted down to 0 Hz.
DEMOD_LOWPASS_HZ = 5_000.0
DEMOD_LOWPASS_ORDER = 4
# Zero-phase filters ring at both ends of a finite record. Everything inside
# EDGE_EXCLUSION_S of an edge is filter transient, not physics.
EDGE_EXCLUSION_S = 0.01

# Local phase reference. The governing inequality is T_reference >> T_event:
# a reference much slower than an event cannot follow it, so the event survives
# the rotation; a reference near the event timescale quietly cancels it.
LOCAL_PHASE_WINDOW_MS = 20.0
LOCAL_PHASE_COMPARE_WINDOWS_MS = (5.0, 10.0, 20.0, 50.0)
LOCAL_PHASE_FOCUS_SPAN_MS = 60.0
# Below this fraction of the median |r| the reference direction is noise.
LOCAL_PHASE_CONFIDENCE_FRACTION = 0.1
LOCAL_PHASE_MASK_HALF_WIDTH_S = 2e-3
# Three cascaded moving averages of width w form a quadratic B-spline. Its
# measured FWHM is ~1.268·w -- wider than the 1.1774·w a true Gaussian of the
# same variance would give, hence the empirical constant.
TRIPLE_BOX_FWHM_FACTOR = 1.268
SYNTHETIC_EVENT_DURATION_S = 1.5e-3
EVENT_RETENTION_WARN_RATIO = 0.5

# Continuous wavelet transform. cmor a-b reads "bandwidth a, center frequency
# b"; larger a buys frequency resolution at the cost of time resolution, which
# is exactly the trade-off the five-wavelet figure exists to show.
COMPARISON_WAVELETS = (
    "cmor0.5-1.0",
    "cmor1.5-1.0",
    "cmor3.0-1.0",
    "cgau1",
    "mexh",
)
TARGETED_WAVELET = "cmor1.5-1.0"
CWT_FREQUENCY_BINS = 48
CWT_DYNAMIC_RANGE_DB = 40.0
CWT_MAX_TIME_COLUMNS = 2000
TARGETED_WINDOW_S = 2e-3

# Traces are thinned to this many points before being written into HTML.
PLOT_MAX_POINTS = 120_000


# ==========================================================================
#  1 - Reading the recording
# ==========================================================================


def load_metadata(metadata_path: Path) -> dict:
    """Read the PicoScope JSON sidecar that describes the BIN file."""
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Place a same-named .json file beside the .bin file or pass "
            "--metadata-json PATH."
        )
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sampling_rate_from_metadata(metadata: dict, source: str) -> float:
    """Sampling rate in Hz: fs = 1 / (sample_interval · 1e-6).

    ``sample_interval`` is in microseconds, matching the acquisition code's own
    ``sample_interval * 1e-6`` conversion.
    """
    if "sample_interval" not in metadata:
        raise KeyError(f"{source} has no 'sample_interval' field.")
    interval_us = float(metadata["sample_interval"])
    if not np.isfinite(interval_us) or interval_us <= 0:
        raise ValueError("JSON 'sample_interval' must be a positive number in us.")
    return 1.0 / (interval_us * 1e-6)


def millivolt_scale_from_metadata(metadata: dict) -> float | None:
    """Volts per count: mV = counts · range_full_scale_mV / max_adc.

    Returns None when the JSON omits either field, in which case the caller
    stays in raw ADC counts rather than inventing a calibration.
    """
    if "range_full_scale_mV" not in metadata or "max_adc" not in metadata:
        return None
    full_scale_mv = float(metadata["range_full_scale_mV"])
    max_adc = float(metadata["max_adc"])
    if not np.isfinite(full_scale_mv) or full_scale_mv <= 0:
        raise ValueError("JSON 'range_full_scale_mV' must be a positive number.")
    if not np.isfinite(max_adc) or max_adc <= 0:
        raise ValueError("JSON 'max_adc' must be a positive number.")
    return full_scale_mv / max_adc


def load_array(path: Path, binary_dtype: str) -> np.ndarray:
    """Open a waveform file, memory-mapping raw binary so nothing large is read.

    A multi-gigabyte .bin is mapped, not loaded; only the requested interval is
    ever materialized, by ``extract_interval``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".bin":
        dtype = np.dtype(binary_dtype)
        if path.stat().st_size % dtype.itemsize:
            raise ValueError(
                f"{path.name} size is not divisible by {dtype} ({dtype.itemsize} bytes)."
            )
        return np.memmap(path, dtype=dtype, mode="r")
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if suffix == ".npz":
        archive = np.load(path)
        if len(archive.files) != 1:
            raise ValueError(
                f"{path.name} contains {archive.files}; save one array or use .npy."
            )
        return archive[archive.files[0]]
    if suffix == ".csv":
        return np.genfromtxt(path, delimiter=",", comments="#")
    if suffix in {".txt", ".dat"}:
        return np.genfromtxt(path, comments="#")
    raise ValueError("Supported inputs: .bin, .npy, .npz, .csv, .txt, and .dat")


def extract_interval(
    data: np.ndarray,
    start: float,
    end: float,
    fs_arg: float | None,
    time_column: int,
    channel: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Cut [start, end) out of a 1-D or 2-D array.

    Returns (absolute time, samples, fs). For 1-D data the time axis is implied
    by fs; for 2-D data a time column may supply it, in which case an explicit
    --fs must agree with the column to within 1%.
    """
    if not (end > start >= 0):
        raise ValueError("Require end_time > start_time >= 0.")

    if data.ndim == 1:
        if fs_arg is None:
            raise ValueError("A 1-D waveform requires --fs (for example, --fs 1e6).")
        fs = float(fs_arg)
        i0 = max(0, int(np.floor(start * fs)))
        i1 = min(data.size, int(np.ceil(end * fs)))
        x = np.asarray(data[i0:i1], dtype=np.float64)
        t = np.arange(i0, i1, dtype=np.float64) / fs
    elif data.ndim == 2:
        if channel < 0 or channel >= data.shape[1]:
            raise ValueError(f"--channel must be between 0 and {data.shape[1] - 1}.")
        if time_column >= 0:
            if time_column >= data.shape[1]:
                raise ValueError(
                    f"--time-column must be between 0 and {data.shape[1] - 1}."
                )
            t_all = np.asarray(data[:, time_column], dtype=np.float64)
            if t_all.size < 2 or np.any(np.diff(t_all) <= 0):
                raise ValueError("The selected time column must be strictly increasing.")
            fs_inferred = 1.0 / float(np.median(np.diff(t_all)))
            fs = float(fs_arg) if fs_arg is not None else fs_inferred
            if fs_arg is not None:
                relative_error = abs(fs - fs_inferred) / fs_inferred
                if relative_error > 0.01:
                    raise ValueError(
                        f"--fs={fs:g} Hz disagrees with the time column "
                        f"({fs_inferred:g} Hz) by more than 1%."
                    )
            i0 = int(np.searchsorted(t_all, start, side="left"))
            i1 = int(np.searchsorted(t_all, end, side="left"))
            t = t_all[i0:i1]
            x = np.asarray(data[i0:i1, channel], dtype=np.float64)
        else:
            if fs_arg is None:
                raise ValueError("Data without a time column requires --fs.")
            fs = float(fs_arg)
            i0 = max(0, int(np.floor(start * fs)))
            i1 = min(data.shape[0], int(np.ceil(end * fs)))
            x = np.asarray(data[i0:i1, channel], dtype=np.float64)
            t = np.arange(i0, i1, dtype=np.float64) / fs
    else:
        raise ValueError(f"Expected a 1-D or 2-D numeric array, got shape {data.shape}.")

    if fs <= 0:
        raise ValueError("Sampling rate must be positive.")
    if x.size < 4:
        raise ValueError("The requested interval contains too few samples.")
    if not np.all(np.isfinite(x)):
        raise ValueError("The requested signal interval contains NaN or infinity.")
    return t, x, fs


# ==========================================================================
#  2 - Signal-processing primitives
# ==========================================================================


def bandpass_filter(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, mean removed first.

    ``sosfiltfilt`` runs the filter forwards then backwards: the magnitude
    response is squared (so the effective order is 2·order) but the phase
    cancels exactly, which is what keeps event timing honest. Second-order
    sections are used because a direct b/a form at 40 kHz / 1 MHz is poorly
    conditioned.
    """
    nyquist = fs / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Bandpass edges must satisfy 0 < low < high < Nyquist "
            f"({nyquist:g} Hz); got {low_hz:g}-{high_hz:g} Hz."
        )
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x - np.mean(x))


def estimate_carrier_frequency(
    analytic: np.ndarray, fs: float, edge_s: float = EDGE_EXCLUSION_S
) -> float:
    """Carrier frequency from the slope of the unwrapped analytic phase.

    A pure tone at f_c has phase φ(t) = 2π·f_c·t + φ₀, so a straight-line fit
    of the unwrapped phase gives f_c = slope / 2π. With a real band the answer
    is the energy-weighted mean instantaneous frequency -- trustworthy when one
    component dominates, misleading if two tones of similar strength share the
    band, since the fit would land between them.
    """
    phase = np.unwrap(np.angle(analytic))
    t_rel = np.arange(analytic.size, dtype=np.float64) / fs
    valid = (t_rel >= edge_s) & (t_rel <= t_rel[-1] - edge_s)
    if np.count_nonzero(valid) < 2:
        valid = np.ones(t_rel.size, dtype=bool)
    slope, _ = np.polyfit(t_rel[valid], phase[valid], 1)
    return slope / (2.0 * np.pi)


def demodulate_to_baseband(
    analytic: np.ndarray, fs: float, carrier_hz: float
) -> np.ndarray:
    """Mix an analytic signal down to 0 Hz and low-pass I and Q identically.

        u(t) = LPF{ z(t) · e^(-j·2π·f_c·t) }

    No factor of 2: the analytic signal is already single-sided, so mixing it
    produces no image at -2f_c that would need to be halved away. Time is
    measured from the start of the interval so the exponent stays small --
    2π·43.6e3·116 would otherwise be ~3·10^7 radians before wrapping.
    """
    nyquist = fs / 2.0
    if not (0 < DEMOD_LOWPASS_HZ < nyquist):
        raise ValueError(
            f"Baseband low-pass cutoff {DEMOD_LOWPASS_HZ:g} Hz must be below "
            f"Nyquist ({nyquist:g} Hz)."
        )
    t_rel = np.arange(analytic.size, dtype=np.float64) / fs
    baseband = analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * t_rel)
    lowpass_sos = signal.butter(
        DEMOD_LOWPASS_ORDER, DEMOD_LOWPASS_HZ, btype="lowpass", fs=fs, output="sos"
    )
    # I and Q get the identical zero-phase filter, so the complex signal is not
    # skewed relative to itself.
    i_baseband = signal.sosfiltfilt(lowpass_sos, np.real(baseband))
    q_baseband = signal.sosfiltfilt(lowpass_sos, np.imag(baseband))
    return i_baseband + 1j * q_baseband


def residual_offset_from_baseband(
    baseband_filtered: np.ndarray, t_rel: np.ndarray, valid: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """How far the mix-down missed the true carrier.

    If the demodulation frequency is wrong by Δf, the baseband keeps turning at
    2π·Δf rad/s. Fitting that residual slope recovers Δf = slope / 2π, which is
    what the second demodulation pass corrects.

    Returns (slope in rad/s, offset in Hz, unwrapped phase with its median
    removed).
    """
    residual_phase = np.unwrap(np.angle(baseband_filtered))
    residual_phase = residual_phase - float(np.median(residual_phase[valid]))
    slope, _ = np.polyfit(t_rel[valid], residual_phase[valid], 1)
    return float(slope), float(slope) / (2.0 * np.pi), residual_phase


def box_width_samples(window_ms: float, fs: float) -> int:
    """Odd box width whose triple cascade has a FWHM of ``window_ms``."""
    if not np.isfinite(window_ms) or window_ms <= 0:
        raise ValueError(f"Smoothing window must be positive, got {window_ms!r} ms.")
    width = int(round(window_ms * 1e-3 * fs / TRIPLE_BOX_FWHM_FACTOR))
    width = max(1, width)
    return width if width % 2 else width + 1


def smooth_zero_phase(x: np.ndarray, window_ms: float, fs: float) -> np.ndarray:
    """Zero-phase low-pass built from three cascaded centered moving averages.

    Three boxes converge quickly on a Gaussian (central limit theorem on the
    kernel), but each pass is a running sum, so the whole thing costs O(N) no
    matter how wide the window is. That matters here: a 50 ms reference at
    1 MHz is a 50,000-sample window, and a direct Gaussian convolution would
    need ~170,000 taps per output point.

    Each pass is symmetric, so the cascade has exactly zero group delay -- no
    feature is nudged in time. ``mode="nearest"`` extends the end samples so
    the boundaries are handled explicitly rather than being padded with zeros.
    """
    width = box_width_samples(window_ms, fs)
    smoothed = np.asarray(x, dtype=np.float64)
    if width <= 1:
        return smoothed.copy()
    for _ in range(3):
        smoothed = ndimage.uniform_filter1d(smoothed, width, mode="nearest")
    return smoothed


def rotate_by_local_phase(
    baseband: np.ndarray,
    fs: float,
    window_ms: float,
    valid_mask: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate a complex baseband into a slowly moving local phase frame.

        r(t) = LPF_slow{u(t)}        the drifting background phasor
        p(t) = r(t) / |r(t)|         its direction alone, unit magnitude
        u_local(t) = u(t) · conj(p(t))

    The smoothing is applied to the *complex* signal, never to arg(u). Averaging
    a wrapped angle is wrong: two samples at +179 deg and -179 deg are 2 deg
    apart but average to 0 deg, pointing the reference in exactly the wrong
    direction. Complex smoothing has no such discontinuity to trip over.

    Since |p| = 1, the rotation is a pure change of coordinates and leaves
    |u_local| = |u| untouched. What it does change is the split between the two
    channels: I_local ~ A(t), and Q_local ~ A(t)·[φ(t) - φ_ref(t)] carries only
    the phase motion the reference was too slow to follow.

    ``exclusion_mask`` marks samples that must not influence the reference --
    typically known events, which would otherwise bias the reference toward
    themselves and get partly rotated away. It is handled by normalized
    convolution, smoothing u·w and w separately and dividing, so the reference
    glides across the gap instead of being pulled toward zero. The same
    mechanism makes the array boundaries behave.

    Returns (baseband_local, reference_phasor, reference_magnitude,
    reliable_mask) at full resolution.
    """
    weight = np.ones(baseband.size, dtype=np.float64)
    if exclusion_mask is not None:
        weight[exclusion_mask] = 0.0
        if not np.any(weight):
            raise ValueError("The exclusion mask removes every sample.")

    # r = LPF{u·w} / LPF{w}: the weights cancel where they are uniform, and
    # interpolate the reference where they are not.
    numerator_real = smooth_zero_phase(np.real(baseband) * weight, window_ms, fs)
    numerator_imag = smooth_zero_phase(np.imag(baseband) * weight, window_ms, fs)
    denominator = smooth_zero_phase(weight, window_ms, fs)
    denominator_floor = np.finfo(float).eps * max(float(np.max(denominator)), 1.0)
    reference = (numerator_real + 1j * numerator_imag) / np.maximum(
        denominator, denominator_floor
    )

    # |r| is the confidence: it collapses when the background phasor spends the
    # window turning rather than pointing, and then arg(r) means nothing.
    reference_magnitude = np.abs(reference)
    scope = valid_mask if valid_mask is not None else np.ones(baseband.size, bool)
    median_magnitude = float(np.median(reference_magnitude[scope]))
    magnitude_floor = np.finfo(float).eps * max(median_magnitude, 1.0)
    reference_phasor = reference / np.maximum(reference_magnitude, magnitude_floor)
    reliable_mask = reference_magnitude > max(
        LOCAL_PHASE_CONFIDENCE_FRACTION * median_magnitude, magnitude_floor
    )
    baseband_local = baseband * np.conj(reference_phasor)
    return baseband_local, reference_phasor, reference_magnitude, reliable_mask


def robust_spread(x: np.ndarray) -> float:
    """Median absolute deviation, MAD = median|x - median(x)|, ignoring NaN.

    Used instead of a standard deviation throughout: a single large transient
    barely moves the MAD, which is the point when the transients are the signal
    and the background is what needs measuring.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(np.abs(finite - np.median(finite))))


def compute_transient_score(
    x: np.ndarray, fs: float, t_start: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Score every frame against the local background it sits in.

        E[m] = Σ_n ( x[n + m·H] · w[n] )²          Hann-windowed frame energy
        S[m] = 10·log10( (E[m] + eps) / (median_20ms(E)[m] + eps) )

    S is in dB above local background, so S > 0 means "louder than this stretch
    of recording usually is" -- a scale-free statement that survives the slow
    drifts in overall level. The baseline is a *median*, not a mean, so a burst
    inside the window cannot lift the very threshold it is being judged against.

    Frame energy is computed as a convolution of x² with the reversed squared
    window and then strided by the hop. That is algebraically the same sum as
    the definition above, but it never materializes the (frames x frame_length)
    matrix, which for this recording would be ~62,000 x 256.

    All sizes derive from ``fs``. Returns (frame_center_times, energy, score_db,
    frame_samples, hop_samples, baseline_frames), times offset by ``t_start``.
    """
    frame_samples = max(1, int(round(FRAME_DURATION_S * fs)))
    hop_samples = max(1, int(round(HOP_DURATION_S * fs)))
    if x.size < frame_samples:
        raise ValueError(
            f"Interval has {x.size} samples but one frame needs {frame_samples}."
        )

    x_zero_mean = x - np.mean(x)
    window = signal.windows.hann(frame_samples, sym=False)
    kernel = (window**2)[::-1]
    energy = signal.fftconvolve(x_zero_mean**2, kernel, mode="valid")[::hop_samples]
    energy = np.maximum(energy, 0.0)  # guard against tiny negative FFT round-off

    # Each frame is timestamped at its center, not its start.
    frame_starts = np.arange(energy.size, dtype=np.float64) * hop_samples
    frame_center_time = t_start + (frame_starts + (frame_samples - 1) / 2.0) / fs

    baseline_frames = max(1, int(round(BASELINE_DURATION_S / (hop_samples / fs))))
    if baseline_frames % 2 == 0:
        baseline_frames += 1  # odd, so the median window is centered
    baseline = ndimage.median_filter(energy, size=baseline_frames, mode="nearest")

    # eps scaled to the data keeps log10 finite in silent stretches without
    # distorting anything at normal levels.
    epsilon = np.finfo(float).eps * max(float(np.max(energy)), 1.0)
    score_db = 10.0 * np.log10((energy + epsilon) / (baseline + epsilon))
    return (
        frame_center_time,
        energy,
        score_db,
        frame_samples,
        hop_samples,
        baseline_frames,
    )


def cwt_scales(wavelet: str, frequencies: np.ndarray, fs: float) -> np.ndarray:
    """Scales that place ``wavelet`` at each physical frequency.

        s = f_central(wavelet) · fs / f

    Every wavelet has its own f_central, so the same physical frequency grid
    maps to genuinely different scales per wavelet -- cmor (f_c = 1.0) lands
    near 21-25 samples here while mexh (f_c = 0.25) lands near 5-6.
    """
    return pywt.central_frequency(wavelet) * fs / frequencies


def cwt_power_db(
    x: np.ndarray,
    fs: float,
    wavelet: str,
    frequencies: np.ndarray,
    max_columns: int | None = None,
    scale_chunk: int = 8,
) -> tuple[np.ndarray, int, float]:
    """Wavelet power P_dB(f, t) = 10·log10(|W(f, t)|² + eps) on a fixed grid.

    Scales are transformed a few at a time and collapsed to dB immediately.
    Done in one call, a 48-scale transform of 2 million samples would allocate
    48 x 2e6 complex128 -- about 1.5 GB -- whereas chunk-and-reduce holds a few
    hundred MB.

    Returns (power_db, stride, coi_seconds). ``stride`` is the time decimation
    applied to the columns; ``coi_seconds`` ~ sqrt(2)·s_max/fs is the cone of
    influence, the margin at each end where the wavelet overlaps the edge of
    the record and the coefficients are partly artifact.
    """
    scales = cwt_scales(wavelet, frequencies, fs)
    stride = 1 if max_columns is None else max(1, int(np.ceil(x.size / max_columns)))
    blocks = []
    for start in range(0, scales.size, scale_chunk):
        coefficients, _ = pywt.cwt(
            x,
            scales[start : start + scale_chunk],
            wavelet,
            sampling_period=1.0 / fs,
            method="fft",
        )
        blocks.append((np.abs(coefficients) ** 2)[:, ::stride])
        del coefficients
    power = np.vstack(blocks)
    epsilon = np.finfo(float).eps * max(float(np.max(power)), 1.0)
    power_db = 10.0 * np.log10(power + epsilon)
    coi_seconds = float(np.sqrt(2.0) * float(np.max(scales)) / fs)
    return power_db, stride, coi_seconds


# ==========================================================================
#  3 - Carrier demodulation analysis
# ==========================================================================


def analyze_carrier_demodulation(
    analytic: np.ndarray,
    fs: float,
    carrier_hz: float,
    local_window_ms: float,
    exclusion_mask: np.ndarray | None = None,
) -> dict:
    """Demodulate in two passes, then rotate globally and locally.

    Pass 1 mixes at the phase-regression estimate f_c and measures what is left
    over; pass 2 mixes at f_c + Δf. The correction matters because the pass-1
    estimate is fitted to the wideband analytic phase, while the residual is
    measured after the 5 kHz low-pass has stripped the noise that biased it --
    so the second number is the better one, and after re-mixing the leftover
    offset drops to ~1e-7 Hz.

    Both rotations are computed: the global one (a single constant φ_ref) is
    kept purely so the local one can be shown to improve on it. Every array
    returned comes from the second pass.
    """
    t_rel = np.arange(analytic.size, dtype=np.float64) / fs
    valid = (t_rel >= EDGE_EXCLUSION_S) & (t_rel <= t_rel[-1] - EDGE_EXCLUSION_S)
    if np.count_nonzero(valid) < 2:
        valid = np.ones(t_rel.size, dtype=bool)

    # --- Pass 1: how far off is the initial estimate? ---
    first_pass = demodulate_to_baseband(analytic, fs, carrier_hz)
    slope, residual_offset_hz, _ = residual_offset_from_baseband(
        first_pass, t_rel, valid
    )
    corrected_carrier_hz = carrier_hz + residual_offset_hz
    del first_pass  # ~32 MB; freed before the second pass allocates its own

    # --- Pass 2: the demodulation everything downstream actually uses ---
    baseband = demodulate_to_baseband(analytic, fs, corrected_carrier_hz)
    final_slope, final_offset_hz, _ = residual_offset_from_baseband(
        baseband, t_rel, valid
    )
    amplitude = np.abs(baseband)

    # --- Global rotation: u·e^(-j·φ_ref), one angle for the whole record ---
    # Works only if the carrier is phase-coherent across the interval. Here it
    # is not, which is what motivates the local reference below.
    reference_phase = float(np.angle(np.mean(baseband[valid])))
    global_rotated = baseband * np.exp(-1j * reference_phase)
    i_global = np.real(global_rotated)
    q_global = np.imag(global_rotated)
    del global_rotated

    # --- Local rotation: the coordinate frame follows the background ---
    local_rotated, reference_phasor, reference_magnitude, reliable = (
        rotate_by_local_phase(baseband, fs, local_window_ms, valid, exclusion_mask)
    )
    amplitude_error = float(
        np.max(np.abs(np.abs(local_rotated[reliable]) - amplitude[reliable]))
    )
    i_local = np.real(local_rotated)
    q_local = np.imag(local_rotated)
    # Where the reference is untrustworthy, show nothing rather than a phase
    # the data cannot support. NaN plots as a gap.
    i_local_display = np.where(reliable, i_local, np.nan)
    q_local_display = np.where(reliable, q_local, np.nan)

    low_confidence_fraction = float(
        np.count_nonzero(~reliable[valid]) / np.count_nonzero(valid)
    )
    # The honest before/after: the same slow filter on both channels, so the
    # comparison measures the rotation and not two different filters.
    q_global_slow = smooth_zero_phase(q_global, local_window_ms, fs)
    q_local_slow = smooth_zero_phase(
        np.where(reliable, q_local, 0.0), local_window_ms, fs
    )

    amplitude_valid = amplitude[valid]
    amplitude_median = float(np.median(amplitude_valid))
    return {
        "t_rel": t_rel,
        "valid": valid,
        "fs": fs,
        "carrier_hz": carrier_hz,
        "corrected_carrier_hz": corrected_carrier_hz,
        "residual_phase_slope_rad_s": slope,
        "residual_offset_hz": residual_offset_hz,
        "final_phase_slope_rad_s": final_slope,
        "final_offset_hz": final_offset_hz,
        "baseband": baseband,
        "amplitude": amplitude,
        "reference_phase_rad": reference_phase,
        "i_global": i_global,
        "q_global": q_global,
        "local_window_ms": local_window_ms,
        "reference_phasor": reference_phasor,
        "reference_magnitude": reference_magnitude,
        "reliable": reliable,
        "i_local": i_local_display,
        "q_local": q_local_display,
        "low_confidence_percent": 100.0 * low_confidence_fraction,
        "reference_magnitude_median": float(np.median(reference_magnitude[valid])),
        "amplitude_median": amplitude_median,
        "amplitude_mad": float(np.median(np.abs(amplitude_valid - amplitude_median))),
        "amplitude_rotation_error": amplitude_error,
        "q_global_spread": robust_spread(q_global[valid]),
        "q_local_spread": robust_spread(q_local_display[valid]),
        "q_global_slow_spread": robust_spread(q_global_slow[valid]),
        "q_local_slow_spread": robust_spread(q_local_slow[valid]),
        "excluded_samples": (
            0 if exclusion_mask is None else int(np.count_nonzero(exclusion_mask))
        ),
    }


def validate_local_phase(
    fs: float, windows_ms: list[float], seed: int = 0
) -> list[dict[str, float]]:
    """Prove on synthetic data that the reference keeps short events.

    Builds a baseband whose truth is known -- slow background phase wander,
    slow amplitude variation, noise, and one ~1.5 ms Gaussian phase bump -- and
    rotates it with each candidate window.

    The yardstick is rotation by the *true* background phasor, the best any
    reference could do, so the retention ratio

        R_peak = max|Q_local| / max|Q_ideal|     over the event

    reads as a fraction of the achievable, not of the raw signal. Watching
    R_peak fall as the window shrinks is the direct measurement of a reference
    starting to track and cancel the event it was supposed to reveal.
    """
    rng = np.random.default_rng(seed)
    duration_s = 0.5
    n = int(round(duration_s * fs))
    t_rel = np.arange(n, dtype=np.float64) / fs

    # Two incommensurate slow tones: several radians of wander, no period that
    # the smoother could lock onto by luck.
    background_phase = (
        5.0 * np.sin(2.0 * np.pi * 1.5 * t_rel)
        + 3.0 * np.sin(2.0 * np.pi * 0.7 * t_rel + 1.0)
    )
    amplitude = 70.0 * (1.0 + 0.2 * np.sin(2.0 * np.pi * 3.0 * t_rel))

    event_center_s = duration_s / 2.0
    event_sigma = SYNTHETIC_EVENT_DURATION_S / 2.355  # FWHM -> sigma
    event_phase = 0.5 * np.exp(-0.5 * ((t_rel - event_center_s) / event_sigma) ** 2)
    event_window = np.abs(t_rel - event_center_s) <= 3.0 * SYNTHETIC_EVENT_DURATION_S

    noise = rng.normal(0.0, 1.5, n) + 1j * rng.normal(0.0, 1.5, n)
    baseband = amplitude * np.exp(1j * (background_phase + event_phase)) + noise

    # The ceiling: rotate by the background we know we put in.
    q_ideal = np.imag(baseband * np.exp(-1j * background_phase))
    ideal_peak = float(np.max(np.abs(q_ideal[event_window])))
    ideal_energy = float(np.sum(q_ideal[event_window] ** 2))
    ideal_peak_time = float(
        t_rel[event_window][np.argmax(np.abs(q_ideal[event_window]))]
    )

    q_global = np.imag(baseband * np.exp(-1j * np.angle(np.mean(baseband))))
    background_reference_spread = robust_spread(
        smooth_zero_phase(q_global, max(windows_ms), fs)
    )

    results: list[dict[str, float]] = []
    for window_ms in windows_ms:
        rotated, _phasor, _magnitude, reliable = rotate_by_local_phase(
            baseband, fs, window_ms, None, None
        )
        q_local = np.where(reliable, np.imag(rotated), np.nan)
        event_values = q_local[event_window]
        peak_index = int(np.nanargmax(np.abs(event_values)))
        peak = float(np.abs(event_values[peak_index]))
        results.append(
            {
                "window_ms": window_ms,
                "peak_retention": peak / ideal_peak if ideal_peak else float("nan"),
                "energy_retention": (
                    float(np.nansum(event_values**2)) / ideal_energy
                    if ideal_energy
                    else float("nan")
                ),
                "peak_shift_ms": (
                    float(t_rel[event_window][peak_index] - ideal_peak_time) * 1e3
                ),
                "slow_spread_reduction": (
                    background_reference_spread
                    / max(
                        robust_spread(
                            smooth_zero_phase(np.nan_to_num(q_local), window_ms, fs)
                        ),
                        np.finfo(float).eps,
                    )
                ),
            }
        )
        del rotated
    return results


# ==========================================================================
#  4 - Plotting helpers
# ==========================================================================


def db_label(unit: str) -> str:
    """Axis label for a dB scale referenced to one unit of amplitude."""
    return f"dB re 1 {unit}" if unit == "mV" else "dB"


def peak_preserving_decimate(
    t: np.ndarray, y: np.ndarray, max_points: int = PLOT_MAX_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Thin a trace for plotting while keeping each bucket's min and max.

    Plain striding samples every k-th point and will happily step over a
    one-sample spike -- exactly the features being hunted. Keeping both extremes
    per bucket preserves the visual envelope at a fraction of the file size.
    NaN survives as NaN, so blanked regions stay visibly blank.
    """
    n = y.size
    if n <= max_points:
        return t, y
    buckets = max(1, max_points // 2)  # two points kept per bucket
    size = int(np.ceil(n / buckets))
    padding = buckets * size - n
    padded = np.append(y, np.full(padding, np.nan))
    grid = padded.reshape(buckets, size)
    # +/-inf stand in for NaN so argmin/argmax ignore them without erroring.
    low = np.argmin(np.where(np.isnan(grid), np.inf, grid), axis=1)
    high = np.argmax(np.where(np.isnan(grid), -np.inf, grid), axis=1)
    base = np.arange(buckets) * size
    index = np.unique(np.concatenate([base + low, base + high]))
    index = index[index < n]
    return t[index], y[index]


def add_edge_shading(fig: go.Figure, t: np.ndarray) -> None:
    """Grey out the filter-transient margins so they are not read as events."""
    for x0, x1 in (
        (float(t[0]), float(t[0]) + EDGE_EXCLUSION_S),
        (float(t[-1]) - EDGE_EXCLUSION_S, float(t[-1])),
    ):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="gray", opacity=0.15, line_width=0,
            row="all", col=1,
        )


# ==========================================================================
#  5 - Figures (numbered as they appear in bubble_raw_plots/)
# ==========================================================================


def save_waveform(t: np.ndarray, x: np.ndarray, output: Path, unit: str) -> None:
    """01 - the untouched signal, the reference against which all else is read."""
    fig = go.Figure(
        go.Scattergl(x=t, y=x, mode="lines", line={"width": 1}, name="Raw signal")
    )
    fig.update_layout(
        title="Raw waveform (unfiltered)",
        xaxis_title="Absolute time (s)",
        yaxis_title=f"Raw amplitude ({unit})",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)


def save_fft(
    x: np.ndarray, fs: float, output: Path, max_frequency: float, unit: str
) -> None:
    """02 - amplitude spectrum of the whole interval, linear and in dB.

    Only the mean is removed; no bandpass or other shaping is applied, so this
    is the honest picture of what lives in the recording.
    """
    x_zero_mean = x - np.mean(x)
    window = signal.windows.hann(x.size, sym=False)
    # Divide by the window's coherent gain Σw so a sinusoid of amplitude A
    # reads back as A rather than as A·Σw/2.
    coherent_gain = np.sum(window)
    spectrum = np.fft.rfft(x_zero_mean * window)
    frequency = np.fft.rfftfreq(x.size, d=1.0 / fs)
    amplitude = 2.0 * np.abs(spectrum) / coherent_gain  # 2x: fold the negative side
    if amplitude.size:
        amplitude[0] *= 0.5  # ...except DC, which has no mirror image

    keep = frequency <= min(max_frequency, fs / 2.0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09)
    fig.add_trace(
        go.Scattergl(x=frequency[keep], y=amplitude[keep], mode="lines", name="Amplitude"),
        row=1,
        col=1,
    )
    db = 20.0 * np.log10(np.maximum(amplitude[keep], np.finfo(float).tiny))
    fig.add_trace(
        go.Scattergl(
            x=frequency[keep], y=db, mode="lines", name=f"Amplitude ({db_label(unit)})"
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text=f"Amplitude ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Amplitude ({db_label(unit)})", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_layout(
        title="FFT of the selected raw interval (Hann window, mean removed)",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)


def save_spectrogram(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    output: Path,
    nperseg: int,
    overlap_fraction: float,
    max_frequency: float,
    dynamic_range_db: float,
    unit: str,
) -> tuple[float, float]:
    """03 - STFT of the raw signal: where the energy sits, and when.

    Returns (window_seconds, hop_seconds) so the caller can report the actual
    resolution rather than the requested one.
    """
    nperseg = min(int(nperseg), x.size)
    if nperseg < 8:
        raise ValueError("--nperseg must be at least 8.")
    noverlap = int(round(nperseg * overlap_fraction))
    noverlap = min(max(0, noverlap), nperseg - 1)

    frequency, relative_time, stft = signal.stft(
        x - np.mean(x),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        detrend=False,
        boundary=None,  # no zero-padded frames invented at the ends
        padded=False,
    )
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(stft), np.finfo(float).tiny))
    keep = frequency <= min(max_frequency, fs / 2.0)
    magnitude_db = magnitude_db[keep]
    frequency = frequency[keep]
    # Clamp to a fixed dynamic range below the peak, so the color scale is spent
    # on the signal instead of on the numerical noise floor.
    peak_db = float(np.max(magnitude_db))
    floor_db = peak_db - dynamic_range_db
    magnitude_db = np.maximum(magnitude_db, floor_db)

    absolute_time = float(t[0]) + relative_time
    fig = go.Figure(
        go.Heatmap(
            x=absolute_time,
            y=frequency,
            z=magnitude_db,
            colorscale="Turbo",
            zmin=floor_db,
            zmax=peak_db,
            colorbar={"title": f"Magnitude<br>({db_label(unit)})"},
            hovertemplate="Time: %{x:.6f} s<br>Frequency: %{y:.1f} Hz<br>Magnitude: %{z:.1f} dB<extra></extra>",
        )
    )
    fig.update_layout(
        title="STFT spectrogram of raw data (unfiltered)",
        xaxis_title="Absolute time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
    )
    fig.write_html(output, include_plotlyjs=True)
    hop_seconds = (nperseg - noverlap) / fs
    window_seconds = nperseg / fs
    return window_seconds, hop_seconds


def save_bandpassed_waveform(
    t: np.ndarray,
    x_filtered: np.ndarray,
    low_hz: float,
    high_hz: float,
    output: Path,
    unit: str,
) -> None:
    """04 - the carrier band alone, with everything else filtered away."""
    fig = go.Figure(
        go.Scattergl(
            x=t,
            y=x_filtered,
            mode="lines",
            line={"width": 1},
            name=f"Bandpassed {low_hz / 1e3:g}-{high_hz / 1e3:g} kHz",
        )
    )
    fig.update_layout(
        title=(
            f"Bandpassed waveform ({low_hz / 1e3:g}-{high_hz / 1e3:g} kHz, "
            "zero-phase Butterworth)"
        ),
        xaxis_title="Absolute time (s)",
        yaxis_title=f"Filtered amplitude ({unit})",
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)


def save_hilbert_and_transient_energy(
    t: np.ndarray,
    x: np.ndarray,
    envelope: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    output: Path,
    unit: str,
) -> tuple[int, int, int]:
    """05 - band envelope over broadband transient energy.

    Two views deliberately taken from different signals: the envelope |z(t)|
    describes the carrier band, the score below describes the *raw* broadband
    signal. A feature appearing in both is band-limited; one appearing only
    below is broadband.

    Returns (frame_samples, hop_samples, baseline_frames) as actually used.
    """
    (
        frame_center_time,
        _energy,
        transient_db,
        frame_samples,
        hop_samples,
        baseline_frames,
    ) = compute_transient_score(x, fs, float(t[0]))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(
        go.Scattergl(
            x=t,
            y=envelope,
            mode="lines",
            line={"width": 1},
            name=f"{low_hz / 1e3:g}-{high_hz / 1e3:g} kHz envelope",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=frame_center_time,
            y=transient_db,
            mode="lines",
            line={"width": 1},
            name="Transient score",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(
        title_text=(
            f"{low_hz / 1e3:g}-{high_hz / 1e3:g} kHz amplitude envelope ({unit})"
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Energy above local baseline (dB)", row=2, col=1)
    fig.update_xaxes(title_text="Recording time (s)", row=2, col=1)
    fig.update_layout(
        title=(
            "Hilbert envelope of the bandpassed signal and broadband transient "
            "energy relative to a moving-median baseline"
        ),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)
    return frame_samples, hop_samples, baseline_frames


def save_carrier_demodulation(
    t: np.ndarray, result: dict, output: Path, unit: str
) -> None:
    """06 - the main result: |u|, then Q and I in the local phase frame.

    Read it as: amplitude on top, fast phase disturbance in the middle, steady
    carrier level at the bottom. The channels are labelled "local" because the
    coordinate system itself drifts slowly with the background.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "Demodulated amplitude |u(t)|",
            "Local phase-relative Q",
            "Local phase-relative I",
        ),
    )
    for row, (values, name) in enumerate(
        (
            (result["amplitude"], "|u|"),
            (result["q_local"], "Q_local"),
            (result["i_local"], "I_local"),
        ),
        start=1,
    ):
        time_axis, thinned = peak_preserving_decimate(t, values)
        fig.add_trace(
            go.Scattergl(
                x=time_axis, y=thinned, mode="lines", line={"width": 1}, name=name
            ),
            row=row,
            col=1,
        )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=2, col=1)
    add_edge_shading(fig, t)

    fig.update_yaxes(title_text=f"|u| ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Q_local ({unit})", row=2, col=1)
    fig.update_yaxes(title_text=f"I_local ({unit})", row=3, col=1)
    fig.update_xaxes(title_text="Recording time (s)", row=3, col=1)
    fig.update_layout(
        title=(
            f"Complex demodulation at the corrected carrier "
            f"{result['corrected_carrier_hz']:,.3f} Hz &mdash; rotated by a "
            f"{result['local_window_ms']:g} ms local phase reference "
            f"({result['low_confidence_percent']:.2f}% low-confidence samples "
            "blanked; shaded edges are filter artifacts)"
        ),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)


def save_wavelet_comparison(
    t: np.ndarray,
    x_filtered: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    output: Path,
) -> list[tuple[str, float, float, float]]:
    """07 - the same band seen through five wavelets.

    Every panel shares one physical frequency grid but gets its own scales, so
    what differs between panels is the wavelet's time/frequency trade-off and
    nothing else. Each panel is normalized to its own peak: the wavelets carry
    different amplitude normalizations, so absolute levels are not comparable
    while shapes are -- hence "dB re panel peak" on the color bar.

    Returns one (wavelet, min_scale, max_scale, coi_seconds) tuple per wavelet.
    """
    frequencies = np.linspace(low_hz, high_hz, CWT_FREQUENCY_BINS)
    titles = []
    summary: list[tuple[str, float, float, float]] = []
    fig = make_subplots(
        rows=len(COMPARISON_WAVELETS),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[w for w in COMPARISON_WAVELETS],
    )
    for row, wavelet in enumerate(COMPARISON_WAVELETS, start=1):
        power_db, stride, coi_seconds = cwt_power_db(
            x_filtered, fs, wavelet, frequencies, CWT_MAX_TIME_COLUMNS
        )
        power_db -= float(np.max(power_db))  # 0 dB = this panel's peak
        time_axis = t[::stride][: power_db.shape[1]]
        scales = cwt_scales(wavelet, frequencies, fs)
        titles.append(
            f"{wavelet} &mdash; scales {scales.min():.1f}&ndash;{scales.max():.1f} "
            f"samples, cone of influence {coi_seconds * 1e3:.2f} ms"
        )
        summary.append(
            (wavelet, float(scales.min()), float(scales.max()), coi_seconds)
        )
        fig.add_trace(
            go.Heatmap(
                x=time_axis,
                y=frequencies / 1e3,
                z=np.round(power_db, 1),  # 1 decimal keeps the HTML small
                coloraxis="coloraxis",  # one shared color bar for all panels
                hovertemplate=(
                    "Time: %{x:.5f} s<br>Frequency: %{y:.2f} kHz"
                    "<br>Power: %{z:.1f} dB<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        for x0, x1 in (
            (float(t[0]), float(t[0]) + coi_seconds),
            (float(t[-1]) - coi_seconds, float(t[-1])),
        ):
            fig.add_vrect(
                x0=x0, x1=x1, fillcolor="white", opacity=0.35, line_width=0,
                row=row, col=1,
            )
        fig.update_yaxes(title_text="Frequency (kHz)", row=row, col=1)
        del power_db  # each panel is ~100 MB before reduction

    # Titles are rewritten after the fact because the scale ranges are only
    # known once each transform has run.
    for annotation, title in zip(fig.layout.annotations, titles):
        annotation.text = title
    fig.update_xaxes(title_text="Recording time (s)", row=len(COMPARISON_WAVELETS), col=1)
    fig.update_layout(
        title=(
            f"CWT wavelet comparison on the {low_hz / 1e3:g}&ndash;{high_hz / 1e3:g} kHz "
            "bandpassed signal &mdash; each panel normalized to its own peak "
            "(shaded edges are cone-of-influence regions)"
        ),
        coloraxis={
            "colorscale": "Turbo",
            "cmin": -CWT_DYNAMIC_RANGE_DB,
            "cmax": 0.0,
            "colorbar": {"title": "Power (dB re<br>panel peak)"},
        },
        template="plotly_white",
        height=1500,
    )
    fig.write_html(output, include_plotlyjs=True)
    return summary


def save_transient_energy_comparison(
    t: np.ndarray,
    x: np.ndarray,
    x_filtered: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    output: Path,
) -> tuple[int, int, int]:
    """08 - is a transient broadband, band-limited, or both?

    Same frames, same hop, same baseline for both panels; only the input signal
    differs. Sharing the y range makes the comparison a matter of reading
    heights rather than of trusting two axes.
    """
    (
        broadband_time,
        _broadband_energy,
        broadband_db,
        frame_samples,
        hop_samples,
        baseline_frames,
    ) = compute_transient_score(x, fs, float(t[0]))
    band_time, _band_energy, band_db, _, _, _ = compute_transient_score(
        x_filtered, fs, float(t[0])
    )

    span = [
        float(min(broadband_db.min(), band_db.min())),
        float(max(broadband_db.max(), band_db.max())),
    ]
    margin = 0.05 * (span[1] - span[0]) or 1.0
    y_range = [span[0] - margin, span[1] + margin]

    band_label = f"{low_hz / 1e3:g}&ndash;{high_hz / 1e3:g} kHz bandpassed"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            "Raw broadband signal",
            band_label.replace("&ndash;", "-"),
        ),
    )
    fig.add_trace(
        go.Scattergl(
            x=broadband_time,
            y=broadband_db,
            mode="lines",
            line={"width": 1},
            name="Broadband",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=band_time,
            y=band_db,
            mode="lines",
            line={"width": 1},
            name=band_label.replace("&ndash;", "-"),
        ),
        row=2,
        col=1,
    )
    for row in (1, 2):
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=row, col=1)
        fig.update_yaxes(
            title_text="Energy above local<br>baseline (dB)",
            range=y_range,
            row=row,
            col=1,
        )
    fig.update_xaxes(title_text="Recording time (s)", row=2, col=1)
    fig.update_layout(
        title=(
            "Transient energy above a 20 ms moving-median baseline: raw broadband "
            f"vs {band_label} (identical {FRAME_DURATION_S * 1e6:g} us frames, "
            f"{HOP_DURATION_S * 1e6:g} us hop)"
        ),
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(output, include_plotlyjs=True)
    return frame_samples, hop_samples, baseline_frames


def save_local_phase_diagnostics(
    t: np.ndarray, result: dict, output: Path, unit: str
) -> None:
    """09 - did the local reference remove wander without erasing events?

    Panels 1 and 2 are the before/after. Panel 3 shows the coordinate system
    that was subtracted, panel 4 the confidence |r| behind it, with the
    low-confidence threshold drawn in red.
    """
    reference_phase_deg = np.degrees(np.unwrap(np.angle(result["reference_phasor"])))
    panels = (
        (result["q_global"], f"Globally rotated Q ({unit})", "Q_global"),
        (result["q_local"], f"Locally rotated Q ({unit})", "Q_local"),
        (reference_phase_deg, "Local reference<br>phase (deg, unwrapped)", "phase"),
        (result["reference_magnitude"], f"Local reference<br>magnitude ({unit})", "|r|"),
    )
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Globally rotated Q &mdash; one constant reference phase "
            "(slow wander dominates)",
            f"Locally rotated Q &mdash; {result['local_window_ms']:g} ms reference "
            "(slow wander removed)",
            "Local reference phase (the slowly moving coordinate system)",
            "Local reference magnitude |r| with the low-confidence threshold",
        ),
    )
    for row, (values, axis_title, name) in enumerate(panels, start=1):
        time_axis, thinned = peak_preserving_decimate(t, values)
        fig.add_trace(
            go.Scattergl(
                x=time_axis, y=thinned, mode="lines", line={"width": 1}, name=name
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=axis_title, row=row, col=1)
    fig.add_hline(
        y=LOCAL_PHASE_CONFIDENCE_FRACTION * result["reference_magnitude_median"],
        line_dash="dash",
        line_color="red",
        row=4,
        col=1,
    )
    add_edge_shading(fig, t)
    fig.update_xaxes(title_text="Recording time (s)", row=4, col=1)
    fig.update_layout(
        title=(
            "Local phase-reference diagnostics at "
            f"{result['corrected_carrier_hz']:,.3f} Hz "
            f"({result['local_window_ms']:g} ms reference)"
        ),
        template="plotly_white",
        height=1200,
    )
    fig.write_html(output, include_plotlyjs=True)


def save_local_phase_window_comparison(
    t: np.ndarray,
    result: dict,
    windows_ms: list[float],
    output: Path,
    unit: str,
    exclusion_mask: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """10 - locally rotated Q across several reference timescales.

    The temptation is to pick whichever window makes Q flattest. That is
    backwards: a short reference flattens Q precisely because it has started
    following -- and cancelling -- the events. The right choice keeps
    T_reference >> T_event and accepts a less tidy background.

    Returns one summary dict per window.
    """
    baseband = result["baseband"]
    fs = result["fs"]
    valid = result["valid"]
    summary: list[dict[str, float]] = []
    series: list[tuple[str, np.ndarray]] = [
        ("Global (constant reference)", result["q_global"])
    ]
    for window_ms in windows_ms:
        rotated, _phasor, magnitude, reliable = rotate_by_local_phase(
            baseband, fs, window_ms, valid, exclusion_mask
        )
        q_values = np.where(reliable, np.imag(rotated), np.nan)
        series.append((f"{window_ms:g} ms local reference", q_values))
        summary.append(
            {
                "window_ms": window_ms,
                "q_spread": robust_spread(q_values[valid]),
                "q_slow_spread": robust_spread(
                    smooth_zero_phase(np.nan_to_num(q_values), window_ms, fs)[valid]
                ),
                "low_confidence_percent": 100.0
                * np.count_nonzero(~reliable[valid])
                / np.count_nonzero(valid),
                "reference_magnitude_median": float(np.median(magnitude[valid])),
            }
        )
        del rotated

    # Local panels share one scale so their differences are real differences;
    # the global panel keeps its own because its wander is an order of
    # magnitude larger and would flatten everything else.
    local_limit = max(
        float(np.nanpercentile(np.abs(values[valid]), 99.9))
        for _label, values in series[1:]
    )
    fig = make_subplots(
        rows=len(series),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=[
            label + (" &mdash; independent y scale" if index == 0 else "")
            for index, (label, _values) in enumerate(series)
        ],
    )
    for row, (label, values) in enumerate(series, start=1):
        time_axis, thinned = peak_preserving_decimate(t, values)
        fig.add_trace(
            go.Scattergl(
                x=time_axis, y=thinned, mode="lines", line={"width": 1}, name=label
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=f"Q ({unit})", row=row, col=1)
        if row > 1:
            fig.update_yaxes(range=[-local_limit, local_limit], row=row, col=1)
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", row=row, col=1)
    add_edge_shading(fig, t)
    fig.update_xaxes(title_text="Recording time (s)", row=len(series), col=1)
    fig.update_layout(
        title=(
            "Local phase-reference window comparison &mdash; locally rotated Q. "
            "Shorter references flatten the background but also start to track "
            "and cancel short events; all local panels share one y scale."
        ),
        template="plotly_white",
        height=260 * len(series),
        showlegend=False,
    )
    fig.write_html(output, include_plotlyjs=True)
    return summary


def save_local_phase_focus(
    t: np.ndarray,
    result: dict,
    windows_ms: list[float],
    center_time: float,
    span_ms: float,
    output: Path,
    unit: str,
    exclusion_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """11 - one candidate event, every reference window side by side.

    Each rotation is computed on the *full* record and only then sliced, so no
    panel suffers edge effects from a short window. All Q panels share one y
    scale, which turns "did contrast improve?" into a question of comparing
    heights by eye.

    Returns per-window peak |Q| and its offset from the requested time.
    """
    fs = result["fs"]
    baseband = result["baseband"]
    valid = result["valid"]
    half_samples = max(4, int(round(span_ms * 1e-3 * fs / 2.0)))
    center_index = int(round((center_time - float(t[0])) * fs))
    i0 = max(0, center_index - half_samples)
    i1 = min(t.size, center_index + half_samples)
    relative_ms = (t[i0:i1] - center_time) * 1e3

    # (label, values, shares_the_common_y_scale)
    panels: list[tuple[str, np.ndarray, bool]] = [
        (f"Demodulated amplitude |u| ({unit})", result["amplitude"][i0:i1], False),
        (f"Globally rotated Q ({unit})", result["q_global"][i0:i1], True),
    ]
    peaks: dict[str, float] = {}
    for window_ms in windows_ms:
        rotated, _phasor, _magnitude, reliable = rotate_by_local_phase(
            baseband, fs, window_ms, valid, exclusion_mask
        )
        q_values = np.where(reliable, np.imag(rotated), np.nan)[i0:i1]
        panels.append((f"Local Q, {window_ms:g} ms reference ({unit})", q_values, True))
        finite = np.isfinite(q_values)
        if np.any(finite):
            peak_index = int(np.nanargmax(np.abs(q_values)))
            peaks[f"peak_{window_ms:g}ms"] = float(np.abs(q_values[peak_index]))
            peaks[f"peak_time_ms_{window_ms:g}ms"] = float(relative_ms[peak_index])
        del rotated

    shared_limit = max(
        float(np.nanmax(np.abs(values))) for _label, values, shared in panels if shared
    )
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[label for label, _values, _shared in panels],
    )
    for row, (label, values, shared) in enumerate(panels, start=1):
        fig.add_trace(
            go.Scattergl(
                x=relative_ms, y=values, mode="lines", line={"width": 1}, name=label
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=row, col=1)
        if shared:
            fig.update_yaxes(range=[-shared_limit, shared_limit], row=row, col=1)
    fig.add_vline(x=0.0, line_dash="dot", line_color="black", row="all", col=1)
    fig.update_xaxes(
        title_text=f"Time relative to {center_time:.6f} s (ms)", row=len(panels), col=1
    )
    fig.update_layout(
        title=(
            f"Candidate event at {center_time:.6f} s &mdash; reference-window "
            "comparison. All Q panels share one y scale; the amplitude panel "
            "is scaled independently."
        ),
        template="plotly_white",
        height=200 * len(panels),
        showlegend=False,
    )
    fig.write_html(output, include_plotlyjs=True)
    peaks["start_time"] = float(t[i0])
    peaks["stop_time"] = float(t[i1 - 1])
    return peaks


def save_targeted_cwt(
    t: np.ndarray,
    x_filtered: np.ndarray,
    fs: float,
    center_time: float,
    low_hz: float,
    high_hz: float,
    output: Path,
    unit: str,
) -> dict[str, float | int | bool]:
    """targeted_cwt_<t>s - 2 ms around one event, waveform over scalogram.

    The segment is sliced from the already-bandpassed record rather than
    filtered on its own: a 4th-order 40-48 kHz filter applied to 2 ms of data
    would be almost entirely start-up transient.

    ``center_time`` is absolute recording time. Returns the exact samples used.
    """
    # Index arithmetic, not searchsorted on floats: rounding at the two window
    # edges would otherwise sometimes yield 2001 samples instead of 2000.
    half_samples = max(1, int(round(TARGETED_WINDOW_S / 2.0 * fs)))
    center_index = int(round((center_time - float(t[0])) * fs))
    i0 = center_index - half_samples
    i1 = center_index + half_samples
    i0_clipped = max(0, i0)
    i1_clipped = min(t.size, i1)
    if i1_clipped - i0_clipped < 8:
        raise ValueError(
            f"The window around {center_time:.6f} s holds too few samples."
        )
    clipped = (i0_clipped != i0) or (i1_clipped != i1)

    segment = x_filtered[i0_clipped:i1_clipped]
    relative_ms = (t[i0_clipped:i1_clipped] - center_time) * 1e3

    frequencies = np.linspace(low_hz, high_hz, CWT_FREQUENCY_BINS)
    power_db, _stride, coi_seconds = cwt_power_db(
        segment, fs, TARGETED_WAVELET, frequencies
    )
    power_db -= float(np.max(power_db))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.35, 0.65],
        subplot_titles=(
            f"Bandpassed waveform ({low_hz / 1e3:g}&ndash;{high_hz / 1e3:g} kHz)",
            f"{TARGETED_WAVELET} scalogram, normalized to the window peak",
        ),
    )
    fig.add_trace(
        go.Scattergl(
            x=relative_ms, y=segment, mode="lines", line={"width": 1}, name="Waveform"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=relative_ms,
            y=frequencies / 1e3,
            z=np.round(power_db, 1),
            colorscale="Turbo",
            zmin=-CWT_DYNAMIC_RANGE_DB,
            zmax=0.0,
            colorbar={"title": "Power (dB re<br>window peak)"},
            hovertemplate=(
                "Time: %{x:.4f} ms<br>Frequency: %{y:.2f} kHz"
                "<br>Power: %{z:.1f} dB<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    for x0, x1 in (
        (float(relative_ms[0]), float(relative_ms[0]) + coi_seconds * 1e3),
        (float(relative_ms[-1]) - coi_seconds * 1e3, float(relative_ms[-1])),
    ):
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor="white", opacity=0.35, line_width=0, row=2, col=1
        )
    fig.add_vline(x=0.0, line_dash="dot", line_color="black", row="all", col=1)
    fig.update_yaxes(title_text=f"Amplitude ({unit})", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (kHz)", row=2, col=1)
    fig.update_xaxes(
        title_text=f"Time relative to {center_time:.6f} s (ms)", row=2, col=1
    )
    fig.update_layout(
        title=(
            f"Targeted CWT at {center_time:.6f} s "
            f"({(i1_clipped - i0_clipped) / fs * 1e3:.3f} ms window"
            f"{', clipped at the interval boundary' if clipped else ''})"
        ),
        template="plotly_white",
        showlegend=False,
    )
    fig.write_html(output, include_plotlyjs=True)
    return {
        "start_index": i0_clipped,
        "stop_index": i1_clipped,
        "start_time": float(t[i0_clipped]),
        "stop_time": float(t[i1_clipped - 1]),
        "duration_ms": (i1_clipped - i0_clipped) / fs * 1e3,
        "clipped": clipped,
    }


# ==========================================================================
#  6 - Command line
# ==========================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # --- Input and interval ---
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="Optional input override. With no argument, the hardcoded same-folder BIN is used.",
    )
    parser.add_argument("--start-time", type=float, default=START_TIME)
    parser.add_argument("--end-time", type=float, default=END_TIME)
    parser.add_argument("--fs", type=float, default=None, help="Sampling rate in Hz")
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Metadata JSON for .bin input (default: same basename as the BIN)",
    )
    parser.add_argument(
        "--binary-dtype",
        default="<i2",
        help="NumPy dtype for .bin input (default: <i2, little-endian int16)",
    )
    parser.add_argument(
        "--time-column",
        type=int,
        default=0,
        help="Time column for 2-D data; use -1 if there is no time column",
    )
    parser.add_argument("--channel", type=int, default=1, help="Signal column for 2-D data")

    # --- Spectral displays ---
    parser.add_argument("--nperseg", type=int, default=512, help="STFT window samples")
    parser.add_argument("--overlap", type=float, default=0.75, help="STFT overlap fraction")
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=100_000.0,
        help="Maximum displayed FFT/STFT frequency in Hz",
    )
    parser.add_argument("--dynamic-range-db", type=float, default=80.0)

    # --- Carrier band ---
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=40_000.0,
        help="Bandpass lower cutoff in Hz for the filtered waveform plot",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=48_000.0,
        help="Bandpass upper cutoff in Hz for the filtered waveform plot",
    )
    parser.add_argument(
        "--bandpass-order",
        type=int,
        default=4,
        help="Butterworth filter order (applied forward and backward)",
    )

    # --- Per-event zooms ---
    parser.add_argument(
        "--targeted-cwt",
        type=float,
        nargs="+",
        default=None,
        metavar="SECONDS",
        help=(
            "One or more event center times in ABSOLUTE recording seconds, using "
            "the same convention as --start-time/--end-time (for example "
            "--targeted-cwt 116.0775 116.5344). Each produces a separate 2 ms "
            f"{TARGETED_WAVELET} CWT figure. Times must fall inside the analyzed "
            "interval; windows near a boundary are clipped with a warning."
        ),
    )

    # --- Local phase reference ---
    parser.add_argument(
        "--local-phase-window-ms",
        type=float,
        default=LOCAL_PHASE_WINDOW_MS,
        help=(
            "Timescale in ms of the slowly varying local phase reference used to "
            f"rotate the corrected baseband (default {LOCAL_PHASE_WINDOW_MS:g}). "
            "Interpreted as the FWHM of a zero-phase Gaussian smoother. Must be "
            "much longer than the events being looked for."
        ),
    )
    parser.add_argument(
        "--compare-local-phase-windows",
        type=float,
        nargs="+",
        default=None,
        metavar="MS",
        help=(
            "Generate a diagnostic figure comparing locally rotated Q for these "
            "reference windows in ms, for example "
            "--compare-local-phase-windows 5 10 20 50."
        ),
    )
    parser.add_argument(
        "--local-phase-focus",
        type=float,
        nargs="+",
        default=None,
        metavar="SECONDS",
        help=(
            "Absolute recording times to zoom for a reference-window comparison "
            "figure (one per timestamp), for example --local-phase-focus 116.344."
        ),
    )
    parser.add_argument(
        "--local-phase-focus-span-ms",
        type=float,
        default=LOCAL_PHASE_FOCUS_SPAN_MS,
        help=(
            "Total width in ms of each --local-phase-focus window "
            f"(default {LOCAL_PHASE_FOCUS_SPAN_MS:g})."
        ),
    )
    parser.add_argument(
        "--local-phase-mask-targets",
        action="store_true",
        help=(
            "Exclude +/-"
            f"{LOCAL_PHASE_MASK_HALF_WIDTH_S * 1e3:g} ms around each --targeted-cwt "
            "and --local-phase-focus timestamp when estimating the local phase "
            "reference, so a strong event cannot bias the reference and rotate "
            "itself away. The reference is interpolated across the gap."
        ),
    )
    parser.add_argument(
        "--validate-local-phase",
        action="store_true",
        help=(
            "Run the synthetic 1.5 ms phase-event retention check for the "
            "comparison windows and print the results."
        ),
    )

    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


# ==========================================================================
#  7 - Pipeline
# ==========================================================================


def load_recording(
    args: argparse.Namespace, script_dir: Path
) -> tuple[np.ndarray, np.ndarray, float, str, float | None]:
    """Resolve the input, cut the interval, and convert counts to millivolts.

    Returns (t, x, fs, unit, mv_scale). ``unit`` is "mV" when the JSON carries a
    calibration and "ADC counts" otherwise, and it propagates into every axis
    label downstream.
    """
    input_path = (
        args.input.expanduser().resolve()
        if args.input is not None
        else script_dir / f"{RECORDING_STEM}.bin"
    )
    fs_arg = args.fs
    mv_scale: float | None = None
    if input_path.suffix.lower() == ".bin":
        metadata_path = (
            args.metadata_json.expanduser().resolve()
            if args.metadata_json is not None
            else input_path.with_suffix(".json")
        )
        metadata = load_metadata(metadata_path)
        if fs_arg is None:
            fs_arg = sampling_rate_from_metadata(metadata, metadata_path.name)
        mv_scale = millivolt_scale_from_metadata(metadata)

    data = load_array(input_path, args.binary_dtype)
    total_samples = data.shape[0]
    if data.ndim == 1 and fs_arg is not None:
        duration_s = total_samples / fs_arg
        if args.start_time >= duration_s:
            raise ValueError(
                f"Requested start time {args.start_time:g} s is outside the "
                f"recording (duration {duration_s:.6f} s)."
            )
    t, x, fs = extract_interval(
        data,
        args.start_time,
        args.end_time,
        fs_arg,
        args.time_column,
        args.channel,
    )
    # Scale once, here, so every amplitude downstream is already physical.
    if mv_scale is not None:
        x = x * mv_scale
        unit = "mV"
    else:
        unit = "ADC counts"
    return t, x, fs, unit, mv_scale


def build_exclusion_mask(
    args: argparse.Namespace, t: np.ndarray, fs: float
) -> tuple[np.ndarray | None, list[tuple[float, float]]]:
    """Mark +/-2 ms around each named timestamp as off-limits to the reference.

    The local reference is estimated from the very signal that may contain the
    events. A strong event can therefore drag the reference toward itself and
    end up partly rotated away -- masking it out breaks that circularity.
    """
    mask_times = list(args.targeted_cwt or []) + list(args.local_phase_focus or [])
    if not args.local_phase_mask_targets:
        return None, []
    if not mask_times:
        print(
            "WARNING: --local-phase-mask-targets was given but no --targeted-cwt "
            "or --local-phase-focus timestamps were supplied; no samples excluded."
        )
        return None, []

    exclusion_mask = np.zeros(t.size, dtype=bool)
    excluded_intervals: list[tuple[float, float]] = []
    for center_time in mask_times:
        lo = max(
            0, int(round((center_time - LOCAL_PHASE_MASK_HALF_WIDTH_S - t[0]) * fs))
        )
        hi = min(
            t.size, int(round((center_time + LOCAL_PHASE_MASK_HALF_WIDTH_S - t[0]) * fs))
        )
        if hi > lo:
            exclusion_mask[lo:hi] = True
            excluded_intervals.append((float(t[lo]), float(t[hi - 1])))
    return exclusion_mask, excluded_intervals


def report_acquisition(
    t: np.ndarray,
    x: np.ndarray,
    fs: float,
    mv_scale: float | None,
    window_s: float,
    hop_s: float,
    energy_ratio: float,
    carrier_hz: float,
    low_hz: float,
    high_hz: float,
) -> None:
    """What was loaded, in what units, and how the band compares to the whole."""
    print(f"Loaded {x.size:,} raw samples from {t[0]:.6f} to {t[-1]:.6f} s")
    print(f"Sampling rate: {fs:,.6f} Hz")
    if mv_scale is not None:
        print(f"Amplitude unit: mV ({mv_scale:.9g} mV per ADC count)")
    else:
        print("Amplitude unit: ADC counts (JSON lacks range_full_scale_mV/max_adc)")
    print(f"STFT window: {window_s * 1e6:,.3f} us")
    print(f"STFT hop:    {hop_s * 1e6:,.3f} us")
    print(
        f"Energy ratio ({low_hz / 1e3:g}-{high_hz / 1e3:g} kHz "
        f"band / raw, mean removed): {energy_ratio:.6f} ({energy_ratio * 100:.4f}%)"
    )
    print(
        f"Estimated carrier ({low_hz / 1e3:g}-"
        f"{high_hz / 1e3:g} kHz band, Hilbert phase slope): "
        f"{carrier_hz:,.3f} Hz"
    )


def report_demodulation(demod: dict, carrier_hz: float, unit: str) -> None:
    """The two-pass carrier correction and the amplitude it recovers."""
    print(f"Estimated carrier: {carrier_hz:,.3f} Hz")
    print(
        "Residual baseband phase slope: "
        f"{demod['residual_phase_slope_rad_s']:,.6f} rad/s"
    )
    print(f"Residual frequency offset: {demod['residual_offset_hz']:,.6f} Hz")
    print(f"Corrected carrier estimate: {demod['corrected_carrier_hz']:,.3f} Hz")
    print(
        "Demodulating at the corrected carrier; leftover offset after the "
        f"second pass: {demod['final_offset_hz']:.3e} Hz "
        f"(slope {demod['final_phase_slope_rad_s']:.3e} rad/s)"
    )
    print(f"Demodulated amplitude median: {demod['amplitude_median']:,.6f} {unit}")
    print(f"Demodulated amplitude MAD:    {demod['amplitude_mad']:,.6f} {unit}")


def report_local_phase(
    demod: dict, unit: str, excluded_intervals: list[tuple[float, float]]
) -> None:
    """Everything about the local reference, including its failure modes.

    The headline is the slow-component pair: overall Q MAD barely moves, but the
    slow part collapses, which is exactly the intended effect.
    """
    print(f"Global background phase phi_ref: {demod['reference_phase_rad']:.6f} rad")
    print(
        f"Local phase-reference window:       "
        f"{demod['local_window_ms']:.3f} ms (Gaussian FWHM, zero phase)"
    )
    print(
        f"Reference-magnitude median:         "
        f"{demod['reference_magnitude_median']:.6f} {unit}"
    )
    print(
        f"Low-confidence reference samples:   "
        f"{demod['low_confidence_percent']:.4f} %"
    )
    print(f"Global Q robust spread:             {demod['q_global_spread']:.6f} {unit}")
    print(f"Local Q robust spread:              {demod['q_local_spread']:.6f} {unit}")
    print(
        f"Global Q slow-component spread:     {demod['q_global_slow_spread']:.6f} {unit}"
    )
    print(
        f"Local Q slow-component spread:      {demod['q_local_slow_spread']:.6f} {unit}"
    )
    print(
        f"Local rotation amplitude error max: "
        f"{demod['amplitude_rotation_error']:.3e} {unit}"
    )
    if demod["q_local_slow_spread"] > 0:
        print(
            "Slow-wander suppression (global/local slow spread): "
            f"{demod['q_global_slow_spread'] / demod['q_local_slow_spread']:.1f}x"
        )
    if excluded_intervals:
        print(
            f"Reference excluded {demod['excluded_samples']:,} samples in "
            f"{len(excluded_intervals)} interval(s):"
        )
        for lo, hi in excluded_intervals:
            print(f"  {lo:.6f}-{hi:.6f} s")


def report_window_comparison(summary: list[dict[str, float]], unit: str) -> None:
    """Per-window spreads. Flattest is not best -- see save_local_phase_window_comparison."""
    print("Reference-window comparison (full interval):")
    for entry in summary:
        print(
            f"  {entry['window_ms']:>6.1f} ms | Q MAD {entry['q_spread']:8.4f} {unit}"
            f" | slow MAD {entry['q_slow_spread']:8.4f} {unit}"
            f" | low-confidence {entry['low_confidence_percent']:6.3f} %"
        )


def report_validation(results: list[dict[str, float]]) -> None:
    """Synthetic retention, flagging any window that eats the injected event."""
    print("Synthetic 1.5 ms phase-event retention (vs true-background rotation):")
    for entry in results:
        flag = (
            "  <-- WARNING: reference suppresses the event"
            if entry["peak_retention"] < EVENT_RETENTION_WARN_RATIO
            else ""
        )
        print(
            f"  {entry['window_ms']:>6.1f} ms | peak {entry['peak_retention'] * 100:6.1f} %"
            f" | energy {entry['energy_retention'] * 100:6.1f} %"
            f" | shift {entry['peak_shift_ms']:+.4f} ms"
            f" | slow-wander suppression {entry['slow_spread_reduction']:7.1f}x{flag}"
        )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.overlap < 1.0:
        raise ValueError("--overlap must satisfy 0 <= overlap < 1.")
    if args.max_frequency <= 0 or args.dynamic_range_db <= 0:
        raise ValueError("Frequency limit and dynamic range must be positive.")

    script_dir = Path(__file__).resolve().parent
    t, x, fs, unit, mv_scale = load_recording(args, script_dir)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else script_dir / "bubble_raw_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform_path = output_dir / "01_raw_waveform.html"
    fft_path = output_dir / "02_raw_fft.html"
    spectrogram_path = output_dir / "03_raw_stft_spectrogram.html"
    bandpassed_path = output_dir / "04_bandpassed_waveform.html"
    hilbert_path = output_dir / "05_hilbert_and_transient_energy.html"
    demodulation_path = output_dir / "06_carrier_demodulation.html"
    wavelet_path = output_dir / "07_wavelet_comparison_cwt.html"
    energy_comparison_path = output_dir / "08_transient_energy_comparison.html"
    local_phase_path = output_dir / "09_local_phase_diagnostics.html"
    window_comparison_path = output_dir / "10_local_phase_window_comparison.html"

    # --- Stage 1: the raw signal, as recorded ---
    save_waveform(t, x, waveform_path, unit)
    save_fft(x, fs, fft_path, args.max_frequency, unit)
    window_s, hop_s = save_spectrogram(
        t,
        x,
        fs,
        spectrogram_path,
        args.nperseg,
        args.overlap,
        args.max_frequency,
        args.dynamic_range_db,
        unit,
    )

    # --- Stage 2: isolate the carrier band and form the analytic signal ---
    x_filtered = bandpass_filter(
        x, fs, args.bandpass_low, args.bandpass_high, args.bandpass_order
    )
    save_bandpassed_waveform(
        t, x_filtered, args.bandpass_low, args.bandpass_high, bandpassed_path, unit
    )
    # Computed once and shared: the envelope, the carrier estimate, and the
    # demodulator all need this same transform.
    analytic = signal.hilbert(x_filtered)
    frame_samples, hop_samples, baseline_frames = save_hilbert_and_transient_energy(
        t,
        x,
        np.abs(analytic),
        fs,
        args.bandpass_low,
        args.bandpass_high,
        hilbert_path,
        unit,
    )

    # --- Stage 3: demodulate, then rotate into the local phase frame ---
    carrier_hz = estimate_carrier_frequency(analytic, fs)
    exclusion_mask, excluded_intervals = build_exclusion_mask(args, t, fs)
    demod = analyze_carrier_demodulation(
        analytic, fs, carrier_hz, args.local_phase_window_ms, exclusion_mask
    )
    save_carrier_demodulation(t, demod, demodulation_path, unit)
    save_local_phase_diagnostics(t, demod, local_phase_path, unit)

    window_summary: list[dict[str, float]] = []
    if args.compare_local_phase_windows:
        window_summary = save_local_phase_window_comparison(
            t,
            demod,
            list(args.compare_local_phase_windows),
            window_comparison_path,
            unit,
            exclusion_mask,
        )

    focus_results: list[tuple[float, Path, dict[str, float]]] = []
    focus_windows = list(
        args.compare_local_phase_windows or LOCAL_PHASE_COMPARE_WINDOWS_MS
    )
    for center_time in args.local_phase_focus or []:
        if not (t[0] <= center_time <= t[-1]):
            print(
                f"WARNING: --local-phase-focus {center_time:g} s lies outside the "
                f"analyzed interval {t[0]:.6f}-{t[-1]:.6f} s; skipping."
            )
            continue
        focus_path = output_dir / f"11_local_phase_focus_{center_time:.6f}s.html"
        peaks = save_local_phase_focus(
            t,
            demod,
            focus_windows,
            center_time,
            args.local_phase_focus_span_ms,
            focus_path,
            unit,
            exclusion_mask,
        )
        focus_results.append((center_time, focus_path, peaks))

    validation_results: list[dict[str, float]] = []
    if args.validate_local_phase:
        validation_results = validate_local_phase(fs, focus_windows)

    # --- Stage 4: independent detectors on the untouched signal ---
    save_transient_energy_comparison(
        t, x, x_filtered, fs, args.bandpass_low, args.bandpass_high,
        energy_comparison_path,
    )
    wavelet_summary = save_wavelet_comparison(
        t, x_filtered, fs, args.bandpass_low, args.bandpass_high, wavelet_path
    )

    targeted_results: list[tuple[float, Path, dict]] = []
    for center_time in args.targeted_cwt or []:
        if not (t[0] <= center_time <= t[-1]):
            print(
                f"WARNING: --targeted-cwt {center_time:g} s lies outside the "
                f"analyzed interval {t[0]:.6f}-{t[-1]:.6f} s; skipping. "
                "Timestamps are absolute recording seconds."
            )
            continue
        targeted_path = output_dir / f"targeted_cwt_{center_time:.6f}s.html"
        info = save_targeted_cwt(
            t, x_filtered, fs, center_time, args.bandpass_low, args.bandpass_high,
            targeted_path, unit,
        )
        if info["clipped"]:
            print(
                f"WARNING: the 2 ms window at {center_time:.6f} s reaches an "
                f"interval boundary; it was clipped to {info['duration_ms']:.3f} ms."
            )
        targeted_results.append((center_time, targeted_path, info))

    # --- Report ---
    # Parseval-style check: what fraction of the AC energy lives in the band.
    # Both sides are mean-removed so a DC offset cannot inflate the denominator.
    raw_energy = float(np.sum((x - np.mean(x)) ** 2))
    band_energy = float(np.sum(x_filtered**2))
    energy_ratio = band_energy / raw_energy

    report_acquisition(
        t, x, fs, mv_scale, window_s, hop_s, energy_ratio, carrier_hz,
        args.bandpass_low, args.bandpass_high,
    )
    print(f"Saved: {waveform_path.resolve()}")
    print(f"Saved: {fft_path.resolve()}")
    print(f"Saved: {spectrogram_path.resolve()}")
    print(f"Saved: {bandpassed_path.resolve()}")
    print(f"Created: {hilbert_path.resolve()}")
    print(
        f"Hilbert band: {args.bandpass_low / 1e3:g}-{args.bandpass_high / 1e3:g} kHz"
    )
    print(f"Frame:    {frame_samples} samples ({frame_samples / fs * 1e6:,.3f} us)")
    print(f"Hop:      {hop_samples} samples ({hop_samples / fs * 1e6:,.3f} us)")
    print(
        f"Baseline: {baseline_frames} frames "
        f"({baseline_frames * hop_samples / fs * 1e3:,.3f} ms moving median)"
    )
    print(f"Created: {demodulation_path.resolve()}")
    report_demodulation(demod, carrier_hz, unit)
    report_local_phase(demod, unit, excluded_intervals)
    print(f"Created: {local_phase_path.resolve()}")
    if window_summary:
        print(f"Created: {window_comparison_path.resolve()}")
        report_window_comparison(window_summary, unit)
    for center_time, focus_path, peaks in focus_results:
        print(f"Created: {focus_path.resolve()}")
        print(
            f"  Focus at {center_time:.6f} s over "
            f"{peaks['start_time']:.6f}-{peaks['stop_time']:.6f} s, peak |Q|:"
        )
        for window_ms in focus_windows:
            key = f"peak_{window_ms:g}ms"
            if key in peaks:
                print(
                    f"    {window_ms:>6.1f} ms reference: {peaks[key]:8.4f} {unit} at "
                    f"{peaks[f'peak_time_ms_{window_ms:g}ms']:+.4f} ms"
                )
    if validation_results:
        report_validation(validation_results)
    print(f"Created: {energy_comparison_path.resolve()}")
    print(f"Created: {wavelet_path.resolve()}")
    print(
        f"CWT grid: {CWT_FREQUENCY_BINS} bins over "
        f"{args.bandpass_low / 1e3:g}-{args.bandpass_high / 1e3:g} kHz"
    )
    for wavelet, min_scale, max_scale, coi_seconds in wavelet_summary:
        print(
            f"  {wavelet:<12} scales {min_scale:7.2f}-{max_scale:7.2f} samples, "
            f"cone of influence {coi_seconds * 1e3:.3f} ms"
        )
    if args.targeted_cwt:
        print(f"Targeted CWT ({TARGETED_WAVELET}, absolute recording time):")
        for center_time, targeted_path, info in targeted_results:
            print(
                f"  {center_time:.6f} s -> samples {info['start_index']}:"
                f"{info['stop_index']} = {info['start_time']:.6f}-"
                f"{info['stop_time']:.6f} s ({info['duration_ms']:.3f} ms)"
                f"{' [clipped]' if info['clipped'] else ''}"
            )
            print(f"    Created: {targeted_path.resolve()}")
        if not targeted_results:
            print("  (no valid timestamps; nothing generated)")


if __name__ == "__main__":
    main()
