"""Carrier-demodulation experiment on one abdomen fiber.

The other pipelines treat the 200 Hz band as acoustics. This one treats it as a carrier
the heartbeat modulates, so the beat lives in the tone's amplitude:

    x(t) -> bandpass -> Hilbert -> mix down by f_c and low-pass -> A(t) = |u(t)|

Demodulation adapted from plot_raw_bubble_event.py, retuned for a 200-230 Hz carrier at
5 kHz with a 25 Hz baseband low-pass (the beat is only 1.8-3 Hz).

A(t) is beat-detected two ways with v7: envelope-aware, which skips v7's Shannon stage
since A(t) already is an envelope, and non-aware, which is stock v7. Each HR figure also
shows the plain bandpassed fiber as a control, so top-vs-bottom isolates the demodulation.

The reference branch is untouched: NST mic -> v7 -> nst_dropouts.csv drift correction.
This experiment owns .out/<patient>/demodulation/ and writes nothing else.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, hilbert, sosfiltfilt

from analyze.data import Audio, FiberData, FiberPair, load_data, windowed
from analyze.drift import correct_drift
from analyze.filters import bp_filter
from analyze.hr import fHROutput, fiber_beats, sot_beats
from analyze.hr.detect_v7 import v7_beat_detector, v7_envelope_aware_beat_detector
from analyze.pipeline import Pipeline
# Private, but these are the repo's HR-comparison format -- the same helpers
# plot_hr_comparison composes, so the figures here match it instead of reimplementing it.
from analyze.plot_hr import (
    _agreement_stats, _agreement_values, _current_hr_func, _figsize, _hr_ylim, _plot_hr_axis,
)
from analyze.sot import SOTResult, load_sot
from constants import FETAL_BPM_RANGE, NST_DRIFT_LOG_FILE, PROJECT_DIR

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Plotly is required for the demodulation experiment's HTML output. Install it "
        "with:\n    .venv/bin/pip install plotly"
    ) from exc


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEMOD_FIBER = "belly_1B"          # _resolve_fiber strips the "belly_" to get "1B"
DEMOD_FIBERS = ("1B", "2A", "2B", "2C", "2D")
DEMOD_BANDPASS_HZ = (200.0, 230.0)
DEMOD_BANDPASS_ORDER = 4

DEMOD_BASEBAND_HZ = 25.0          # only has to pass a 1.8-3 Hz beat
DEMOD_LOWPASS_ORDER = 4

DEMOD_EDGE_EXCLUSION_S = 0.5

DEMOD_AMPLITUDE_FLOOR_PCT = 25.0

PLOT_MAX_POINTS = 40_000          # 3M samples inlined verbatim is a browser-hostile file

ZOOM_SPAN_S = 3.0
ZOOM_START_S = 375.6              # None centres on the median beat; out-of-range is clamped

HR_OUTLIER_HALF_WINDOW = 5        # each side, excluding the point itself
HR_OUTLIER_TOLERANCE = 0.20       # reject beyond +/-20% of the neighbour median
HR_SMOOTH_WINDOW = 20             # must match the window _inst_hr_v2 hardcodes


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemodulationAgreement:
    """One v7 mode vs corrected NST, using the existing smoothed HR score.

    Full precision is retained; CSV and optimized figure share the legacy display
    precision via ``display_values``. Undefined agreement is represented by NaN.
    """

    r: float
    mae: float

    @property
    def display_values(self) -> Tuple[str, str]:
        return f"{self.r:.2f}", f"{self.mae:.1f}"


@dataclass
class DemodulationResult:
    """Everything the demodulation produced, accumulated through the pipeline stages."""

    fiber: str
    bandpass_hz: Tuple[float, float]
    baseband_hz: float

    bandpassed: Audio          # x_BP(t), the carrier band on the original time axis
    amplitude: Audio           # A(t) = |u(t)|, same time axis

    initial_carrier_hz: float
    residual_offset_hz: float
    corrected_carrier_hz: float
    final_residual_hz: float

    envelope_aware: Optional[fHROutput] = None
    non_aware: Optional[fHROutput] = None
    bandpass_only: Optional[fHROutput] = None   # control: v7 on x_BP(t), no demodulation
    agreement: Optional[DemodulationAgreement] = None
    non_aware_agreement: Optional[DemodulationAgreement] = None


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def _check_below_nyquist(label: str, value_hz: float, hz: float) -> None:
    """Reject a cutoff at or above Nyquist before scipy turns it into a cryptic error."""
    nyquist = hz / 2.0
    if not (0.0 < value_hz < nyquist):
        raise ValueError(
            f"{label} must satisfy 0 < f < Nyquist ({nyquist:g} Hz) for a {hz:g} Hz "
            f"signal; got {value_hz:g} Hz."
        )


def _check_band_below_nyquist(label: str, band: Tuple[float, float], hz: float) -> None:
    low, high = float(band[0]), float(band[1])
    nyquist = hz / 2.0
    if not (0.0 < low < high < nyquist):
        raise ValueError(
            f"{label} must satisfy 0 < low < high < Nyquist ({nyquist:g} Hz) for a "
            f"{hz:g} Hz signal; got {low:g}-{high:g} Hz."
        )


def _resolve_fiber(fiber: str) -> str:
    """Accept ``belly_1B`` as a spelling of the abdomen bundle's ``1B``."""
    name = str(fiber)
    return name[len("belly_"):] if name.lower().startswith("belly_") else name


def _interior_mask(n: int, hz: float, edge_s: float) -> np.ndarray:
    """Samples at least ``edge_s`` from either end, or everything if that leaves nothing.

    The fallback keeps a short clip usable: an empty mask would make the fit undefined
    rather than just edge-contaminated.
    """
    t_rel = np.arange(n, dtype=float) / hz
    mask = (t_rel >= edge_s) & (t_rel <= t_rel[-1] - edge_s if n else False)
    if np.count_nonzero(mask) < 2:
        return np.ones(n, dtype=bool)
    return mask


def estimate_carrier_hz(
        analytic: np.ndarray,
        hz: float,
        edge_s: float = DEMOD_EDGE_EXCLUSION_S,
        amplitude_floor_pct: float = DEMOD_AMPLITUDE_FLOOR_PCT,
) -> Tuple[float, np.ndarray]:
    """Carrier frequency from the slope of the unwrapped analytic phase.

    A tone at f_c has phase 2*pi*f_c*t + phi_0, so a line fit gives f_c = slope / (2*pi).
    Over a real band this is the amplitude-weighted mean instantaneous frequency -- fine
    while one component dominates, misleading if two comparable tones share the band.

    Unwrapping happens over the whole array before anything is dropped: unwrap needs every
    consecutive difference, so masking first could hide a 2*pi step. Only the fit is
    restricted, to samples clear of the edges and above ``amplitude_floor_pct`` of |z|.

    Returns ``(carrier_hz, fit_mask)``.
    """
    analytic = np.asarray(analytic)
    n = analytic.size
    if n < 2:
        raise ValueError("Carrier estimation needs at least 2 samples.")

    phase = np.unwrap(np.angle(analytic))
    t_rel = np.arange(n, dtype=float) / hz

    valid = _interior_mask(n, hz, edge_s)
    magnitude = np.abs(analytic)
    if 0.0 < amplitude_floor_pct < 100.0:
        floor = np.percentile(magnitude[valid], amplitude_floor_pct)
        strong = valid & (magnitude >= floor)
        # Only tighten if enough survives to still define a line.
        if np.count_nonzero(strong) >= 2:
            valid = strong

    slope = np.polyfit(t_rel[valid], phase[valid], 1)[0]
    return float(slope / (2.0 * np.pi)), valid


def demodulate_to_baseband(
        analytic: np.ndarray,
        hz: float,
        carrier_hz: float,
        cutoff_hz: float = DEMOD_BASEBAND_HZ,
        order: int = DEMOD_LOWPASS_ORDER,
) -> np.ndarray:
    """Mix an analytic signal down to 0 Hz and low-pass I and Q identically.

        u(t) = LPF{ z(t) * e^(-j*2*pi*f_c*t) }

    No factor of 2: z is single-sided already, so there is no image at -2*f_c to halve
    away. Time runs from the start of the array so the exponent stays small.

    I and Q get the same zero-phase filter. Filtering them differently would rotate the
    phase and corrupt the residual the second pass is trying to measure.
    """
    _check_below_nyquist("Baseband low-pass cutoff", cutoff_hz, hz)
    n = np.asarray(analytic).size
    t_rel = np.arange(n, dtype=float) / hz
    baseband = np.asarray(analytic) * np.exp(-1j * 2.0 * np.pi * carrier_hz * t_rel)
    sos = butter(order, cutoff_hz, btype="lowpass", fs=hz, output="sos")
    return sosfiltfilt(sos, np.real(baseband)) + 1j * sosfiltfilt(sos, np.imag(baseband))


def residual_offset_hz(baseband: np.ndarray, hz: float, valid: np.ndarray) -> float:
    """How far the mix-down missed the true carrier.

    Mixing at f_c + delta leaves the baseband turning at 2*pi*delta rad/s, so fitting that
    slope recovers delta. Measured after the low-pass, which is why it beats the wideband
    estimate: the noise that biased the first fit is gone by then.
    """
    n = np.asarray(baseband).size
    t_rel = np.arange(n, dtype=float) / hz
    phase = np.unwrap(np.angle(baseband))
    return float(np.polyfit(t_rel[valid], phase[valid], 1)[0] / (2.0 * np.pi))


def demodulate_audio(
        bandpassed: Audio,
        baseband_hz: float = DEMOD_BASEBAND_HZ,
        edge_s: float = DEMOD_EDGE_EXCLUSION_S,
        amplitude_floor_pct: float = DEMOD_AMPLITUDE_FLOOR_PCT,
        lowpass_order: int = DEMOD_LOWPASS_ORDER,
) -> Tuple[Audio, float, float, float, float]:
    """Two-pass complex demodulation of an already-bandpassed Audio.

    Pass 1 mixes at the phase-fit estimate and measures the leftover; pass 2 mixes at the
    corrected carrier. Returns ``(amplitude, initial, residual, corrected, final_residual)``.
    Every step is sample-wise, so A(t) lands on the input's own time axis unchanged.
    """
    hz = float(bandpassed.hz)
    _check_below_nyquist("Baseband low-pass cutoff", baseband_hz, hz)

    analytic = hilbert(np.asarray(bandpassed.data, dtype=float))
    initial_carrier, valid = estimate_carrier_hz(analytic, hz, edge_s, amplitude_floor_pct)

    first = demodulate_to_baseband(analytic, hz, initial_carrier, baseband_hz, lowpass_order)
    residual = residual_offset_hz(first, hz, valid)
    corrected_carrier = initial_carrier + residual

    second = demodulate_to_baseband(analytic, hz, corrected_carrier, baseband_hz, lowpass_order)
    final_residual = residual_offset_hz(second, hz, valid)

    amplitude = np.abs(second)   # A(t) = sqrt(I^2 + Q^2), non-negative by construction
    return (
        Audio(bandpassed.time, bandpassed.hz, amplitude),
        initial_carrier,
        residual,
        corrected_carrier,
        final_residual,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _decimate_for_plot(t: np.ndarray, y: np.ndarray,
                       max_points: int = PLOT_MAX_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    """Thin a trace for HTML, keeping each bucket's min and max.

    Plain striding would step over one-sample peaks, which on a beat trace are the point.
    Keeping both extremes preserves the visual envelope at a fraction of the file size.
    """
    n = y.size
    if n <= max_points:
        return t, y
    buckets = max(1, max_points // 2)
    size = int(np.ceil(n / buckets))
    padded = np.append(y, np.full(buckets * size - n, np.nan))
    grid = padded.reshape(buckets, size)
    low = np.argmin(np.where(np.isnan(grid), np.inf, grid), axis=1)
    high = np.argmax(np.where(np.isnan(grid), -np.inf, grid), axis=1)
    base = np.arange(buckets) * size
    index = np.unique(np.concatenate([base + low, base + high]))
    index = index[index < n]
    return t[index], y[index]


def _write_demodulated_html(result: DemodulationResult, path: Path) -> None:
    """Output 1: the second-pass demodulated amplitude A(t), with the run's parameters."""
    t, y = _decimate_for_plot(np.asarray(result.amplitude.time),
                              np.asarray(result.amplitude.data))
    low, high = result.bandpass_hz
    subtitle = (
        f"fiber {result.fiber} &middot; bandpass {low:g}-{high:g} Hz &middot; "
        f"baseband LPF {result.baseband_hz:g} Hz &middot; "
        f"carrier {result.initial_carrier_hz:.4f} Hz initial "
        f"&rarr; {result.corrected_carrier_hz:.4f} Hz corrected "
        f"(residual {result.residual_offset_hz:+.4f} Hz, "
        f"final {result.final_residual_hz:+.3e} Hz)"
    )
    fig = go.Figure(
        go.Scattergl(x=t, y=y, mode="lines", line={"width": 1}, name="A(t) = |u(t)|")
    )
    fig.update_layout(
        title=f"Demodulated fiber amplitude A(t) = |u(t)|<br><sub>{subtitle}</sub>",
        xaxis_title="Time (s)",
        yaxis_title="A(t) = |u(t)|",
        template="plotly_white",
        hovermode="x unified",
        # The same numbers as machine-readable metadata, not just title text.
        meta={
            "fiber": result.fiber,
            "bandpass_hz": [low, high],
            "baseband_lowpass_hz": result.baseband_hz,
            "initial_carrier_hz": result.initial_carrier_hz,
            "residual_offset_hz": result.residual_offset_hz,
            "corrected_carrier_hz": result.corrected_carrier_hz,
            "final_residual_hz": result.final_residual_hz,
        },
    )
    fig.write_html(path, include_plotlyjs=True)


def _write_bandpass_html(result: DemodulationResult, path: Path,
                         zoom_span_s: float = ZOOM_SPAN_S,
                         zoom_start_s: Optional[float] = ZOOM_START_S) -> None:
    """Output 4: the bandpassed fiber marked with each mode's beats, twice over.

    Rows 1-2 show the whole window thinned (where the beats fall over the record); rows 3-4
    a short excerpt at full rate (what the waveform looks like under a beat). Only the
    latter can resolve the carrier -- see ZOOM_SPAN_S.

    Every panel draws the same waveform, so within a pair only the dots differ. Dot heights
    come from the full-rate waveform, not the thinned trace.
    """
    t_full = np.asarray(result.bandpassed.time, dtype=float)
    y_full = np.asarray(result.bandpassed.data, dtype=float)
    t_plot, y_plot = _decimate_for_plot(t_full, y_full)
    low, high = result.bandpass_hz
    hz = float(result.bandpassed.hz)

    modes = (
        ("Envelope-aware v7", result.envelope_aware),
        ("Non-aware v7 (standard preprocessing)", result.non_aware),
    )

    t_lo, t_hi = float(t_full[0]), float(t_full[-1])
    if zoom_start_s is None:
        # Centre on a stretch that has beats, falling back to the middle of the record.
        ref = np.asarray([] if modes[0][1] is None else modes[0][1].fetal_beats, dtype=float)
        centre = float(np.median(ref)) if ref.size else float(t_full[len(t_full) // 2])
        z0 = centre - zoom_span_s / 2.0
    else:
        z0 = float(zoom_start_s)

    # Slide rather than truncate, so the excerpt is always the requested length. The inner
    # max() guards a record shorter than one span.
    z0 = min(max(z0, t_lo), max(t_lo, t_hi - zoom_span_s))
    if zoom_start_s is not None and abs(z0 - float(zoom_start_s)) > 1e-9:
        print(f"[demodulation] zoom start {float(zoom_start_s):g}s is outside "
              f"{t_lo:g}-{t_hi:g}s; clamped to {z0:g}s")
    z1 = min(t_hi, z0 + zoom_span_s)
    zoom = (t_full >= z0) & (t_full <= z1)
    t_zoom, y_zoom = t_full[zoom], y_full[zoom]

    titles = []
    for label, o in modes:
        titles.append(f"{label} &mdash; {0 if o is None else len(o.fetal_beats)} beats, "
                      f"whole window (thinned: carrier not resolvable here)")
    for label, _o in modes:
        titles.append(f"{label} &mdash; {z0:.2f}-{z1:.2f} s at the full {hz:g} Hz "
                      f"sample rate (carrier visible)")

    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.06,
                        subplot_titles=titles)

    for i, (label, output) in enumerate(modes):
        beats = np.asarray([] if output is None else output.fetal_beats, dtype=float)
        for row, (tt, yy, bb) in (
            (i + 1, (t_plot, y_plot, beats)),
            (i + 3, (t_zoom, y_zoom, beats[(beats >= z0) & (beats <= z1)])),
        ):
            fig.add_trace(
                go.Scattergl(x=tt, y=yy, mode="lines", line={"width": 1},
                             name=f"{low:g}-{high:g} Hz fiber", legendgroup="wave",
                             showlegend=row == 1),
                row=row, col=1,
            )
            if bb.size:
                fig.add_trace(
                    go.Scattergl(
                        x=bb, y=np.interp(bb, t_full, y_full), mode="markers",
                        marker={"color": "red", "size": 6}, name=f"{label} beats",
                        legendgroup=label, showlegend=row <= 2,
                        hovertemplate="Beat: %{x:.4f} s<br>Amplitude: %{y:.4g}<extra></extra>",
                    ),
                    row=row, col=1,
                )
            fig.update_yaxes(title_text=f"{low:g}-{high:g} Hz", row=row, col=1)

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    # Two independent pairs, so zooming an overview does not drag the excerpt panels along.
    fig.update_xaxes(matches="x", row=2, col=1)
    fig.update_xaxes(matches="x3", row=4, col=1)
    fig.update_layout(
        title=(
            f"Bandpassed fiber {result.fiber} ({low:g}-{high:g} Hz) with v7 beat times "
            f"&mdash; identical waveform in every panel"
        ),
        template="plotly_white",
        hovermode="x unified",
        height=1180,
    )
    fig.write_html(path, include_plotlyjs=True)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def demodulation_demodulate(
        fiber: str = DEMOD_FIBER,
        bandpass_hz: Tuple[float, float] = DEMOD_BANDPASS_HZ,
        baseband_hz: float = DEMOD_BASEBAND_HZ,
        bandpass_order: int = DEMOD_BANDPASS_ORDER,
):
    """Pipeline stage: select the fiber, bandpass it, and demodulate to A(t)."""

    def run_demodulation_demodulate(data: FiberData) -> DemodulationResult:
        name = _resolve_fiber(fiber)
        if name not in data.abdomen:
            raise ValueError(
                f"demodulation: fiber {name!r} not in data (have {list(data.abdomen)})"
            )
        raw = data.abdomen[name]
        _check_band_below_nyquist("Fiber bandpass", bandpass_hz, float(raw.hz))

        # The repo's own bandpass; butter to match run_raw_bandpass, not bp_filter's cheby1.
        bandpassed = bp_filter(raw, bandpass_hz[0], bandpass_hz[1],
                               order=bandpass_order, filter_type="butter")
        amplitude, initial, residual, corrected, final = demodulate_audio(
            bandpassed, baseband_hz=baseband_hz,
        )

        assert len(amplitude.time) == len(raw.time), "demodulation changed the sample count"
        assert np.array_equal(amplitude.time, raw.time), "demodulation altered the time axis"

        print(f"[demodulation] fiber {name} @ {raw.hz} Hz, "
              f"bandpass {bandpass_hz[0]:g}-{bandpass_hz[1]:g} Hz, "
              f"baseband LPF {baseband_hz:g} Hz")
        print(f"[demodulation] initial carrier:    {initial:.6f} Hz")
        print(f"[demodulation] residual offset:    {residual:+.6f} Hz")
        print(f"[demodulation] corrected carrier:  {corrected:.6f} Hz")
        print(f"[demodulation] final residual:     {final:+.6e} Hz")

        return DemodulationResult(
            fiber=name,
            bandpass_hz=(float(bandpass_hz[0]), float(bandpass_hz[1])),
            baseband_hz=float(baseband_hz),
            bandpassed=bandpassed,
            amplitude=amplitude,
            initial_carrier_hz=initial,
            residual_offset_hz=residual,
            corrected_carrier_hz=corrected,
            final_residual_hz=final,
        )

    run_demodulation_demodulate.__name__ = "demodulation_demodulate"
    return run_demodulation_demodulate


def demodulation_envelope_aware_beats(out: Path, fetal_bpm=FETAL_BPM_RANGE):
    """Pipeline stage: v7 over A(t) with its Shannon-energy stage skipped."""

    def run_demodulation_envelope_aware_beats(result: DemodulationResult) -> DemodulationResult:
        detect = fiber_beats(v7_envelope_aware_beat_detector, Path(out), fetal_bpm=fetal_bpm)
        output = detect(FiberPair(None, result.amplitude))
        print(f"[demodulation] envelope-aware v7 beats: {len(output.fetal_beats)}")
        return replace(result, envelope_aware=output)

    run_demodulation_envelope_aware_beats.__name__ = "demodulation_envelope_aware_beats"
    return run_demodulation_envelope_aware_beats


def demodulation_non_aware_beats(out: Path, fetal_bpm=FETAL_BPM_RANGE):
    """Pipeline stage: the stock v7 over A(t), preprocessing untouched."""

    def run_demodulation_non_aware_beats(result: DemodulationResult) -> DemodulationResult:
        detect = fiber_beats(v7_beat_detector, Path(out), fetal_bpm=fetal_bpm)
        output = detect(FiberPair(None, result.amplitude))
        print(f"[demodulation] non-aware v7 beats:      {len(output.fetal_beats)}")
        return replace(result, non_aware=output)

    run_demodulation_non_aware_beats.__name__ = "demodulation_non_aware_beats"
    return run_demodulation_non_aware_beats


def demodulation_bandpass_v7_beats(out: Path, fetal_bpm=FETAL_BPM_RANGE):
    """Pipeline stage: stock v7 over the bandpassed fiber, no demodulation.

    The control for both HR figures: same fiber, band and scoring, so top-vs-bottom
    isolates the demodulation.
    """

    def run_demodulation_bandpass_v7_beats(result: DemodulationResult) -> DemodulationResult:
        detect = fiber_beats(v7_beat_detector, Path(out), fetal_bpm=fetal_bpm)
        output = detect(FiberPair(None, result.bandpassed))
        print(f"[demodulation] bandpass-only v7 beats:  {len(output.fetal_beats)}")
        return replace(result, bandpass_only=output)

    run_demodulation_bandpass_v7_beats.__name__ = "demodulation_bandpass_v7_beats"
    return run_demodulation_bandpass_v7_beats


def _agreement(sot_beats, pred_beats) -> str:
    """``r=..., MAE=... bpm`` from ``plot_hr._agreement_stats``.

    Reused rather than recomputed, so the summary cannot drift from the figure titles.
    """
    st, sy = _current_hr_func()(np.asarray(sot_beats, dtype=float), FETAL_BPM_RANGE)
    pt, py = _current_hr_func()(np.asarray(pred_beats, dtype=float), FETAL_BPM_RANGE)
    return _agreement_stats(st, sy, pt, py).replace("|", "").strip() or "n/a"


# ---------------------------------------------------------------------------
# Optional HR outlier rejection
# ---------------------------------------------------------------------------

def hr_outlier_mask(bpm: np.ndarray,
                    half_window: int = HR_OUTLIER_HALF_WINDOW,
                    tolerance: float = HR_OUTLIER_TOLERANCE) -> np.ndarray:
    """Boolean mask of HR points more than ``tolerance`` from their neighbour median.

    The window excludes the point itself -- including it would let a spike pull the median
    it is judged against. Near the ends it is whatever neighbours exist.

    Nothing is written back during the pass, so the mask describes the original trace and
    one rejection cannot change the verdict on the next.
    """
    bpm = np.asarray(bpm, dtype=float)
    n = bpm.size
    reject = np.zeros(n, dtype=bool)
    if n == 0:
        return reject

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        neighbours = np.concatenate([bpm[lo:i], bpm[i + 1:hi]])
        neighbours = neighbours[np.isfinite(neighbours)]
        if neighbours.size == 0:
            continue
        median = float(np.median(neighbours))
        if median <= 0.0 or not np.isfinite(bpm[i]):
            continue
        if abs(bpm[i] - median) > tolerance * median:
            reject[i] = True
    return reject


def moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    """``util.moving_average_v2`` with NaN treated as missing rather than contagious.

    The stock version convolves directly, so one NaN erases every output within a window of
    it -- 20 points for a single rejected beat. Summing the finite values and dividing by
    how many there were gives the same centred average over the same window. An entirely
    empty window yields NaN, which plots as a gap.
    """
    y = np.asarray(y, dtype=float)
    w = int(window)
    if w <= 1 or y.size == 0:
        return y.copy()
    w = min(w, y.size)                       # same length guard as moving_average_v2
    kernel = np.ones(w)
    valid = np.isfinite(y)
    sums = np.convolve(np.where(valid, y, 0.0), kernel, mode="same")
    counts = np.convolve(valid.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(counts > 0, sums / counts, np.nan)
    return out


def inst_hr_rejected(beats, band: Tuple[float, float],
                     half_window: int = HR_OUTLIER_HALF_WINDOW,
                     tolerance: float = HR_OUTLIER_TOLERANCE,
                     smooth_window: int = HR_SMOOTH_WINDOW):
    """Instantaneous HR with outliers blanked, as ``(t, bpm, stats)``.

    Mirrors ``plot_hr._inst_hr_v2`` -- sort, 60/IBI, drop out-of-band -- then rejects
    between the raw HR and the moving average. That is the only place it fits: earlier
    there is no HR to test, later the spike is already smeared across its window.

    Rejected points keep their timestamp and lose only their value, so nothing shifts.
    """
    beats = np.sort(np.asarray(beats, dtype=float))
    stats = {"total": 0, "rejected": 0, "reject_pct": 0.0, "coverage_pct": 0.0}
    if beats.size < 2:
        return np.array([]), np.array([]), stats

    bpm = 60.0 / np.clip(np.diff(beats), 1e-6, None)
    t = beats[1:]
    keep = (bpm >= band[0]) & (bpm <= band[1])
    bpm, t = bpm[keep], t[keep]

    reject = hr_outlier_mask(bpm, half_window, tolerance)
    cleaned = bpm.astype(float).copy()
    cleaned[reject] = np.nan
    smoothed = moving_average_nan(cleaned, smooth_window)

    total = int(bpm.size)
    n_rejected = int(np.count_nonzero(reject))
    stats = {
        "total": total,
        "rejected": n_rejected,
        "reject_pct": 100.0 * n_rejected / total if total else 0.0,
        # Two coverages, because they differ a lot: valid_pct is what survived rejection,
        # coverage_pct is how much of the *smoothed* trace has a value -- usually 100%,
        # since a 20-point window only empties where 20 consecutive points were rejected.
        # Reporting only the latter would make any rejection rate look harmless.
        "valid_pct": 100.0 * (total - n_rejected) / total if total else 0.0,
        "coverage_pct": 100.0 * np.count_nonzero(np.isfinite(smoothed)) / total if total else 0.0,
    }
    return t, smoothed, stats


def _agreement_masked(sot_t, sot_y, pred_t, pred_y, grid_hz: float = 1.0):
    """``r``, ``MAE`` and grid coverage, ignoring samples where either trace is missing.

    Same 1 Hz grid as ``plot_hr._agreement_stats``, so a gap-free trace returns exactly the
    number in the other figure titles. ``np.interp`` gives NaN for any grid point bracketed
    by a NaN, so blanked stretches drop out instead of being bridged by a made-up line.
    """
    sot_t, sot_y = np.asarray(sot_t, float), np.asarray(sot_y, float)
    pred_t, pred_y = np.asarray(pred_t, float), np.asarray(pred_y, float)
    if sot_t.size < 2 or pred_t.size < 2:
        return float("nan"), float("nan"), 0, 0

    lo = max(float(sot_t[0]), float(pred_t[0]))
    hi = min(float(sot_t[-1]), float(pred_t[-1]))
    if hi <= lo:
        return float("nan"), float("nan"), 0, 0

    grid = np.arange(lo, hi, 1.0 / grid_hz)
    if grid.size < 2:
        return float("nan"), float("nan"), 0, int(grid.size)

    a = np.interp(grid, sot_t, sot_y)
    b = np.interp(grid, pred_t, pred_y)
    both = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(both) < 2 or a[both].std() == 0 or b[both].std() == 0:
        return float("nan"), float("nan"), int(np.count_nonzero(both)), int(grid.size)

    r = float(np.corrcoef(a[both], b[both])[0, 1])
    mae = float(np.mean(np.abs(a[both] - b[both])))
    return r, mae, int(np.count_nonzero(both)), int(grid.size)


def _plot_hr_axis_traces(ax, sot_t, sot_y, pred_t, pred_y, pred_color: str,
                         sot_label: str, pred_label: str, title: str,
                         band: Tuple[float, float]) -> None:
    """``plot_hr._plot_hr_axis`` for traces that are already computed.

    Same styling and title stat. It exists because the stock one takes *beats* and derives
    the HR itself, which cannot express a trace with values selectively blanked.
    """
    r, mae, n_valid, n_grid = _agreement_masked(sot_t, sot_y, pred_t, pred_y)
    if np.isfinite(r):
        title += f"  |  r={r:.2f}, MAE={mae:.1f} bpm"
        if n_valid < n_grid:
            title += f" over {n_valid}/{n_grid}s"

    if sot_t.size:
        ax.plot(sot_t, sot_y, color='tab:red', lw=1.4, marker='o', ms=3, alpha=0.9,
                label=f'{sot_label} (median {float(np.nanmedian(sot_y)):.1f})')
    if pred_t.size and np.any(np.isfinite(pred_y)):
        ax.plot(pred_t, pred_y, color=pred_color, lw=1.1, marker='s', ms=3, alpha=0.8,
                label=f'{pred_label} (median {float(np.nanmedian(pred_y)):.1f})')

    ax.set_ylim(*_hr_ylim([(pred_t, pred_y), (sot_t, sot_y)], band))
    ax.set_ylabel("Instantaneous HR (BPM)", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(title, fontsize=9)


def _write_no_outlier_comparison(result: DemodulationResult, sot: SOTResult,
                                 out: Path) -> dict:
    """Extra output: the envelope-aware HR before and after outlier rejection.

    Both panels score against the same drift-corrected NST, so only the rejection differs.
    Returns the stats for the caller to print.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    band = FETAL_BPM_RANGE

    sot_t, sot_y = _current_hr_func()(np.asarray(sot.mic_beats, float), band)
    beats = np.asarray(result.envelope_aware.fetal_beats, float)
    raw_t, raw_y = _current_hr_func()(beats, band)
    clean_t, clean_y, stats = inst_hr_rejected(beats, band)

    r_before, mae_before, _, _ = _agreement_masked(sot_t, sot_y, raw_t, raw_y)
    r_after, mae_after, n_valid, n_grid = _agreement_masked(sot_t, sot_y, clean_t, clean_y)
    stats.update({"r_before": r_before, "mae_before": mae_before,
                  "r_after": r_after, "mae_after": mae_after,
                  "grid_valid": n_valid, "grid_total": n_grid})

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=_figsize(2), sharex=True,
                                         constrained_layout=True)
    _plot_hr_axis_traces(
        ax_top, sot_t, sot_y, raw_t, raw_y, 'tab:green',
        'Mic (SOT, drift-corrected)', 'Demodulated fiber (v7, envelope-aware)',
        "Fetal instantaneous HR — envelope-aware, original", band,
    )
    _plot_hr_axis_traces(
        ax_bot, sot_t, sot_y, clean_t, clean_y, 'tab:purple',
        'Mic (SOT, drift-corrected)', 'Demodulated fiber (v7, outliers rejected)',
        f"Fetal instantaneous HR — envelope-aware, outliers rejected "
        f"({stats['rejected']}/{stats['total']} = {stats['reject_pct']:.1f}% blanked)", band,
    )
    ax_bot.set_xlabel("Time (s)", fontsize=8)
    if sot_t.size:
        ax_bot.set_xlim(float(sot.mic.time[0]), float(sot.mic.time[-1]))
    fig.suptitle(
        f"Envelope-aware demodulated HR, fiber {result.fiber} — outlier rejection at "
        f"±{HR_OUTLIER_TOLERANCE:.0%} of a ±{HR_OUTLIER_HALF_WINDOW}-point median",
        fontsize=10,
    )
    plt.savefig(out / "hr_comparison_no_outlier.png", dpi=150)
    plt.close()
    return stats


def _write_hr_comparison(result: DemodulationResult, demod_output: fHROutput,
                         sot: SOTResult, out: Path, filename: str, mode_label: str,
                         optimized_patient: Optional[str] = None) -> None:
    """Outputs 2 and 3: bandpassed-fiber v7 HR on top, demodulated v7 HR below.

    The top panel was the maternal one and sat blank here, since this experiment has no
    chest channel. The bandpass control makes it answer the actual question: did
    demodulating help compared with not demodulating?
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    low, high = result.bandpass_hz

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=_figsize(2), sharex=True,
                                         constrained_layout=True)
    _plot_hr_axis(
        ax_top,
        sot_beats=sot.mic_beats,
        pred_beats=result.bandpass_only.fetal_beats,
        sot_color='tab:red', pred_color='tab:blue',
        sot_label='Mic (SOT, drift-corrected)',
        pred_label=f'Bandpassed fiber {result.fiber} (v7)',
        title=f"Fetal instantaneous HR — bandpassed {low:g}-{high:g} Hz fiber vs SOT",
        band=FETAL_BPM_RANGE,
    )
    _plot_hr_axis(
        ax_bot,
        sot_beats=sot.mic_beats,
        pred_beats=demod_output.fetal_beats,
        sot_color='tab:red', pred_color='tab:green',
        sot_label='Mic (SOT, drift-corrected)',
        pred_label=f'Demodulated fiber (v7, {mode_label})',
        title=f"Fetal instantaneous HR — demodulated fiber, {mode_label} v7, vs SOT",
        band=FETAL_BPM_RANGE,
    )

    ax_bot.set_xlabel("Time (s)", fontsize=8)
    ax_bot.set_xlim(float(sot.mic.time[0]), float(sot.mic.time[-1]))
    fig.suptitle(
        f"Instantaneous heart rate: bandpass control vs carrier demodulation "
        f"({mode_label} v7), fiber {result.fiber}",
        fontsize=11,
    )
    if optimized_patient is not None:
        non_aware = mode_label == "non-aware"
        agreement = result.non_aware_agreement if non_aware else result.agreement
        r, mae = agreement.display_values
        mode_suffix = " (non-aware v7)" if non_aware else ""
        fig.suptitle(
            f"{optimized_patient} — Optimized Demodulation{mode_suffix}\n"
            f"Fiber {result.fiber} | Band {low:g}–{high:g} Hz | R={r} | MAE={mae} BPM",
            fontsize=11,
        )
    plt.savefig(out / filename, dpi=150)
    plt.close()


def demodulation_plots(sot: SOTResult, drift_log_path, out: Path,
                       zoom_start_s: Optional[float] = ZOOM_START_S,
                       zoom_span_s: float = ZOOM_SPAN_S,
                       reject_hr_outliers: bool = False,
                       write_hr_comparisons: bool = True):
    """Pipeline stage: write the experiment's outputs. Pass-through.

    ``reject_hr_outliers`` adds a fifth file, ``hr_comparison_no_outlier.png``, and changes
    nothing else -- the four standard outputs, the other two traces and the NST branch are
    all produced exactly as they are with the flag off.

    ``write_hr_comparisons=False`` keeps the HTML diagnostics but suppresses the
    two baseline HR filenames for an optimized run.
    """

    def run_demodulation_plots(result: DemodulationResult) -> DemodulationResult:
        out_path = Path(out)
        out_path.mkdir(parents=True, exist_ok=True)

        # The reference branch, unchanged: v7 mic beats shifted by the NST drift log.
        corrected_sot = replace(sot, mic_beats=correct_drift(sot.mic_beats, drift_log_path))

        _write_demodulated_html(result, out_path / "demodulated_fiber.html")
        if write_hr_comparisons:
            _write_hr_comparison(result, result.envelope_aware, corrected_sot, out_path,
                                 "hr_comparison_demodulated.png", "envelope-aware")
            _write_hr_comparison(result, result.non_aware, corrected_sot, out_path,
                                 "hr_comparison_demodulated_non_aware.png", "non-aware")
        _write_bandpass_html(result, out_path / "bandpass_fiber.html",
                             zoom_span_s=zoom_span_s, zoom_start_s=zoom_start_s)

        if reject_hr_outliers:
            s = _write_no_outlier_comparison(result, corrected_sot, out_path)
            print()
            print(f"[demodulation] HR outlier rejection (envelope-aware only, "
                  f"+/-{HR_OUTLIER_TOLERANCE:.0%} of a +/-{HR_OUTLIER_HALF_WINDOW}-point "
                  f"neighbour median)")
            print(f"[demodulation]   instantaneous HR points : {s['total']}")
            print(f"[demodulation]   rejected                : {s['rejected']} "
                  f"({s['reject_pct']:.2f}%)")
            print(f"[demodulation]   valid after rejection   : {s['valid_pct']:.2f}% "
                  f"of points, {s['coverage_pct']:.2f}% of the smoothed trace")
            print(f"[demodulation]   before  r={s['r_before']:+.3f}  "
                  f"MAE={s['mae_before']:.2f} bpm")
            print(f"[demodulation]   after   r={s['r_after']:+.3f}  "
                  f"MAE={s['mae_after']:.2f} bpm   "
                  f"(scored on {s['grid_valid']}/{s['grid_total']} s where both are valid)")
        return result

    run_demodulation_plots.__name__ = "demodulation_plots"
    return run_demodulation_plots


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_demodulation_pipeline(
        patient: str,
        window: Tuple[float, float],
        datadir: str,
        fiber: str = DEMOD_FIBER,
        bandpass_hz: Tuple[float, float] = DEMOD_BANDPASS_HZ,
        baseband_hz: float = DEMOD_BASEBAND_HZ,
        zoom_start_s: Optional[float] = ZOOM_START_S,
        reject_hr_outliers: bool = False,
        optimized: bool = False,
) -> DemodulationResult:
    """End-to-end: fiber -> bandpass -> two-pass demodulation -> two v7 modes -> plots.

    ``zoom_start_s`` places the full-rate excerpt in ``bandpass_fiber.html`` (rows 3-4);
    the span is fixed at ``ZOOM_SPAN_S``, and ``None`` centres it on the median beat.

    ``reject_hr_outliers`` is off by default and purely additive -- it writes one extra
    figure from the envelope-aware trace and touches nothing else.

    ``agreement`` and ``non_aware_agreement`` on the returned result contain the
    two modes' HR comparison statistics (before optional outlier rejection).
    ``optimized=True`` writes ``hr_comparison_optimized.png`` (envelope-aware)
    and ``hr_comparison_optimized_nonaware.png`` (non-aware), and isolates all
    other artifacts under ``optimized/``. It never
    writes the baseline comparison files. The batch caller runs the fixed baseline
    separately on the same window.

    Writes only under ``.out/<patient>/demodulation/``. Detector diagnostics go to
    ``diagnostics/``, one subdirectory per mode, since v7 writes the same
    ``detect_v7_fetal.png`` filename every time.
    """
    name = _resolve_fiber(fiber)
    if name not in DEMOD_FIBERS:
        raise ValueError(f"fiber must be one of {', '.join(DEMOD_FIBERS)}; got {fiber!r}")
    low, high = bandpass_hz
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("acoustic band must satisfy 0 < low_hz < high_hz (finite Hz)")

    out_path = Path(f"{PROJECT_DIR}.out/{patient}/demodulation/")
    out_path.mkdir(parents=True, exist_ok=True)
    artifacts = out_path / "optimized" if optimized else out_path
    diagnostics = artifacts / "diagnostics"

    # Reference branch: NST mic -> v7 -> drift correction, untouched, but on its own cache
    # key so it cannot collide with the raw-bandpass or FUNet SOT caches.
    sot = Pipeline([
        load_sot(),
        windowed(window[0], window[1]),
        sot_beats(v7_beat_detector, diagnostics / "sot"),
    ], f"{artifacts}/cache_demodulation_sot/", play_sound=False).process(datadir)

    result = Pipeline([
        load_data,
        windowed(window[0], window[1]),
        demodulation_demodulate(fiber, bandpass_hz, baseband_hz),
        demodulation_envelope_aware_beats(diagnostics / "envelope_aware"),
        demodulation_non_aware_beats(diagnostics / "non_aware"),
        demodulation_bandpass_v7_beats(diagnostics / "bandpass_only"),
        demodulation_plots(sot, f"{Path(datadir) / NST_DRIFT_LOG_FILE}", artifacts,
                           zoom_start_s=zoom_start_s,
                           reject_hr_outliers=reject_hr_outliers,
                           write_hr_comparisons=not optimized),
    ], f"{artifacts}/cache_demodulation/", play_sound=False).process(datadir)

    corrected_sot = correct_drift(sot.mic_beats, f"{Path(datadir) / NST_DRIFT_LOG_FILE}")
    st, sy = _current_hr_func()(np.asarray(corrected_sot, float), FETAL_BPM_RANGE)

    def agreement_for(output):
        pt, py = _current_hr_func()(output.fetal_beats, FETAL_BPM_RANGE)
        values = _agreement_values(st, sy, pt, py)
        return DemodulationAgreement(
            *(values if values is not None else (float("nan"), float("nan")))
        )

    result = replace(result, agreement=agreement_for(result.envelope_aware),
                     non_aware_agreement=agreement_for(result.non_aware))
    if optimized:
        _write_hr_comparison(
            result, result.envelope_aware, replace(sot, mic_beats=corrected_sot),
            out_path, "hr_comparison_optimized.png", "envelope-aware",
            optimized_patient=patient,
        )
        _write_hr_comparison(
            result, result.non_aware, replace(sot, mic_beats=corrected_sot),
            out_path, "hr_comparison_optimized_nonaware.png", "non-aware",
            optimized_patient=patient,
        )
    rows = (
        ("NST reference (v7, drift-corrected)", corrected_sot, ""),
        ("bandpass control (v7)", result.bandpass_only.fetal_beats,
         _agreement(corrected_sot, result.bandpass_only.fetal_beats)),
        ("demodulated, envelope-aware v7", result.envelope_aware.fetal_beats,
         _agreement(corrected_sot, result.envelope_aware.fetal_beats)),
        ("demodulated, non-aware v7", result.non_aware.fetal_beats,
         _agreement(corrected_sot, result.non_aware.fetal_beats)),
    )

    print()
    print("=" * 72)
    print(f"  DEMODULATION -- {patient}  {window[0]:g}-{window[1]:g}s  fiber {result.fiber}")
    print(f"  bandpass {result.bandpass_hz[0]:g}-{result.bandpass_hz[1]:g} Hz   "
          f"baseband LPF {result.baseband_hz:g} Hz   "
          f"zoom {'auto' if zoom_start_s is None else f'{zoom_start_s:g}s'}"
          f"+{ZOOM_SPAN_S:g}s")
    print(f"  carrier  {result.initial_carrier_hz:.4f} Hz -> {result.corrected_carrier_hz:.4f} Hz"
          f"   (residual {result.residual_offset_hz:+.4f} Hz, "
          f"final {result.final_residual_hz:+.2e} Hz)")
    print("-" * 72)
    print(f"  {'trace':<36}{'beats':>7}   vs NST")
    for label, beats, stat in rows:
        print(f"  {label:<36}{len(beats):>7}   {stat}")
    print("-" * 72)
    written = ["demodulated_fiber.html", "bandpass_fiber.html"] if optimized else [
        "demodulated_fiber.html", "hr_comparison_demodulated.png",
        "hr_comparison_demodulated_non_aware.png", "bandpass_fiber.html"]
    if reject_hr_outliers:
        written.append("hr_comparison_no_outlier.png")
    if optimized:
        written = [f"optimized/{name}" for name in written]
        written.append("hr_comparison_optimized.png")
        written.append("hr_comparison_optimized_nonaware.png")
    for name in written:
        print(f"  {name}")
    print(f"  in {out_path.resolve()}")
    print("=" * 72)
    return result
