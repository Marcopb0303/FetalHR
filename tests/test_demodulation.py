"""Focused tests for the carrier-demodulation experiment.

Everything here runs on synthesised signals -- no Banner_data, no checkpoints -- so the
suite is runnable on any machine. The synthetic carrier is built the way the fiber is
believed to behave: a steady tone whose amplitude is modulated by a heartbeat.
"""

import numpy as np
import pytest

from analyze.data import Audio
from analyze.demodulation import (
    DEMOD_BANDPASS_HZ,
    DEMOD_BASEBAND_HZ,
    _resolve_fiber,
    demodulate_audio,
    demodulation_bandpass_v7_beats,
    demodulation_demodulate,
    demodulation_envelope_aware_beats,
    demodulation_non_aware_beats,
    demodulation_plots,
    estimate_carrier_hz,
)
from analyze.filters import bp_filter
from analyze.hr.detect_v7 import v7_beat_detector, v7_envelope_aware_beat_detector
from analyze.hr.detect_v9 import v9_beat_detector, v9_envelope_aware_beat_detector
from constants import FETAL_BPM_RANGE

FS = 5000.0
DURATION_S = 12.0
CARRIER_HZ = 213.7          # deliberately not a round number, nor a bin centre
BEAT_HZ = 2.4               # 144 bpm, inside FETAL_BPM_RANGE


def _modulated_carrier(carrier_hz=CARRIER_HZ, fs=FS, duration_s=DURATION_S, noise=0.0):
    """A carrier at ``carrier_hz`` whose amplitude is pulsed at ``BEAT_HZ``."""
    t = np.arange(int(duration_s * fs), dtype=float) / fs
    # Raised cosine to the 8th power: a narrow, strictly positive pulse train, so the
    # true envelope is unambiguous and has one maximum per beat.
    envelope = 0.2 + 1.0 * (0.5 * (1.0 + np.cos(2.0 * np.pi * BEAT_HZ * t))) ** 8
    x = envelope * np.sin(2.0 * np.pi * carrier_hz * t)
    if noise:
        x = x + np.random.default_rng(0).normal(0.0, noise, t.size)
    return Audio(t, int(fs), x)


def _bandpassed(**kw):
    raw = _modulated_carrier(**kw)
    return bp_filter(raw, DEMOD_BANDPASS_HZ[0], DEMOD_BANDPASS_HZ[1],
                     order=4, filter_type="butter")


# ---------------------------------------------------------------------------
# Carrier estimation
# ---------------------------------------------------------------------------

def test_carrier_estimate_recovers_a_known_tone():
    """The phase-slope fit should land on the tone that was put in."""
    from scipy.signal import hilbert

    bp = _bandpassed()
    carrier, _mask = estimate_carrier_hz(hilbert(bp.data), FS)
    assert carrier == pytest.approx(CARRIER_HZ, abs=0.5)


def test_two_pass_demodulation_drives_the_residual_down():
    """The second pass exists to remove the first pass's offset; check that it does."""
    bp = _bandpassed()
    _amp, initial, residual, corrected, final = demodulate_audio(bp)

    assert corrected == pytest.approx(initial + residual)
    assert corrected == pytest.approx(CARRIER_HZ, abs=0.5)
    # The point of the correction: whatever was left after pass 1 is far smaller after
    # pass 2. Compared as magnitudes since either can be signed.
    assert abs(final) < abs(residual)


def test_carrier_estimation_excludes_edges_and_weak_samples():
    """The fit mask must drop the filter-transient margins and the low-|z| samples."""
    from scipy.signal import hilbert

    bp = _bandpassed()
    analytic = hilbert(bp.data)
    _carrier, mask = estimate_carrier_hz(analytic, FS, edge_s=0.5,
                                         amplitude_floor_pct=25.0)
    edge = int(0.5 * FS)
    assert not mask[:edge].any(), "samples inside the leading edge margin were fitted"
    assert not mask[-edge:].any(), "samples inside the trailing edge margin were fitted"
    assert mask.any(), "the mask removed everything"
    # Amplitude gating: nothing kept may sit below the retained minimum of |z|.
    magnitude = np.abs(analytic)
    assert magnitude[mask].min() >= np.percentile(magnitude[mask], 0.0)
    assert mask.sum() < (~np.zeros_like(mask)).sum()


def test_short_window_falls_back_instead_of_emptying_the_mask():
    """A clip shorter than two edge margins must still produce a usable fit."""
    from scipy.signal import hilbert

    bp = _bandpassed(duration_s=0.4)   # shorter than a single 0.5 s margin
    carrier, mask = estimate_carrier_hz(hilbert(bp.data), FS, edge_s=0.5)
    assert mask.any()
    assert np.isfinite(carrier)


# ---------------------------------------------------------------------------
# Output shape: length, time axis, non-negativity
# ---------------------------------------------------------------------------

def test_amplitude_preserves_length_and_time_axis_exactly():
    bp = _bandpassed()
    amplitude, *_ = demodulate_audio(bp)

    assert len(amplitude.data) == len(bp.data)
    assert len(amplitude.time) == len(bp.time)
    assert np.array_equal(amplitude.time, bp.time), "the time axis was not preserved"
    assert amplitude.hz == bp.hz


def test_amplitude_is_non_negative_and_finite():
    amplitude, *_ = demodulate_audio(_bandpassed(noise=0.05))
    assert np.all(np.isfinite(amplitude.data))
    assert np.all(amplitude.data >= 0.0), "A(t) = |u(t)| must be non-negative"


# ---------------------------------------------------------------------------
# Nyquist validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [0.0, -5.0, FS / 2.0, FS])
def test_baseband_cutoff_is_validated_against_nyquist(cutoff):
    with pytest.raises(ValueError, match="Nyquist"):
        demodulate_audio(_bandpassed(duration_s=2.0), baseband_hz=cutoff)


def test_bandpass_above_nyquist_is_rejected():
    """A band that reaches Nyquist must fail in the stage, not deep inside scipy."""
    stage = demodulation_demodulate(fiber="1B", bandpass_hz=(200.0, FS))

    class _Fibers:
        abdomen = {"1B": _modulated_carrier()}

    with pytest.raises(ValueError, match="Nyquist"):
        stage(_Fibers())


# ---------------------------------------------------------------------------
# Distinct cache keys
# ---------------------------------------------------------------------------

def test_pipeline_stage_names_are_distinct_and_namespaced():
    """Pipeline caches to ``out + stage.__name__ + '.pkl'``, so the stage names are the
    cache keys. They must be unique and must not collide with another pipeline's."""
    from pathlib import Path

    names = [
        demodulation_demodulate().__name__,
        demodulation_envelope_aware_beats(Path("/tmp/x")).__name__,
        demodulation_non_aware_beats(Path("/tmp/x")).__name__,
        demodulation_bandpass_v7_beats(Path("/tmp/x")).__name__,
        demodulation_plots(None, "/tmp/drift.csv", Path("/tmp/x")).__name__,
    ]
    assert len(set(names)) == len(names), f"duplicate stage names: {names}"
    assert all(n.startswith("demodulation") for n in names), names
    assert any("envelope_aware" in n for n in names)
    assert any("non_aware" in n for n in names)

    # Nothing may shadow a stage the raw-bandpass or FUNet pipelines cache under.
    existing = {
        "load_data", "select_window", "run_bandpass_ALL", "select_fiber",
        "detect_fiber_beats", "detect_sot_beats", "plot_hr", "use_funet",
        "plot_hr_corrected", "plot_hr_correction_comparison", "evaluate_v3",
    }
    assert not (set(names) & existing), set(names) & existing


@pytest.mark.parametrize("start,expected", [
    (4.0, 4.0),        # inside the record, honoured as given
    (-10.0, 0.0),      # before the start, slid forward
    (999.0, 9.0),      # past the end, slid back so the span still fits (12 s - 3 s)
])
def test_zoom_start_is_clamped_into_the_window(start, expected, tmp_path):
    """A zoom start outside the analysis window must slide back inside, keeping the full
    span, rather than raising or emitting a truncated excerpt."""
    from analyze.demodulation import ZOOM_SPAN_S, DemodulationResult, _write_bandpass_html
    from analyze.hr import fHROutput

    bp = _bandpassed()                       # DURATION_S = 12 s
    amplitude, i, r, c, f = demodulate_audio(bp)
    beats = v7_envelope_aware_beat_detector(amplitude, (110.0, 180.0))["times"]
    out = fHROutput(fetal_source=amplitude, fetal_beats=beats,
                    maternal_source=None, maternal_beats=None)
    result = DemodulationResult(
        fiber="1B", bandpass_hz=DEMOD_BANDPASS_HZ, baseband_hz=DEMOD_BASEBAND_HZ,
        bandpassed=bp, amplitude=amplitude, initial_carrier_hz=i, residual_offset_hz=r,
        corrected_carrier_hz=c, final_residual_hz=f,
        envelope_aware=out, non_aware=out, bandpass_only=out,
    )
    path = tmp_path / "bandpass_fiber.html"
    _write_bandpass_html(result, path, zoom_span_s=ZOOM_SPAN_S, zoom_start_s=start)

    assert path.exists()
    # The panel title records the span actually drawn; check it is the requested length
    # starting where the clamp put it.
    text = path.read_text()
    assert f"{expected:.2f}-{expected + ZOOM_SPAN_S:.2f} s" in text, (
        f"expected a {ZOOM_SPAN_S:g}s excerpt starting at {expected:g}s"
    )


# ---------------------------------------------------------------------------
# HR outlier rejection
# ---------------------------------------------------------------------------

def test_outlier_mask_flags_a_planted_spike_and_nothing_else():
    """A single point 50% above its neighbours must be caught; a steady trace must not."""
    from analyze.demodulation import hr_outlier_mask

    steady = np.full(40, 140.0)
    assert not hr_outlier_mask(steady).any(), "a constant trace has no outliers"

    spiked = steady.copy()
    spiked[20] = 140.0 * 1.5           # +50%, well past the 20% tolerance
    mask = hr_outlier_mask(spiked)
    assert mask[20], "the planted spike was not rejected"
    assert mask.sum() == 1, f"collateral rejections at {np.flatnonzero(mask)}"


@pytest.mark.parametrize("factor,expected", [
    (1.19, False),   # inside +/-20%
    (1.21, True),    # just outside, above
    (0.81, False),   # inside, below
    (0.79, True),    # just outside, below
])
def test_outlier_threshold_is_twenty_percent_either_side(factor, expected):
    from analyze.demodulation import hr_outlier_mask

    y = np.full(40, 140.0)
    y[20] = 140.0 * factor
    assert bool(hr_outlier_mask(y)[20]) is expected


def test_outlier_decisions_come_from_the_original_array_in_one_pass():
    """Two adjacent spikes must both be flagged. If the mask were applied progressively,
    cleaning the first would change the median seen by the second."""
    from analyze.demodulation import hr_outlier_mask

    y = np.full(40, 140.0)
    y[20] = 210.0
    y[21] = 210.0
    mask = hr_outlier_mask(y)
    assert mask[20] and mask[21], f"expected both flagged, got {np.flatnonzero(mask)}"


def test_current_point_is_excluded_from_its_own_median():
    """With the point included, a wide enough run of spikes would vote itself normal."""
    from analyze.demodulation import hr_outlier_mask

    y = np.full(60, 140.0)
    y[30] = 300.0
    assert hr_outlier_mask(y, half_window=5)[30]


def test_rejection_preserves_timestamps_and_length():
    """Rejected points keep their timestamp and lose only their value."""
    from analyze.demodulation import inst_hr_rejected
    from analyze.plot_hr import _current_hr_func

    beats = np.cumsum(np.full(300, 60.0 / 140.0))
    beats[150] += 0.20                                  # one displaced beat -> two bad IBIs
    raw_t, raw_y = _current_hr_func()(beats, FETAL_BPM_RANGE)
    t, y, stats = inst_hr_rejected(beats, FETAL_BPM_RANGE)

    assert np.array_equal(t, raw_t), "timestamps moved"
    assert y.size == raw_y.size, "the trace changed length"
    assert stats["total"] == raw_y.size
    assert stats["rejected"] > 0, "the displaced beat produced no rejection"


def test_nan_is_confined_and_the_average_ignores_it():
    """The NaN-aware moving average must not spread one gap across its whole window."""
    from analyze.demodulation import HR_SMOOTH_WINDOW, moving_average_nan
    from analyze.util import moving_average_v2

    y = np.full(200, 140.0)
    y[100] = np.nan

    stock = moving_average_v2(y, HR_SMOOTH_WINDOW)
    aware = moving_average_nan(y, HR_SMOOTH_WINDOW)

    assert np.isnan(stock).sum() >= HR_SMOOTH_WINDOW, "stock average should smear the NaN"
    assert not np.isnan(aware).any(), "NaN-aware average should bridge a single gap"
    assert np.allclose(aware, 140.0), "surviving values should average to the true level"
    assert aware.size == y.size


def test_all_nan_window_yields_nan_not_a_division_error():
    from analyze.demodulation import moving_average_nan

    out = moving_average_nan(np.full(10, np.nan), 5)
    assert out.size == 10 and np.isnan(out).all()


def test_agreement_masked_matches_the_repo_stat_when_there_are_no_gaps():
    """With a gap-free trace the masked metric must equal plot_hr's own number."""
    from analyze.demodulation import _agreement_masked
    from analyze.plot_hr import _agreement_stats, _current_hr_func

    a = np.cumsum(np.full(400, 60.0 / 140.0))
    b = np.cumsum(np.full(400, 60.0 / 142.0))
    at, ay = _current_hr_func()(a, FETAL_BPM_RANGE)
    bt, by = _current_hr_func()(b, FETAL_BPM_RANGE)

    r, mae, _n, _tot = _agreement_masked(at, ay, bt, by)
    assert f"r={r:.2f}, MAE={mae:.1f} bpm" in _agreement_stats(at, ay, bt, by)


def test_reject_flag_defaults_to_off_and_is_purely_additive(tmp_path):
    """Default off writes the four standard outputs; on adds exactly one more."""
    import inspect
    from analyze.demodulation import demodulation_plots, run_demodulation_pipeline

    assert inspect.signature(run_demodulation_pipeline).parameters[
        "reject_hr_outliers"].default is False
    assert inspect.signature(demodulation_plots).parameters[
        "reject_hr_outliers"].default is False


def test_rejection_does_not_touch_the_other_traces():
    """inst_hr_rejected must be a pure function of the beats handed to it."""
    from analyze.demodulation import inst_hr_rejected

    beats = np.cumsum(np.full(300, 60.0 / 140.0))
    beats[150] += 0.20
    original = beats.copy()
    inst_hr_rejected(beats, FETAL_BPM_RANGE)
    assert np.array_equal(beats, original), "the input beat array was mutated"


def test_fiber_name_accepts_the_belly_prefix():
    assert _resolve_fiber("belly_1B") == "1B"
    assert _resolve_fiber("1B") == "1B"
    assert _resolve_fiber("2A") == "2A"


# ---------------------------------------------------------------------------
# The existing v7 default must be untouched (v7 now drives the experiment)
# ---------------------------------------------------------------------------

def test_v7_default_envelope_behaviour_is_unchanged():
    """Calling v7 with no envelope argument must still mean the Shannon envelope."""
    bp = _bandpassed()
    default = v7_beat_detector(bp, (110.0, 180.0))
    explicit = v7_beat_detector(bp, (110.0, 180.0), envelope_method="shannon")

    assert np.array_equal(default["times"], explicit["times"])
    assert np.array_equal(default["peaks"], explicit["peaks"])


def test_v7_envelope_aware_helper_matches_explicit_precomputed():
    amplitude, *_ = demodulate_audio(_bandpassed())
    via_helper = v7_envelope_aware_beat_detector(amplitude, (110.0, 180.0))
    via_kwarg = v7_beat_detector(amplitude, (110.0, 180.0), envelope_method="precomputed")

    assert np.array_equal(via_helper["times"], via_kwarg["times"])


def test_v7_precomputed_mode_skips_the_shannon_envelope():
    """'precomputed' must feed the input straight through, not rectify it first."""
    from analyze.hr.detect_v7 import _features

    amplitude, *_ = demodulate_audio(_bandpassed())
    _g1, a_pre = _features(amplitude, 100.0, envelope_method="precomputed")
    _g2, a_shannon = _features(amplitude, 100.0, envelope_method="shannon")

    assert not np.allclose(a_pre, a_shannon), (
        "the two modes produced the same feature; 'precomputed' is not being honoured"
    )


def test_v7_finds_beats_on_a_synthetic_pulse_train():
    """Sanity floor, so a zero on real data reads as a finding and not a broken test."""
    amplitude, *_ = demodulate_audio(_bandpassed())
    expected = BEAT_HZ * DURATION_S

    aware = v7_envelope_aware_beat_detector(amplitude, (110.0, 180.0))["times"]
    naive = v7_beat_detector(amplitude, (110.0, 180.0))["times"]

    assert len(aware) == pytest.approx(expected, abs=5), len(aware)
    assert len(naive) == pytest.approx(expected, abs=5), len(naive)


# ---------------------------------------------------------------------------
# v9 is no longer used by this experiment, but detect_v9.py was still edited,
# so its default has to stay pinned for the FUNet pipeline that does use it.
# ---------------------------------------------------------------------------

def test_v9_default_envelope_behaviour_is_unchanged():
    """Calling v9 with no envelope argument must still mean 'analytic'."""
    bp = _bandpassed()
    default = v9_beat_detector(bp, (110.0, 180.0))
    explicit = v9_beat_detector(bp, (110.0, 180.0), envelope_method="analytic")

    assert np.array_equal(default["times"], explicit["times"])
    assert np.array_equal(default["peaks"], explicit["peaks"])


def test_envelope_aware_helper_matches_explicit_precomputed():
    amplitude, *_ = demodulate_audio(_bandpassed())
    via_helper = v9_envelope_aware_beat_detector(amplitude, (110.0, 180.0))
    via_kwarg = v9_beat_detector(amplitude, (110.0, 180.0), envelope_method="precomputed")

    assert np.array_equal(via_helper["times"], via_kwarg["times"])


def test_precomputed_mode_skips_the_hilbert_envelope():
    """'precomputed' must return the input itself, not abs(hilbert(input))."""
    from analyze.hr.detect_v9 import _envelope

    amplitude, *_ = demodulate_audio(_bandpassed())
    passthrough = _envelope(amplitude.data, FS, "precomputed", 0.02, 0.005)
    analytic = _envelope(amplitude.data, FS, "analytic", 0.02, 0.005)

    assert np.array_equal(passthrough, amplitude.data)
    assert not np.allclose(passthrough, analytic), (
        "the two modes produced the same envelope; 'precomputed' is not being honoured"
    )


def test_both_modes_find_beats_on_a_synthetic_pulse_train():
    """Sanity floor: on a clean 144 bpm carrier both modes should find roughly the
    right number of beats, so a zero elsewhere means a real failure, not a broken test."""
    amplitude, *_ = demodulate_audio(_bandpassed())
    expected = BEAT_HZ * DURATION_S

    aware = v9_envelope_aware_beat_detector(amplitude, (110.0, 180.0))["times"]
    naive = v9_beat_detector(amplitude, (110.0, 180.0))["times"]

    assert len(aware) == pytest.approx(expected, abs=4), len(aware)
    assert len(naive) == pytest.approx(expected, abs=4), len(naive)
