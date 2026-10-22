"""Batch isolation, legacy HR statistics, and the additive optimized figure."""

import csv
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import analyze.demodulation as demod
from analyze.data import Audio, FiberData
from analyze.hr import fHROutput
from analyze.plot_hr import _agreement_stats, _agreement_values, _current_hr_func
from analyze.sot import SOTResult
from bin import run_demodulation_batch as batch
from constants import FETAL_BPM_RANGE


def test_direct_cli_execution():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "src/bin/run_demodulation_batch.py", "--help"],
        cwd=root, env={**os.environ, "PYTHONPATH": "src", "MPLBACKEND": "Agg"},
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--case" in completed.stdout


def test_agreement_retains_one_hz_zero_lag_interpolation():
    times = np.array([0., 2., 4.])
    reference = np.array([120., 140., 160.])
    prediction = np.array([125., 135., 165.])
    # The original scoring grid is [0, 1, 2, 3], excluding the upper endpoint.
    r, mae = _agreement_values(times, reference, times, prediction)
    assert r == pytest.approx(np.corrcoef([120, 130, 140, 150], [125, 130, 135, 150])[0, 1])
    assert mae == 2.5
    assert _agreement_stats(times, reference, times, prediction) == "  |  r=0.96, MAE=2.5 bpm"


@pytest.mark.parametrize("case", [
    "Patient 6,140,150,3A", "Patient 6,150,140,2A", "Patient 6,140,140,2A",
    "Patient 6,nan,150,2A", "Patient 6,140,inf,2A", "Patient 6,low,150,2A",
    "Patient 6,140,150", "../Patient 6,140,150,2A",
])
def test_invalid_case_is_rejected(case):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        batch.parse_case(case)


@pytest.mark.parametrize("kwargs", [
    {"fiber": "3A"}, {"bandpass_hz": (150, 140)},
])
def test_pipeline_validates_before_loading(kwargs):
    with pytest.raises(ValueError):
        demod.run_demodulation_pipeline("unused", (0, 120), "/missing", **kwargs)


@pytest.mark.parametrize("failure_stage", ["baseline", "optimized"])
def test_batch_continues_and_writes_matching_summary(tmp_path, monkeypatch, capsys, failure_stage):
    monkeypatch.setattr(batch, "PROJECT_DIR", str(tmp_path))
    calls = []

    def run(patient, window, datadir, **kwargs):
        calls.append((patient, window, datadir, kwargs))
        if patient == "missing" and (failure_stage == "baseline" or kwargs.get("optimized")):
            raise FileNotFoundError("no data")
        return SimpleNamespace(
            agreement=demod.DemodulationAgreement(0.87654, 3.654),
            non_aware_agreement=demod.DemodulationAgreement(0.4567, 8.123),
        )

    monkeypatch.setattr(batch, "run_demodulation_pipeline", run)
    status = batch.main([
        "--dataset-dir", str(tmp_path / "dataset"), "--window", "0", "120",
        "--case", "Patient 6,140,150,2A", "--case", "missing,195,205,2B",
        "--case", "Patient 7,195,205,2B",
    ])
    assert status == 1
    assert len(calls) == (5 if failure_stage == "baseline" else 6)
    assert calls[0] == (
        "Patient 6", (0, 120), str(tmp_path / "dataset" / "Patient 6"),
        dict(fiber="1B", bandpass_hz=(200, 230)),
    )
    assert calls[1] == (
        "Patient 6", (0, 120), str(tmp_path / "dataset" / "Patient 6"),
        dict(fiber="2A", bandpass_hz=(140, 150), optimized=True),
    )
    with (tmp_path / ".out/demodulation_adaptive_summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["success", "error", "success"]
    assert rows[0]["optimized_R"] == "0.88" and rows[0]["optimized_MAE"] == "3.7"
    assert rows[0]["baseline_non_aware_R"] == "0.46"
    assert rows[0]["baseline_non_aware_MAE"] == "8.1"
    assert rows[1]["optimized_R"] == rows[1]["optimized_MAE"] == ""
    assert rows[1]["baseline_envelope_aware_R"] == ("" if failure_stage == "baseline" else "0.88")
    assert "FileNotFoundError" in rows[1]["error"]
    assert "Patient 7" in capsys.readouterr().out


@pytest.mark.parametrize("patient,low,high,fiber", [
    ("PT12_2", 210., 220., "1B"), ("Patient 6", 140., 150., "2A"),
])
def test_baseline_and_optimized_outputs_are_separate(
        tmp_path, monkeypatch, patient, low, high, fiber):
    """Run the real DSP, plotting stages and CLI; substitute data IO and beat detection."""
    monkeypatch.setattr(demod, "PROJECT_DIR", str(tmp_path) + "/")
    monkeypatch.setattr(batch, "PROJECT_DIR", str(tmp_path))
    ref = np.cumsum(60 / (140 + 10 * np.sin(np.arange(60) / 5)))
    t = np.arange(30_000) / 1000.
    audio = Audio(t, 1000, np.sin(2 * np.pi * 215 * t) + np.sin(2 * np.pi * 145 * t))
    sot = SOTResult(None, None, audio, ref)
    monkeypatch.setattr(demod, "load_data", lambda path: FiberData(None, {"1B": audio, "2A": audio}))
    monkeypatch.setattr(demod, "load_sot", lambda: lambda path: sot)

    def reference_beats(detector, out):
        def detect(data):
            out.mkdir(parents=True, exist_ok=True)
            (out / "reference.txt").write_text("reference diagnostic")
            return data
        return detect

    def fiber_beats(detector, out, **kwargs):
        def detect(pair):
            shift = (3 if "optimized" in out.parts else 0) + (7 if out.name == "non_aware" else 0)
            beats = np.cumsum(60 / (140 + shift + 8 * np.sin(np.arange(60) / 6)))
            out.mkdir(parents=True, exist_ok=True)
            (out / "detector.txt").write_text(str(shift))
            return fHROutput(pair.abdomen, beats, None, None)
        return detect

    monkeypatch.setattr(demod, "sot_beats", reference_beats)
    monkeypatch.setattr(demod, "fiber_beats", fiber_beats)
    monkeypatch.setattr(demod, "correct_drift", lambda beats, path: beats + 0.75)
    figures = {}
    savefig = demod.plt.savefig

    def save(path, **kwargs):
        fig = demod.plt.gcf()
        figures[Path(path).name] = (fig._suptitle.get_text(), fig.axes[-1].get_title(),
                                   fig.axes[-1].get_xlim())
        savefig(path, **kwargs)

    monkeypatch.setattr(demod.plt, "savefig", save)
    out = tmp_path / f".out/{patient}/demodulation"
    results = []

    def run(*args, **kwargs):
        before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()} if results else {}
        result = demod.run_demodulation_pipeline(*args, **kwargs)
        if results:
            for path, content in before.items():
                assert path.read_bytes() == content, f"optimized run changed {path}"
        else:
            assert not (out / "hr_comparison_optimized.png").exists()
            assert not (out / "hr_comparison_optimized_nonaware.png").exists()
        results.append(result)
        return result

    monkeypatch.setattr(batch, "run_demodulation_pipeline", run)
    assert batch.main([
        "--dataset-dir", str(tmp_path), "--window", "2", "28",
        "--case", f"{patient},{low},{high},{fiber}",
    ]) == 0
    baseline, optimized = results
    assert (baseline.fiber, baseline.bandpass_hz) == ("1B", (200., 230.))
    assert (optimized.fiber, optimized.bandpass_hz) == (fiber, (low, high))
    assert np.array_equal(baseline.amplitude.time, optimized.amplitude.time)
    assert baseline.amplitude.time[[0, -1]].tolist() == [2., 28.]
    with (tmp_path / ".out/demodulation_adaptive_summary.csv").open() as stream:
        row, = csv.DictReader(stream)
    assert list(row) == [
        "patient", "optimized_fiber", "optimized_band_low_hz", "optimized_band_high_hz",
        "baseline_envelope_aware_R", "baseline_envelope_aware_MAE",
        "baseline_non_aware_R", "baseline_non_aware_MAE", "optimized_R", "optimized_MAE",
        "status", "error",
    ]
    for prefix, result, mode, filename in (
        ("baseline_envelope_aware", baseline, "envelope_aware", "hr_comparison_demodulated.png"),
        ("baseline_non_aware", baseline, "non_aware", "hr_comparison_demodulated_non_aware.png"),
        ("optimized", optimized, "envelope_aware", "hr_comparison_optimized.png"),
    ):
        hr = _current_hr_func()
        traces = (*hr(sot.window(2, 28).mic_beats + 0.75, FETAL_BPM_RANGE),
                  *hr(getattr(result, mode).fetal_beats, FETAL_BPM_RANGE))
        r, mae = _agreement_values(*traces)
        assert row[f"{prefix}_R"] == f"{r:.2f}"
        assert row[f"{prefix}_MAE"] == f"{mae:.1f}"
        assert figures[filename][1].endswith(_agreement_stats(*traces))
        assert figures[filename][2] == (2., 28.)
    assert figures["hr_comparison_optimized.png"][0] == (
        f"{patient} — Optimized Demodulation\n"
        f"Fiber {fiber} | Band {low:g}–{high:g} Hz | R={row['optimized_R']} | MAE={row['optimized_MAE']} BPM"
    )
    hr = _current_hr_func()
    traces = (*hr(sot.window(2, 28).mic_beats + 0.75, FETAL_BPM_RANGE),
              *hr(optimized.non_aware.fetal_beats, FETAL_BPM_RANGE))
    r, mae = _agreement_values(*traces)
    assert optimized.non_aware_agreement == demod.DemodulationAgreement(r, mae)
    title, axis_title, limits = figures["hr_comparison_optimized_nonaware.png"]
    assert title == (
        f"{patient} — Optimized Demodulation (non-aware v7)\n"
        f"Fiber {fiber} | Band {low:g}–{high:g} Hz | R={r:.2f} | MAE={mae:.1f} BPM"
    )
    assert "non-aware v7" in axis_title
    assert axis_title.endswith(_agreement_stats(*traces))
    assert limits == (2., 28.)
    assert optimized.non_aware_agreement != optimized.agreement
    assert len(figures) == 4
    assert (out / "hr_comparison_optimized.png").read_bytes().startswith(b"\x89PNG")
    assert (out / "hr_comparison_optimized_nonaware.png").read_bytes().startswith(b"\x89PNG")
    assert not list((out / "optimized").glob("hr_comparison*.png"))
    for artifacts in (out, out / "optimized"):
        assert (artifacts / "demodulated_fiber.html").exists()
        assert (artifacts / "bandpass_fiber.html").exists()
        assert (artifacts / "cache_demodulation").is_dir()
        for mode in ("sot", "envelope_aware", "non_aware", "bandpass_only"):
            assert list((artifacts / "diagnostics" / mode).glob("*.txt"))


@pytest.mark.parametrize("times,values", [
    (np.array([]), np.array([])),
    (np.arange(5.), np.ones(5)),
])
def test_undefined_agreement_preserves_legacy_empty_title(times, values):
    assert _agreement_values(times, values, times, values) is None
    assert _agreement_stats(times, values, times, values) == ""
