# fhr-analysis — Repo Guide

Verified against this local directory on October 5, 2026. This guide describes
`summer2026/fhr-analysis`, not the separate `fhr-yongyi/fhr-analysis` checkout.
Defaults below are a snapshot of executable code; changing the code/configuration
changes the workflow. No performance benchmark is implied by a selected default.

This document is a map of the repository: what each folder is for, what the
scripts inside it do (at a glance, not line-by-line), and the different
end-to-end workflows ("pipelines") you can run. Read this alongside
[README.md](README.md), which has the quick-start setup commands.

## What this project does

The project extracts **fetal heart rate (FHR)** from acoustic sensor
recordings taken on a pregnant patient's abdomen ("Banner" fiber-optic
sensors), and scores how accurate the extraction is against a "source of
truth" (SOT) — a microphone or PPG (pulse-oximeter) recording taken at the
same time as a separate reference.

The hard part is that the abdomen recording is a *mixture*: fetal heart
sounds, maternal heart sounds, lung/breathing, and noise are all picked up
together. So most of the code is about **source separation** (pulling the
fetal heartbeat out of that mixture) and then **beat detection** (finding
individual heartbeat timestamps in the separated signal), followed by
**evaluation** (comparing detected beats/heart-rate-over-time against the
SOT).

Several separation methods are implemented, and each is a different
"pipeline" you can choose to run for a given recording:

| Method | What it is |
| --- | --- |
| **ICA** | Independent Component Analysis — classic blind source separation |
| **MNMF** | Multichannel Non-negative Matrix Factorization |
| **MLCMED** | Multi-lag covariance matrix-based eigenvalue decomposition |
| **NMCF** | Non-negative Matrix Co-Factorization |
| **NeoSSNet** | A pretrained deep-learning source-separation model (submodule), optionally fine-tuned on this project's own recordings |
| **FUNet** | A custom-trained deep-learning model that predicts *beat activity* directly (skips separating a clean waveform) |
| **Carrier demodulation** | Bandpass a fiber, estimate/downconvert its carrier, and compare beat detection on its amplitude envelope and acoustic control |
| **Raw bandpass** | No separation at all — just bandpass-filter the raw abdomen fiber into the fetal acoustic band |

The current entry point runs carrier demodulation, FUNet, NeoSSNet, and raw
bandpass. ICA/MNMF/MLCMED/NMCF remain available as optional baselines. Which
method performs best must be established on matched recordings and settings.

---

## Top-level layout

```
fhr-analysis/
├── setup.sh              Bootstraps the Python venv + submodules (see README)
├── requirements.txt       Python deps for the analysis code (src/) and the models (lib/)
├── bin/                   Standalone data-collection apps (own venvs, PyQt GUIs)
├── lib/                   The separation/detection models: submodule + training code + checkpoints
├── src/                   The main analysis library, CLI tools, and the beat-marking web app
└── tests/test_demodulation.py  Tests for the carrier-demodulation experiment
```

Two directories referenced throughout are **gitignored** (not in version
control, must exist locally or be fetched separately):
- `Banner_data/` — the actual patient recordings (fiber `.npy` bundles,
  microphone `.wav`, hand-marked beat `.npy` files). Nothing runs without this.
- `.out/` — cache and output files (plots, evaluation results) written by
  pipeline runs, usually under patient/method directories. Not every runner
  includes the window or configuration in its output path; later runs can overwrite plots.

---

## `bin/` — Data collection apps

Two standalone PyQt5 desktop apps used to **record** sensor data in the field.
These are separate from the analysis code (each has its own
`requirements.txt`) and aren't imported by anything in `src/`.

```
bin/
├── record/                      Basic multi-channel PicoScope recorder
│   ├── src/main.py              PyQt5 GUI: live scope view, start/stop recording to .npy
│   └── lib/pico/                Git submodule: PicoScope Python SDK wrapper
└── record_with_realtime_tracking/
    └── src/
        ├── main.py              GUI entry point (adds BLE + live HR tracking on top of record/)
        ├── hr_analysis.py       Real-time heart-rate estimation while recording
        ├── hr_panel.py          Live HR display widget
        └── epoch_axis.py        Time-axis helper for the live plot
```

`record_with_realtime_tracking` additionally pulls in a Bluetooth reference
device (`bleak`) and live audio (`pyaudio`) so a maternal/reference HR trace
can be watched in real time while recording — useful for confirming sensor
placement is good before a session ends.

---

## `lib/` — Models: pretrained code, fine-tuning, and checkpoints

Everything needed to separate/detect the fetal heartbeat via deep learning
lives here: the model architectures, their training scripts, and the trained
weights (checkpoints), versioned as you iterate.

```
lib/
├── neossnet/            Git submodule (external repo, not vendored):
│                         base pretrained NeoSSNet source-separation model + its
│                         own training code. Must be fetched via
│                         `git submodule update --init --recursive`.
│
├── tune-ssnet/           Fine-tunes NeoSSNet on this project's own recordings
│   ├── src/main.py       Fine-tuning entry point (imports the submodule's model/train code)
│   ├── fetal-tune-config.yaml     Config: fine-tune towards the FETAL source
│   ├── maternal-tune-config.yaml  Config: fine-tune towards the MATERNAL source
│   ├── training_clips.yaml        Which recording windows/fibers to build snippets from
│   ├── generate_training_snippets.sh  Builds the training snippet set from training_clips.yaml
│   └── models/                    One folder per fine-tuning run (tuned-model-v1 … v13,
│                                   maternal-tuned-model-v2), each with model_best.pt +
│                                   model.yaml (architecture config used for that checkpoint)
│
├── common/               Shared FUNet training/eval infrastructure.
│   │                     Not a model itself. The current tune-ssnet/src/main.py instead
│   │                     imports its training loop from the NeoSSNet submodule.
│   ├── config.py          Config dataclasses + the YAML loader (see "Config schema" below)
│   ├── task.py            The seam each model implements (build model, loss, data)
│   ├── engine.py          The training loop itself
│   ├── losses.py          Beat-activity losses (SNR, correlation, corr_amp, MSE, KL-div)
│   ├── metrics.py         HR-trace agreement between predicted activity and target
│   ├── optim.py           Optimiser + LR schedule construction ('none'/'cosine'/'plateau')
│   ├── augment.py         Random train-only waveform augmentation
│   ├── preprocess.py      Deterministic transforms ('bandpass', 'normalize') applied to
│   │                       every split AND to inference
│   ├── audio.py           Snippet I/O, resampling, cropping/padding helpers
│   ├── device.py / errors.py / io.py / diagnostics.py   Device pick, exceptions,
│   │                       atomic writes + training-curve plots, end-of-run diagnostics
│   └── phases/            The three things a model can do:
│       ├── train.py        Build everything from a config and run the loop
│       ├── inference.py    Load a checkpoint and run it over a long recording
│       │                    (windowing, the activity readout — see below)
│       └── optimize.py     Optuna hyperparameter search over any Task
│
├── funet/                A custom model (FUNet) trained from scratch to output a
│   │                     per-frame "beat activity" signal instead of a separated waveform
│   ├── funet/             The importable `funet` package (needs lib/funet on sys.path)
│   │   ├── model.py       FUNet architecture (U-Net-style conv encoder/decoder)
│   │   ├── config.py      FUNet's own config dataclasses, extending common.config
│   │   ├── data.py        Dataset for FUNet's snippet format + the passband crop helper
│   │   ├── task.py        Wires config → model → loss for common.engine
│   │   ├── train.py       Training entry point (thin wrapper over common.phases.train)
│   │   ├── optimize.py    Hyperparameter-search entry point
│   │   └── inference.py   Runs a trained FUNet checkpoint on a raw waveform
│   ├── src/               Legacy flat layout. The current analysis runner imports funet/,
│   │                       not this tree; use funet/ for that execution path.
│   ├── fetal-config.yaml           Default training config (architecture + hyperparameters)
│   ├── training_clips.yaml         Which recording windows/fibers to build snippets from
│   ├── generate_training_snippets.sh  Builds the training snippet set
│   ├── rfp_train/                 "Rough-then-fine pass" alternative training recipe:
│   │   ├── rough-pass-config.yaml / rough_pass_training_clips.yaml   stage 1: broad training
│   │   ├── fine-pass-config.yaml / fine_pass_training_clips.yaml     stage 2: fine-tune stage 1's output
│   │   └── intermediary/v1/, v2/  checkpoints saved between the two stages
│   └── models/                    One folder per training run (funet-v1 … v32, v34 … v36,
│                                   plus a "(CONTROL)" baseline), each with fetal-config.yaml +
│                                   model_best.pt / model_last.pt / training_curves.png
│
└── beats/                Hand-marked reference beat timestamps (.npy) for a few
                           patients, exported from the beat-marking app (src/beat_app);
                           used as ground truth outside the normal Banner_data layout.
```

**Model versioning convention:** configs use numbered folders under `models/`
(`funet-v36`, `tuned-model-v13`, …). Choose a new `model_dir` for a new run;
numbering alone does not prevent overwriting checkpoints in a reused directory. Which
checkpoint is actually *used* by the analysis pipelines is selected in
`src/constants.py` (`FUNET_MODEL`, `FETAL_MODEL`, etc. — see below).

A checkpoint is only loadable together with the config beside it: the number of
input channels, the spectrogram size and the passband crop all have to match the
weights. Two differences worth knowing when switching versions:

- **Channel count varies by version.** `funet-v24` is 5-channel, `funet-v36` is
  3-channel. The fiber list passed to `run_funet_pipeline` must be the same
  length as `config.model.channels`, or inference raises rather than guessing.
- **v35 and v36 crop the spectrogram** to a passband (`model.freq_crop_hz`,
  80–350 Hz) before the first convolution. Earlier versions leave it at full
  height. This is why they cannot run under the old `lib/funet/src` code.

### Config schema

Configs are strict: an unknown key raises instead of being silently ignored, so
a typo'd hyperparameter can't quietly keep its default. The three sections split
along one rule — **if changing a field invalidates an existing checkpoint, it
belongs in `model:`**.

| Section | Holds | Examples |
| --- | --- | --- |
| `model:` | Architecture + the input contract | `channels`, `dilations`, `base_channels`, `n_fft`, `hop_length`, `freq_crop_hz` |
| `train:` | The optimisation loop and regularisation | `optimizer`, `learning_rate`, `epochs`, `crop_len`, `loss`, `lr_schedule`, `augment`, `freq_mask`, `time_mask` |
| `data:` | Where snippets live and how they load | `train_dir`, `val_dir`, `val_fraction`, `num_workers`, `preprocess` |

Configs written for the older flat layout put `n_fft`/`hop_length` under `data:`,
called the held-out split `test_dir`, and kept `augment`/`freq_mask`/`time_mask`
in `data:`. These fields describe the current `funet/` package loader. Legacy `src/` code
and externally supplied configs may use other schemas; check the exact loader
and error rather than assuming every historical config is compatible.

---

## `src/` — Main analysis library, CLI tools, and beat-marking app

```
src/
├── constants.py           Shared config: paths, which model checkpoint is
│                          "active", sample rates, acoustic frequency bands, BPM ranges
├── analyze/               The core library (imported as `from analyze.X import ...`)
├── bin/                    Standalone CLI utilities (data prep, plotting, SNR)
└── beat_app/               Local web app for viewing/hand-correcting beat marks
```

### `src/constants.py`

The single place that picks **which trained model checkpoint each pipeline
uses** (currently `FUNET_MODEL = "funet-v24"`, `FETAL_MODEL = "tuned-model-v13"`),
plus shared physical constants: acoustic frequency bands for filtering
(maternal vs. fetal), BPM ranges used to sanity-check detected beats, device
sample rates, and per-patient data file names (`MIC_BEATS_FILE`,
`NST_DRIFT_LOG_FILE`, etc.). **To switch which model version a pipeline runs
against, edit this file.**

### `src/analyze/` — the core library

```
src/analyze/
├── data.py                 Audio / FiberData / FiberPair containers; loads
│                            raw recordings from a Banner_data patient folder
├── filters.py               Bandpass / notch filter stage builders
├── pipeline.py              Pipeline class: chains processing "stages" with
│                            optional source/input-hash caching; FORCE=True
│                            currently makes every stage rerun
├── main.py                  Entry point: wires stages together into full
│                            end-to-end runs (see "Pipelines" below)
│
│   -- Source separation methods (pick one per pipeline) --
├── ica.py                   ICA-based separation
├── mnmf.py                  Multichannel NMF separation
├── mlcmed.py                 MLCMED separation
├── nmcf.py                  NMCF separation
├── neossnet.py               NeoSSNet deep-learning separation pipeline
├── funet_runner.py           FUNet beat-activity model pipeline. Named *_runner, not
│                            funet.py, so it cannot shadow the real `funet` package:
│                            running src/analyze/main.py puts this directory on
│                            sys.path[0], where a funet.py would be found first
│
├── demodulation.py           Carrier/envelope experiment; called first by main.py
├── plot_raw_bubble_event.py   Focused raw-signal diagnostic
├── anc.py                    Adaptive noise cancellation (NLMS / Kalman) —
│                            used after separation to cancel remaining maternal contamination
├── imf_select.py             Selects which decomposition component is "the fetal one"
├── wavelet_denoise.py         Optional wavelet-based denoising stage
├── clean_data_template.py     Template/scratch script for one-off data cleaning
│
├── hr/                        Beat detection ("find heartbeat timestamps in a waveform")
│   ├── detect.py … detect_v9.py   Nine detector versions (v1 through v9), each
│   │                               a different peak-picking/segmentation strategy.
│   │                               Later versions do NOT simply supersede earlier
│   │                               ones — which detector is best depends on the
│   │                               signal (see "Choosing a beat detector" below).
│   │                               Each file's docstring explains why that version
│   │                               exists and how it differs from its predecessor.
│   ├── classify.py             Given several candidate separated sources, picks
│   │                            which one is "the fetal source"
│   └── utils.py                 Shared beat-detection helpers
│
├── sot.py                     Loads the "source of truth" (mic or PPG) and
│                              detects its reference beats
├── drift.py                    Corrects NST/mic beat timestamps for dropped-sample
│                              clock drift, given a per-patient drift log (see
│                              "Correcting NST clock drift" below)
├── evaluate.py / evaluate_v2.py / evaluate_v3.py
│                              Score detected beats against the SOT (precision/
│                              recall/F1, HR-over-time comparison plots) — three
│                              different alignment/scoring definitions; standard model
│                              runners currently call v3
├── plot_hr.py                 Plots heart-rate-over-time and beat/peak overlays
│                              (includes the drift-corrected comparison plot)
└── util.py                    Misc shared helpers (e.g. invoking NeoSSNet)
```

### `src/bin/` — standalone CLI utilities

Use the Python entry points with their actual CLI arguments and the repository
environment. Some scripts add import paths themselves. Shell wrappers vary:
several hardcode patients/paths and do not forward arguments or activate a venv.
Inspect a wrapper before using it; do not assume it is a generic CLI.

```
src/bin/
├── generate_training_snippets.py   Builds paired fetal-HR + SOT training
│                                    snippet datasets (used by both funet/
│                                    and tune-ssnet/ to prepare model training data)
├── pico2data.py (+ .sh)             Converts raw PicoScope CSV exports into
│                                    this project's .npy / .wav data format
├── concat.py                        Concatenates two .npy recordings into one
├── change_start.py                  Prepends silence to a .npy recording to
│                                    correct for a recording that started late
├── s1s2_diagnostic.py                Diagnostic script checking whether FUNet
│                                    timing errors are S1/S2 heart-sound swaps
├── analyze/
│   ├── analyze_waveforms.py (+ .sh)  Plots every waveform in a Banner_data
│   │                                folder (sanity-check raw recordings)
│   ├── plot_clips.py (+ .sh)         Plots the fiber/SOT clips referenced by a
│   │                                training-snippet yaml (sanity-check training data)
│   └── plot_ppg_window.py            Plots a time window of the PPG (pvs.npy) trace
├── peak_det/
│   └── plot_peak_detectors.py (+ .sh)  Compares its explicit v1–v8 registry
│                                      side-by-side; v9 is not registered
└── snr/
    └── calculate_snr.py (+ .sh)      Computes signal-to-noise ratio for a recording
```

### `src/beat_app/` — the beat-marking web app

A local tool for creating/correcting the hand-marked "ground truth" beat
timestamps used elsewhere (e.g. `lib/beats/*.npy`, `MIC_BEATS_FILE`). Load a
WAV, run any `hr/*_beat_detector` on it as a starting point, then manually
add/move/delete beat markers on a waveform canvas, and export to `.yaml` or
`.npy`. See [src/beat_app/README.md](src/beat_app/README.md) for the full
workflow and keyboard/mouse controls.

```
src/beat_app/
├── server.py            Stdlib HTTP server: routes, sessions, background detect jobs
├── detectors.py          Auto-discovers hr/*_beat_detector functions for the dropdown
├── audio_io.py            WAV loading; YAML/.npy beat (de)serialization
└── static/                 Frontend: index.html, style.css, app.js (Canvas UI), hr.html (pop-out HR graph)
```

Run it with:
```bash
python3 src/beat_app/server.py            # opens http://127.0.0.1:8000
```

---

## Setup

```bash
./setup.sh
source .venv/bin/activate
```

`main.py` and other `src/analyze/` entry points import `analyze.*` / `constants`
as top-level packages, so `src/` needs to be on `PYTHONPATH` — either
`PYTHONPATH=src python src/analyze/main.py` from the repository root.

The model packages resolve the same way, at runtime rather than by install:
`funet_runner.py` puts `lib/` and `lib/funet/` on `sys.path` so `import common.*`
and `import funet.*` work, and `util.py` adds `lib/neossnet` to `sys.path`
for its bare `from utils import ...`. These local model/analysis packages are not installed by `setup.sh`; their
import names resolve through these path entries. Dependencies are pip-installed.

This creates `.venv`, installs `requirements.txt`, and fetches the
`lib/neossnet` (and `bin/record/lib/pico`) git submodules. `bin/record*`
apps have their own separate `requirements.txt` — install those into the
same or a separate venv only if you need the recording GUIs.

---

## Running an analysis pipeline

The main way to run an end-to-end analysis is `src/analyze/main.py`. It
defines one function per separation method (`ica()`, `run_mlcmed_pipeline()`,
`run_nmcf_pipeline()`, `run_mnmf_pipeline()`, `run_raw_bandpass()`, plus
`run_funet_pipeline(...)` / `run_neossnet_pipeline(...)` imported from
`analyze.funet_runner` / `analyze.neossnet`). At the bottom of the file, under
`if __name__ == '__main__':`, the calls you want are left un-commented (most
are commented out — that's the mechanism for choosing which pipeline(s) run).

```bash
# 1. Pick the patient + time window at the top of the file:
#      PATIENT = "PT12_2"
#      WINDOW = 0, 340
# 2. Uncomment the pipeline call(s) you want at the bottom of the file.
# 3. Run it:
PYTHONPATH=src python src/analyze/main.py
```

### What the current entry point actually runs

`src/analyze/main.py` selects `PATIENT = "Patient 6"`, `WINDOW = (0, 800)`,
and `DATA_DIR = <repo>/Banner_data/Banner_test_20251220/Patient 6`.
Its executable block calls the following independent experiments in order:

1. `run_demodulation_pipeline(PATIENT, WINDOW, DATA_DIR, zoom_start_s=100.0)`.
2. `run_funet_pipeline(..., fibers=["1B", "2A", "2B", "2C", "2D"])`.
3. `run_neossnet_pipeline(PATIENT, WINDOW, DATA_DIR)`.
4. `run_raw_bandpass()`, including its extra v9 comparison branch.

These are separate analyses of the recording, not stages feeding one another.
An exception in an earlier call prevents later calls from running. Model selection
remains `funet-v24` and `tuned-model-v13` in `src/constants.py`.

### Shared loading and reference branch

`data.py::load_fibers` returns chest plus five named abdomen `Audio` objects;
`windowed` selects each signal by its own timestamps. The two PicoScope bundles
have independently inferred sample rates. Offline `use_funet` truncates selected
arrays to their shortest length and uses the first fiber's time axis/rate; it
does not align separate device clocks or verify identical sampling grids.

`src/analyze/sot.py::load_sot` loads the entire recording: PPG column 1 from
`pvs.npy` is detrended and Butterworth-filtered at 0.7–4 Hz; microphone channel 0
is assigned `sample_index / mic_fs` timestamps and Chebyshev-I-filtered at
190–220 Hz. It returns `SOTData`, despite its stale `SOTResult` annotation.
The standard runners then window it and call `hr/__init__.py::sot_beats`.
That stage filters the microphone again at 190–220 Hz before the chosen
fetal detector, and uses `sot.py::detect_ppg_beats` for maternal reference beats.
This reference is detected from audio/PPG, not an imported NST numerical HR curve.

`sot_beats(..., data_dir=...)` prefers hand-marked `mic_beats.npy` if present.
The standard model/raw/demodulation reference calls do not pass `data_dir`,
so they use automatic detection even if a hand-label file exists. Belly-machine
runners do pass it. See the detector table below for other variants.

### Active signal paths

| Experiment | Execution after loading/windowing | Outputs under `.out/<patient>/` |
| --- | --- | --- |
| Demodulation | `demodulation_demodulate`: default `belly_1B` maps to `1B`; fourth-order Butterworth 200–230 Hz → analytic signal → carrier estimate and two-pass complex downconversion/25 Hz low-pass → amplitude envelope. Run envelope-aware v7 and ordinary v7 on the envelope, plus ordinary v7 on the bandpassed acoustic control. Compare with dropout-corrected v7 microphone beats. | `demodulation/`: `demodulated_fiber.html`, `bandpass_fiber.html`, `hr_comparison_demodulated.png`, `hr_comparison_demodulated_non_aware.png`, plus per-mode `diagnostics/` |
| FUNet | `use_funet`: stack all five ordered fibers → `funet.inference.run_funet`: resample to 4 kHz, normalize crop-length blocks, apply configured preprocessing, log1p power spectrogram, windowed model activity, native-grid interpolation → `fiber_beats(v9)` → raw/corrected HR plots → `evaluate_v3(hr_smooth=20)`. Selected v24 uses 7-second crops, MSE, no enabled `data.preprocess` transforms and no configured spectral crop. | `funet/`: `funet_activity.png`, `hr_comparison.png`, `hr_comparison_corrected.png`, `evaluation_v3.png`, detector diagnostics |
| NeoSSNet | Butterworth abdomen 190–220 Hz → select `1B` → `use_model`/`_run_neossnet_chunked` in chunks up to 30 seconds; peak-normalize and resample to/from 4 kHz via `util.run_neossnet` → select heart output key `"1"` → Butterworth 190–210 Hz → `fiber_beats(v7)` → raw/corrected plots → `evaluate_v3(hr_smooth=20)`. | `neossnet/`: `model_output.png`, `model_output_15_3.png`, raw/corrected HR comparisons, `evaluation_v3.png`, detector diagnostics |
| Raw bandpass | `FiberData.apply(bp(190,220,"butter"))` filters chest and abdomen → select `1B` → `fiber_beats(v7)` → raw and before/after-correction plots → `evaluate_v3`. Then rerun a separate v9 reference/fiber branch for the additional comparison plot. | `raw_bandpass_190_220/`: `hr_comparison.png`, `hr_comparison_before_after.png`, `evaluation_v3.png`, `hr_comparison_before_after_v9.png`, diagnostics |

`fiber_beats` also detects maternal chest beats when chest is present, using
its supplied detector after a 40–80 Hz filter. Thus changing the fiber detector
can affect both fetal and maternal results. The raw runner has already filtered
chest at 190–220 Hz before this maternal stage; this is the actual current chain.
The fetal wrapper defaults to 90–280 bpm; the maternal wrapper to 45–140 bpm.

Demodulation's `reject_hr_outliers=False` default means its additional
`hr_comparison_no_outlier.png` is **not** generated by the current `main.py` call.
Enabling it adds median-neighborhood HR rejection and a separate diagnostic;
it does not change the other comparison traces. The 100-second zoom setting
controls a 3-second full-rate excerpt in `bandpass_fiber.html`, not the overall
analysis window. See `tests/test_demodulation.py` for the existing DSP tests.
This experiment does not perform adaptive five-fiber bandwidth selection.

### Optional pipelines

These functions exist but are not called by the current executable block:

- `main.py::ica`: load/prepare fibers and references → segmented two-channel
  ICA (`ica.py::run_ica`, default pair `(0,1)`, belly on ps4000 and `2A`) →
  fetal bandpass → source classification → ANC → fiber v1 → evaluator v1.
- `main.py::run_mlcmed_pipeline`: 20-second windows → 20–250 Hz filter and
  50 Hz notch → multi-lag covariance/eigenvalue separation → fetal bandpass →
  classification/ANC → fiber v1 → evaluator v1; combine window results.
- `main.py::run_nmcf_pipeline`: broadband/notch preprocessing → `2B` →
  EMD/NMCF → source classification → fiber v3 → evaluator v1.
- `main.py::run_mnmf_pipeline`: broadband/notch preprocessing → multichannel
  abdomen MNMF → source classification → fiber v2 → evaluator v2. This path
  loads full SOT for its lag search, unlike the standard windowed model references.
- `funet_runner.py` and `neossnet.py` provide no-SOT, belly-machine and other
  variants. Their hardcoded channel lists must match the selected model. In
  particular, the FUNet three-channel belly-machine helper and two-channel
  no-SOT combinations do not match the currently selected five-channel v24.
- `clean_data_template.py::main` is a separate large analysis/grid-search
  workflow, not a preprocessing stage automatically run by `main.py`.

### Caching and run identity

`src/analyze/pipeline.py::Pipeline.FORCE` is currently **True**, so every stage
runs on every invocation. `Pipeline` still writes source/input signatures,
`pipe.json`, and stage-named `.pkl` results for stages lasting at least the
configured `min_cache_seconds` (default 2 seconds).

If `FORCE` is disabled, reuse depends on input/source signatures, pipeline stage
order, and cached-file existence. These signatures are not a complete manifest:
a path-string input does not hash the recording's contents, and captured settings,
checkpoint contents and YAML changes are not all represented by stage source.
Keep different experiment settings in separate output/cache paths. Several
reference runners share `cache_sot/neossnet/<patient>`; outputs are not uniformly
keyed by window, checkpoint, detector and fiber choice.

### Choosing a pipeline / model version

- **Which separation method** → which function you call in `main.py`
  (`run_funet_pipeline`, `run_neossnet_pipeline`, `run_mnmf_pipeline`, etc).
- **Which trained checkpoint** that method uses → edit `src/constants.py`
  (`FUNET_MODEL`, `FETAL_MODEL`, `MATERNAL_MODEL_PATH`, `NEOSSNET_MODEL_PATH`).
- **Which patient recording / time window** → the `PATIENT` / `WINDOW`
  constants at the top of `main.py`.

### The activity readout (FUNet inference)

FUNet doesn't emit beat times. It emits one raw number per frame (about 15.6 per
second at hop 256 on the 4 kHz model rate), those numbers can be negative, and
whether their absolute scale is calibrated depends on the training loss.
The last step of inference turns them into the non-negative envelope every beat detector expects. That step is the
**readout**, and it is chosen from the training loss
(`common.phases.inference.ACTIVITY_POSTPROCESS`):

| `train.loss` | Readout | Why |
| --- | --- | --- |
| `kldiv` | `exp` | the forward pass already applied `log_softmax` |
| `mse` | `clamp` | trained toward a unit-peak comb, so the raw output is already calibrated |
| `snr`, `corr`, `corr_amp` | `standardize` | those losses are affine-invariant, so nothing pins scale or offset |

Older code applied a blanket softmax for every signal head. That was actively
wrong for the affine-invariant losses — softmax responds to scale
*exponentially*, so an output that drifted large collapsed toward an argmax — and
lossy for `mse`, where it flattened calibrated peaks toward uniform.

The readout can change peak shape and therefore detected timing/counts.
`normalize_blocks` also normalizes over training-crop-sized blocks and interpolates
the gains between block centers, rather than allowing one transient to set the
gain for an entire recording. It uses a shared peak across channels.

### Choosing a beat detector

Detector versions are alternative methods, not an accuracy ranking:

- **v2** (`hr/detect_v2.py`): Shannon energy → 100-sample smoothing → Hilbert
  envelope → spacing-constrained peaks and lobe recentering. A median/MAD-based
  prominence floor is available but **disabled by default** (`use_floor=False`).
- **v7** (`hr/detect_v7.py`): duration-dependent cardiac-state segmentation
  using transient-suppressed Shannon-energy features.
- **v8** (`hr/detect_v8.py`): amplitude envelope (RMS by default) with a local
  adaptive positive-peak threshold.
- **v9** (`hr/detect_v9.py`): envelope-based detection with a per-block adaptive
  floor, region/lobe processing, and interval rejection.

The v7/v9 files also define envelope-aware wrappers that skip extracting another
envelope from an input that is already an envelope. Demodulation uses the v7
wrapper; standard FUNet uses ordinary v9. `beat_app/detectors.py` dynamically
discovers detector functions, whereas the standalone comparison script explicitly
registers only v1–v8. Performance must be compared with fixed reference preparation,
recording/window, checkpoint, correction policy and HR metrics; historical numbers
from a different experiment do not establish the current default's accuracy.
The standalone comparison script runs its detectors on band-limited microphone
audio, with optional hand-marked mic beats as the comparison reference; it is not
a comparison of all fiber/model pipelines.

### Current wiring

| Pipeline | SOT detector | Fiber detector | Model |
| --- | --- | --- | --- |
| `run_demodulation_pipeline` | v7 | envelope-aware v7, ordinary v7; acoustic v7 control | none |
| `run_funet_pipeline` | v7 | **v9** | `FUNET_MODEL` (funet-v24 → 5 fibers) |
| `run_raw_bandpass` | v7 | v7 | none |
| `run_neossnet_pipeline` | v7 | v7 | `FETAL_MODEL` |
| ICA / MLCMED | v2 | v1 | none |
| NMCF | v2 | v3 | none |
| MNMF | v2 | v2 | none |
| FUNet belly-machine | hand labels if available, otherwise v7 | v2 | must match supplied channels |
| FUNet no-SOT pairs | none | v2 | must match two-channel inputs |
| NeoSSNet no-SOT | none | v2 (110–180 bpm fetal gate) | `FETAL_MODEL`, per fiber |
| NeoSSNet-on-NST | v5 | v5 | `FETAL_MODEL` |
| NeoSSNet belly-machine | hand labels if available, otherwise v7 | v7 | `FETAL_MODEL`, per fiber |

`use_funet` (in `src/analyze/funet_runner.py`) stacks the abdomen fibers passed
to `run_funet_pipeline` as channels and runs them through the U-Net in
`lib/funet/funet/model.py`. The fiber list must be as long as
`config.model.channels` — `main.py` passes all five (`1B, 2A, 2B, 2C, 2D`) for
the 5-channel funet-v24, overriding the function's 3-fiber default.

`run_raw_bandpass` additionally runs a second branch with v9 on *both* sides,
writing `hr_comparison_before_after_v9.png` plus detector diagnostics, so the v9 variant can be
eyeballed without disturbing the scored v7 baseline. It uses its own cache keys
(`sot_v9`, `cache_v9`) — this separates them if caching is enabled later; with the current `FORCE=True`,
both branches recompute.

`funet_runner.py` also defines `funet_beats()`, a peak-picker built specifically
for FUNet's activity envelope (adaptive-threshold `find_peaks`, no HMM), but no
pipeline calls it — it's there if you want another point of comparison.

---

## Correcting NST clock drift

The NST/mic recording's timestamps are computed as `sample_index / mic_fs`.
If the recording drops samples partway through, that nominal clock
increasingly falls behind real elapsed time — every beat timestamp
`detect_v7` (or any detector) reports on the NST gets progressively later in
real terms than the nominal time says. `analyze.drift` corrects for this
given a log of when dropouts happened and how much time each one cost.

**Format** — a CSV named `nst_dropouts.csv` (constant: `NST_DRIFT_LOG_FILE` in
`src/constants.py`), placed in the patient folder next to `microphone.wav`:

```csv
time_s,seconds_lost
12.340,0.008
45.900,0.015
201.220,0.006
```

- `time_s` — nominal NST time (same clock as `microphone.wav`) the dropout
  was detected at.
- `seconds_lost` — how much real time that dropout cost, in seconds (convert
  from a sample count using the mic's sample rate if that's what you have).

**What happens with it:** for each NST beat time, `drift.correct_drift` sums
`seconds_lost` over every dropout at or before that beat's nominal time, and
shifts the beat later by that cumulative amount. `run_funet_pipeline` and
`run_neossnet_pipeline` both run this automatically and write a **second**,
separate plot — `hr_comparison_corrected.png` — alongside the original
`hr_comparison.png`, so corrected vs. uncorrected can be compared directly.
`run_raw_bandpass` writes the stacked before/after version instead, as
`hr_comparison_before_after.png` (plus `hr_comparison_before_after_v9.png` from
its v9 branch). If `nst_dropouts.csv` doesn't exist for a patient, the corrected
plot is a no-op copy of the original.

**Correction applies to a plot copy, not the subsequent evaluator.**
`plot_hr_corrected` and `plot_hr_correction_comparison` build a replacement
`SOTResult` with corrected microphone beat times and return the original fiber
result unchanged. `evaluate_v3(sot, ...)` still holds the original, uncorrected
reference. PPG beats and the microphone waveform/time axis are not corrected by
these plotting stages. Demodulation explicitly uses corrected beats for its own
plots and printed agreement statistics. Always identify which reference a score uses.

## HR calculation, evaluation, and statistics

`plot_hr.py::_current_hr_func` selects `_inst_hr_v2`: sort beats → `60 / diff(beats)`
at the second beat of each interval → discard out-of-band rates → 20-beat
`moving_average_v2`. Fetal plot band is 90–280 bpm; maternal is 30–160 bpm.
Alternative HR functions in the file are not the selected default.
`_agreement_stats` reports zero-lag Pearson r and MAE on a 1 Hz overlapping grid.
Dropping intervals before interpolation can bridge gaps; smooth plots do not imply
complete beat coverage.

- `evaluate.py`: Gaussian-smoothed impulse-train lag estimation and robust
  one-to-one beat matching, with ±250 ms candidate tolerance, MAD rejection,
  and ±50 ms correct threshold; precision/recall/F1 and `evaluation.png`.
- `evaluate_v2.py`: windowed sigmoid-train lag tracking and timing evaluation;
  `evaluation_v2.png`. Its varying lag is distinct from logged dropout correction.
- `evaluate_v3.py`: 50 Hz HR-grid normalized cross-correlation, default ±5-second
  lag search, peak refinement, then timing matching after shifting predictions.
  Headline score is peak fetal HR correlation, not plot Pearson r or BPM MAE;
  writes `evaluation_v3.png` and console statistics. Correlation smoothing is
  fixed at 5 beats by default; display smoothing is separate (20 beats explicitly
  passed by FUNet/NeoSSNet, duration-scaled by default elsewhere).

Although evaluator v3's documentation expects full SOT for outside-window lag
support, standard runners window SOT before detection. They therefore cannot
supply reference samples outside that window. The raw runner's comment implying
full SOT does not match its actual stage list.

Training diagnostics are separate again: `lib/funet/funet/task.py::make_val_scorer`
uses **v2** on both predicted activity and the target comb, despite final FUNet
inference using v9. `lib/common/metrics.py::HRCorrelation` builds HR at interval
midpoints and compares traces on a 10 Hz grid. This checkout uses
`train.measure_hr_corr` and optimization objectives `loss`/`hr_corr`; it does not
have Yongyi's newer `HRMetrics`/`hr_agree` API. Ordinary checkpoint selection in
`common.engine.fit` follows validation loss, not the plotted HR correlation.

---

## Training / fine-tuning a model

Both deep-learning methods follow the same three-step recipe: **build
training snippets → train → point `constants.py` at the new checkpoint.**

### FUNet (train from scratch)

```bash
# 1. Build training snippets from lib/funet/training_clips.yaml
./lib/funet/generate_training_snippets.sh

# 2. Train (src also supports the optional HR validation scorer — see Setup)
PYTHONPATH=src:lib:lib/funet .venv/bin/python -m funet.train lib/funet/fetal-config.yaml

# Optional: Optuna hyperparameter search instead of a single run
PYTHONPATH=src:lib:lib/funet .venv/bin/python -m funet.optimize lib/funet/fetal-config.yaml
```
Checkpoints land in the config's `model_dir` (choose a new
`lib/funet/models/funet-vN/` per experiment). The current shared trainer archives
**`config.yaml`**, whereas `src/constants.py::FUNET_CONFIG` currently constructs
**`fetal-config.yaml`** beside the selected checkpoint. When using a newly trained
run, explicitly point `FUNET_CONFIG` at the actual archived config; changing only
`FUNET_MODEL` is insufficient if that expected filename is absent.
An alternative "rough pass, then fine pass" recipe lives in
`lib/funet/rfp_train/` for staged training.

### NeoSSNet fine-tuning

```bash
# 1. Build training snippets from lib/tune-ssnet/training_clips.yaml
./lib/tune-ssnet/generate_training_snippets.sh

# 2. Fine-tune towards the fetal source (or maternal-tune-config.yaml for the maternal source)
.venv/bin/python lib/tune-ssnet/src/main.py lib/tune-ssnet/fetal-tune-config.yaml
```
This starts from the pretrained `lib/neossnet` submodule model and fine-tunes
on this project's own recordings. Checkpoints land under
`lib/tune-ssnet/models/tuned-model-vN/`.

After training, update checkpoint **and configuration** paths in `src/constants.py`
and verify the runner's ordered fibers match the checkpoint's input channels.
Do not assume the default training config, snippet-generation output directory,
and selected analysis checkpoint refer to the same experiment.

### Snippet generation and labels

`src/bin/generate_training_snippets.py::build_jobs` reads each patient's ordered
`fibers`, `sections`, and train/test `mode` from YAML. Stereo mode stacks the
selected fibers; mono mode creates one snippet per fiber. `stack_resampled`
resamples to 4 kHz; `write_multichannel` peak-normalizes across channels.
`write_snippet` writes `{i}_mix.wav`, `{i}_heart.wav`, and configured optional
lung/noise targets and diagnostics.

With `mic_beats` enabled, available hand-marked microphone beats are preferred;
otherwise `fetal_detector` runs v7 on a **detrended, robust-clipped raw mic**.
This differs from `load_sot`/`sot_beats`' filtered reference preparation.
`heart_target` creates Gaussian or gated targets; optional `snap_to_energy`
uses filtered **channel 0** of the mixture and can move labels. Its bandpass is
on the mixture for target construction, not on the reference mic.

The FUNet shell wrapper currently writes `lib/funet/training/stereo_v12/`; the
SSNet wrapper writes `lib/tune-ssnet/training/training_clips_mono_v11/`.
Check the training config's `data.train_dir`/`val_dir` (or the SSNet loader's
schema) against these paths before starting training.

---

## Other common workflows

Run these from the repository root with its environment active. Placeholders
below must be replaced with actual paths. These are Python CLI forms; the shell
wrappers may contain hardcoded experiments instead of forwarding arguments.

```bash
# Convert one PicoScope CSV (choose --npy or --wav; output is positional).
python src/bin/pico2data.py <export.csv> <output.npy> --npy

# Plot raw recordings.
python src/bin/analyze/analyze_waveforms.py <patient_dir> --out-dir <plot_dir>

# Compare the explicitly registered v1–v8 detectors.
python src/bin/peak_det/plot_peak_detectors.py <patient> --start 100 --end 240 --smooth 10

# SNR utility takes a CSV and an output image, not a patient directory.
python src/bin/snr/calculate_snr.py <input.csv> <output.png>

# Beat-marking app.
python src/beat_app/server.py

# Recording transformations: the final positional argument is the output path.
python src/bin/change_start.py <input.npy> <seconds_to_prepend> <output.npy>
python src/bin/concat.py <first.npy> <second.npy> <output.npy>
```

`src/bin/peak_det/plot_peak_detectors.sh` currently hardcodes patient8-session1
and a 100–240 second window. `src/bin/pico2data.sh` currently executes two
`change_start.py` calls on hardcoded belly-machine files; its name does not mean
it is a generic safe wrapper for converting an arbitrary CSV.

---

## Data layout expected under `Banner_data/`

Each patient/session is a folder containing (names driven by
`src/constants.py`):
- `ps4000.npy` (`FIBER_BUNDLE_A`) — column 0 time, column 1 chest,
  column 2 abdominal **`1B`**.
- `ps3000a.npy` (`FIBER_BUNDLE_B`) — column 0 time, columns 1–4 abdominal
  **`2A`, `2B`, `2C`, `2D`**, respectively.
- `microphone.wav` — reference microphone recording (SOT)
- `mic_beats.npy` — hand-marked fetal beat times for that mic recording
  (from the beat-marking app)
- `pvs.npy` — PPG reference data; required by standard `load_sot` paths,
  omitted only by explicitly no-PPG/no-SOT variants
- `nst_dropouts.csv` *(optional)* — dropped-sample drift log for the NST/mic
  recording; see "Correcting NST clock drift" above

This directory is gitignored; you need real patient data locally (or
generated synthetic/test recordings) for any pipeline to run.
