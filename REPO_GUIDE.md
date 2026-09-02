# fhr-analysis — Repo Guide

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
| **MLCMED** | An NMF/median-filter based separation method |
| **NMCF** | Non-negative Matrix Co-Factorization |
| **NeoSSNet** | A pretrained deep-learning source-separation model (submodule), optionally fine-tuned on this project's own recordings |
| **FUNet** | A custom-trained deep-learning model that predicts *beat activity* directly (skips separating a clean waveform) |
| **Raw bandpass** | No separation at all — just bandpass-filter the raw abdomen fiber into the fetal acoustic band |

FUNet and NeoSSNet are the two actively-developed, best-performing methods;
ICA/MNMF/MLCMED/NMCF are earlier/classical baselines kept for comparison.

---

## Top-level layout

```
fhr-analysis/
├── setup.sh              Bootstraps the Python venv + submodules (see README)
├── requirements.txt       Python deps for the analysis code (src/) and the models (lib/)
├── bin/                   Standalone data-collection apps (own venvs, PyQt GUIs)
├── lib/                   The separation/detection models: submodule + training code + checkpoints
└── src/                   The main analysis library, CLI tools, and the beat-marking web app
```

Two directories referenced throughout are **gitignored** (not in version
control, must exist locally or be fetched separately):
- `Banner_data/` — the actual patient recordings (fiber `.npy` bundles,
  microphone `.wav`, hand-marked beat `.npy` files). Nothing runs without this.
- `.out/` — cache and output files (plots, evaluation results) written by
  pipeline runs, keyed by patient/method/window.

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
├── common/               Shared training/eval infrastructure, used by every model in lib/.
│   │                     Not a model itself — it holds the parts that were duplicated
│   │                     across FUNet and tune-ssnet until they were factored out.
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
│   ├── src/               DEAD CODE — the pre-restructuring flat layout. Nothing imports
│   │                       it; kept only so old runs can be reproduced. Edit funet/, not this.
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

**Model versioning convention:** every training run gets its own numbered
folder under `models/` (`funet-v36`, `tuned-model-v13`, …) so old checkpoints
are never overwritten and results stay comparable across iterations. Which
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
in `data:`. Every config in this repo has been migrated; if you bring one in from
elsewhere and it fails to load, that relocation is why.

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
│                            automatic content-hash caching (re-running a
│                            pipeline skips stages whose inputs haven't changed)
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
│                              versions with progressively better lag-alignment
│                              and windowing logic; v3 is the most current
├── plot_hr.py                 Plots heart-rate-over-time and beat/peak overlays
│                              (includes the drift-corrected comparison plot)
└── util.py                    Misc shared helpers (e.g. invoking NeoSSNet)
```

### `src/bin/` — standalone CLI utilities

Each script inserts `src/` onto `sys.path` itself, so these can be run
directly (`python3 src/bin/<script>.py`) without extra setup, from anywhere.
Most have a matching `.sh` wrapper that activates the venv first.

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
│   └── plot_peak_detectors.py (+ .sh)  Runs and compares every hr/ beat
│                                      detector side-by-side on one patient
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
`PYTHONPATH=src python src/analyze/main.py`, or run from inside `src/`.

The model packages resolve the same way, at runtime rather than by install:
`funet_runner.py` puts `lib/` and `lib/funet/` on `sys.path` so `import common.*`
and `import funet.*` work, and `util.py` relies on `lib/neossnet` being importable
for its bare `from utils import ...`. Nothing here is pip-installed, so the import
names come entirely from those path entries.

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

Every pipeline follows the same shape:
1. Load the SOT (mic/PPG) reference and detect its reference beats.
2. Load the raw fiber recording, window it to the time range of interest, and
   bandpass/notch filter it.
3. Run the chosen separation method (or skip straight to bandpass-only).
4. Detect beats in the separated fetal source with one of the `hr/detect_v*`
   detectors.
5. Evaluate detected beats against the SOT and write plots/scores to
   `.out/<patient>/<method>/`.

Each `Pipeline` caches intermediate stage outputs by content hash under an
`out/.cache*` folder, so re-running with the same inputs is fast — only
stages whose upstream inputs actually changed are recomputed.

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
their absolute scale isn't pinned by training. The last step of inference turns
them into the non-negative envelope every beat detector expects. That step is the
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

This changes the envelope's **shape**, not just its scale, which is why it
matters downstream: on `funet-v24` the median sample sits at 26% of the peak
under `clamp` versus 67% under softmax. Inference also peak-normalises in
**blocks** of one training crop rather than once over the whole recording
(`normalize_blocks`), so a single loud transient can't shrink everything else.

### Choosing a beat detector

Detectors mainly differ in how they set their height floor, and that decides
which signals they suit:

- **v7** — Schmidt-style segmentation with transient suppression, gated on
  constants chosen against real heart-sound acoustics.
- **v2** — Hilbert envelope with a median + MAD floor, re-derived from the data.
- **v9** — `v4` with the percentile height floor measured per 4-second block
  instead of once over the whole recording.

Measured on PT12_2 against a drift-corrected SOT (Pearson R at 1 Hz):

| Pipeline | What the fiber detector sees | Best | Runner-up |
| --- | --- | --- | --- |
| FUNet | model activity envelope | **v9** (R +0.84) | v7 falls to +0.70 under the `clamp` readout |
| Raw bandpass | bandpassed fiber acoustics | **v7** (R +0.83) | v9 falls to +0.33 |

Same two detectors, opposite verdicts, for one reason: v9 *lowers* its gate
inside quiet blocks. On FUNet's activity envelope a quiet block still contains
real beats, so that recovers them. On raw acoustics a quiet block is quiet
because there is noise and no clear beat, so it admits the noise — 17.7% of v9's
intervals on the bandpassed fiber come out faster than 210 bpm, which is not
physiologically possible. Don't assume a higher version number is an upgrade;
measure it on the signal you actually have.

**The SOT side stays on v7 everywhere.** It runs on the real microphone, which
none of the model-side changes touch, and it is the yardstick every method is
scored against — moving it silently invalidates comparison with every earlier run.

### Current wiring

| Pipeline | SOT detector | Fiber detector | Model |
| --- | --- | --- | --- |
| `run_funet_pipeline` | v7 | **v9** | `FUNET_MODEL` (funet-v24 → 5 fibers) |
| `run_raw_bandpass` | v7 | v7 | none |
| `run_neossnet_pipeline` | v7 | v7 | `FETAL_MODEL` |

`use_funet` (in `src/analyze/funet_runner.py`) stacks the abdomen fibers passed
to `run_funet_pipeline` as channels and runs them through the U-Net in
`lib/funet/funet/model.py`. The fiber list must be as long as
`config.model.channels` — `main.py` passes all five (`1B, 2A, 2B, 2C, 2D`) for
the 5-channel funet-v24, overriding the function's 3-fiber default.

`run_raw_bandpass` additionally runs a second branch with v9 on *both* sides,
writing only `hr_comparison_before_after_v9.png`, so the v9 variant can be
eyeballed without disturbing the scored v7 baseline. It uses its own cache keys
(`sot_v9`, `cache_v9`) — sharing them would make `Pipeline` hand back the v7
run's cached results and produce two identical plots.

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

**This correction dominates every other effect in the pipeline.** On PT12_2 it
shifts the last beat of a 644 s window by **+18.5 s**, and fetal-HR agreement
goes from R ≈ 0.45 uncorrected to R ≈ 0.83 corrected. Any score compared against
a run without the drift log is not comparable at all — check which one you are
looking at before reading anything into a difference.

---

## Training / fine-tuning a model

Both deep-learning methods follow the same three-step recipe: **build
training snippets → train → point `constants.py` at the new checkpoint.**

### FUNet (train from scratch)

```bash
# 1. Build training snippets from lib/funet/training_clips.yaml
./lib/funet/generate_training_snippets.sh

# 2. Train (lib and lib/funet must be importable — see Setup)
PYTHONPATH=lib:lib/funet .venv/bin/python -m funet.train lib/funet/fetal-config.yaml

# Optional: Optuna hyperparameter search instead of a single run
PYTHONPATH=lib:lib/funet .venv/bin/python -m funet.optimize lib/funet/fetal-config.yaml
```
Checkpoints land in the config's `model_dir` (convention: a new
`lib/funet/models/funet-vN/` per run), alongside a copy of the config that
produced them — so a run directory is self-contained and can be pointed at
directly. An alternative "rough pass, then fine pass" recipe lives in
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

After either training run, update `FUNET_MODEL` / `FETAL_MODEL` in
`src/constants.py` to point at the new version so the analysis pipelines
pick it up.

---

## Other common workflows

- **Bring in a new PicoScope recording:**
  `.venv/bin/python src/bin/pico2data.py <export.csv> --out <dest>` to convert
  it into the `.npy`/`.wav` layout the rest of the code expects.
- **Sanity-check a raw recording:**
  `src/bin/analyze/analyze_waveforms.sh <patient_dir>` to plot every channel.
- **Compare beat detectors on one recording:**
  `src/bin/peak_det/plot_peak_detectors.sh <patient>` to run all of
  `hr/detect_v1`…`v9` side-by-side against the hand-marked SOT.
- **Check signal quality:** `src/bin/snr/calculate_snr.sh <patient_dir>`.
- **Create/correct ground-truth beat marks:** `python3 src/beat_app/server.py`.
- **Fix a recording that started late, or splice two recordings together:**
  `src/bin/change_start.py`, `src/bin/concat.py`.

---

## Data layout expected under `Banner_data/`

Each patient/session is a folder containing (names driven by
`src/constants.py`):
- `ps4000.npy` — chest sensor bundle (`FIBER_BUNDLE_A`)
- `ps3000a.npy` — abdomen sensor bundle, fibers `1B`/`2A`/`2B`/`2C`/`2D`
  (`FIBER_BUNDLE_B`)
- `microphone.wav` — reference microphone recording (SOT)
- `mic_beats.npy` — hand-marked fetal beat times for that mic recording
  (from the beat-marking app)
- `pvs.npy` — PPG-derived source-of-truth data, where available
- `nst_dropouts.csv` *(optional)* — dropped-sample drift log for the NST/mic
  recording; see "Correcting NST clock drift" above

This directory is gitignored; you need real patient data locally (or
generated synthetic/test recordings) for any pipeline to run.
