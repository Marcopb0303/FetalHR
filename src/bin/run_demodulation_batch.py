r"""Compare fixed baseline and patient-specific optimized carrier demodulation.

Example (from the repository root):
    PYTHONPATH=src python src/bin/run_demodulation_batch.py \
        --dataset-dir Banner_data/Banner_test_20251220 --window 0 120 \
        --case "Patient 6,140,150,2A" --case "Patient 7,195,205,2B"

R and MAE use the existing comparison's display precision (2 and 1 decimals).
NaN means agreement is undefined, e.g. insufficient beats or constant HR.
Each case runs the 1B / 200–230 Hz baseline (both v7 modes), then the selected
optimized fiber/band on the same window. Optimized diagnostics live in optimized/.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

# Direct script execution otherwise finds src/bin/analyze before src/analyze.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze.demodulation import DEMOD_FIBERS, run_demodulation_pipeline
from constants import PROJECT_DIR


def parse_case(value):
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("case must be PATIENT,LOW_HZ,HIGH_HZ,FIBER")
    patient, low, high, fiber = parts
    if not patient or patient in (".", "..") or Path(patient).name != patient:
        raise argparse.ArgumentTypeError("PATIENT must be a patient directory name")
    if fiber not in DEMOD_FIBERS:
        raise argparse.ArgumentTypeError(f"FIBER must be one of {', '.join(DEMOD_FIBERS)}")
    try:
        low, high = float(low), float(high)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("LOW_HZ and HIGH_HZ must be numbers") from exc
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
        raise argparse.ArgumentTypeError("band must satisfy 0 < low_hz < high_hz (finite Hz)")
    return patient, low, high, fiber


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--window", type=float, nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--case", type=parse_case, action="append", required=True,
                        metavar="PATIENT,LOW_HZ,HIGH_HZ,FIBER")
    args = parser.parse_args(argv)
    start, end = args.window
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        parser.error("window must have finite START < END")

    rows = []
    for patient, low, high, fiber in args.case:
        row = dict(patient=patient, optimized_fiber=fiber,
                   optimized_band_low_hz=low, optimized_band_high_hz=high,
                   baseline_envelope_aware_R="", baseline_envelope_aware_MAE="",
                   baseline_non_aware_R="", baseline_non_aware_MAE="",
                   optimized_R="", optimized_MAE="", status="success", error="")
        try:
            baseline = run_demodulation_pipeline(
                patient, (start, end), str(args.dataset_dir / patient),
                fiber="1B", bandpass_hz=(200.0, 230.0),
            )
            row["baseline_envelope_aware_R"], row["baseline_envelope_aware_MAE"] = (
                baseline.agreement.display_values
            )
            row["baseline_non_aware_R"], row["baseline_non_aware_MAE"] = (
                baseline.non_aware_agreement.display_values
            )
            result = run_demodulation_pipeline(
                patient, (start, end), str(args.dataset_dir / patient),
                fiber=fiber, bandpass_hz=(low, high), optimized=True,
            )
            row["optimized_R"], row["optimized_MAE"] = result.agreement.display_values
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
        detail = (f"optimized R={row['optimized_R']} MAE={row['optimized_MAE']} BPM"
                  if row["status"] == "success"
                  else row["error"])
        print(f"[batch] {patient} | {fiber} | {low:g}–{high:g} Hz | {row['status']} | {detail}")

    summary = Path(PROJECT_DIR) / ".out" / "demodulation_adaptive_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    table = [fields] + [[str(row[field]).replace("\n", " ") for field in fields] for row in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(fields))]
    print("\nDemodulation summary (baseline 1B / 200–230 Hz; optimized envelope-aware v7; MAE in BPM)")
    for row in table:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
    print(f"Saved {summary}")
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
