"""
validate_heartbeat_extraction.py

Visual QA for R-peak detection and heartbeat extraction, BEFORE any
heartbeat-based training happens. Does not modify or retrain anything.

Samples a few ECGs per class, re-runs peak detection on Lead 2 (same
logic as the extraction script), and plots:
  - full Lead 2 signal
  - all detected candidate peaks (from np.abs(lead))
  - the ONE peak that was actually selected for extraction
  - the 300-sample extraction window boundaries

Purpose: let you visually confirm whether the selected peak is a true
R-peak, or whether abs() + middle-peak selection picked something else
(e.g. a large T-wave).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ------------------------------------------------------------------
# CONFIG — must match build_heartbeat_dataset.py exactly
# ------------------------------------------------------------------
X_PATH = "processed/X.npy"              # original full ECGs (12, 5000)
Y_PATH = "processed/y.npy"
X_HB_PATH = "processed/X_heartbeat.npy" # extracted heartbeats (12, 300)

FS = 500
BEFORE_R = int(0.20 * FS)   # 100
AFTER_R = int(0.40 * FS)    # 200
PEAK_DISTANCE = int(0.20 * FS)
PEAK_PROMINENCE = 0.5

CLASS_NAMES = {
    0: "Atrial Fibrillation",
    1: "Sinus Bradycardia",
    2: "Sinus Rhythm",
    3: "Sinus Tachycardia",
    4: "Supraventricular Tachycardia",
}

SAMPLES_PER_CLASS = 2  # how many examples to check per class


def detect_peaks_same_as_extraction(lead2_signal):
    """
    Reproduces the exact detection logic from build_heartbeat_dataset.py
    so we're validating what was ACTUALLY used, not a cleaner version.
    """
    detection_signal = np.abs(lead2_signal)
    peaks, _ = find_peaks(
        detection_signal,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )

    # Same validity filter as extraction: must allow a full 300-sample window
    valid_peaks = [
        p for p in peaks
        if (p - BEFORE_R) >= 0 and (p + AFTER_R) <= len(lead2_signal)
    ]
    return peaks, valid_peaks


def plot_validation(ecg, class_id, sample_idx, save_path):
    """
    ecg: (12, 5000) original signal for this sample
    """
    lead2 = ecg[1]  # index 1 == Lead 2, matches extraction code

    all_peaks, valid_peaks = detect_peaks_same_as_extraction(lead2)

    if len(valid_peaks) == 0:
        print(f"  [WARNING] No valid peaks found for class {class_id}, sample {sample_idx} — extraction would have failed/skipped this one.")
        return

    selected_peak = valid_peaks[len(valid_peaks) // 2]  # same selection rule as extraction
    start = selected_peak - BEFORE_R
    end = selected_peak + AFTER_R

    fig, ax = plt.subplots(figsize=(14, 4))
    time_axis = np.arange(len(lead2)) / FS

    ax.plot(time_axis, lead2, color="black", linewidth=0.8, label="Lead 2")

    # All candidate peaks (from abs(lead)) — grey
    ax.scatter(np.array(all_peaks) / FS, lead2[all_peaks], color="grey", s=25, zorder=3, label="All detected candidates")

    # Valid peaks (window fits) — blue
    ax.scatter(np.array(valid_peaks) / FS, lead2[valid_peaks], color="blue", s=35, zorder=4, label="Valid candidates (window fits)")

    # The ONE peak actually selected for extraction — red star
    ax.scatter([selected_peak / FS], [lead2[selected_peak]], color="red", s=140, marker="*", zorder=5, label="SELECTED peak (used for extraction)")

    # Extraction window boundaries
    ax.axvspan(start / FS, end / FS, color="orange", alpha=0.15, label="Extracted 300-sample window")
    ax.axvline(start / FS, color="orange", linestyle="--", linewidth=1)
    ax.axvline(end / FS, color="orange", linestyle="--", linewidth=1)

    ax.set_title(f"Class: {CLASS_NAMES[class_id]}  |  Sample #{sample_idx}  |  "
                 f"{len(all_peaks)} candidates, {len(valid_peaks)} valid, selected index {len(valid_peaks)//2}")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    print("Loading original ECG data...")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    print(f"X shape: {X.shape}, y shape: {y.shape}\n")

    for class_id in sorted(CLASS_NAMES.keys()):
        indices = np.where(y == class_id)[0]
        if len(indices) == 0:
            print(f"Class {class_id} ({CLASS_NAMES[class_id]}): no samples found, skipping.")
            continue

        chosen = np.random.choice(indices, size=min(SAMPLES_PER_CLASS, len(indices)), replace=False)

        print(f"Class {class_id} ({CLASS_NAMES[class_id]}): validating {len(chosen)} sample(s)...")
        for sample_idx in chosen:
            ecg = np.asarray(X[sample_idx], dtype=np.float32)
            save_path = f"heartbeat_validation_class{class_id}_sample{sample_idx}.png"
            plot_validation(ecg, class_id, sample_idx, save_path)
        print()

    print("Done. Open the saved PNGs and check: does the red star sit on an actual R-peak")
    print("(the tall sharp spike), or did it land on a T-wave or other deflection?")


if __name__ == "__main__":
    main()