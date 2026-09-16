"""
validate_heartbeat_extraction_v2.py

Tests a CORRECTED R-peak detection method against the ORIGINAL (buggy)
method on the same ECG samples, so the improvement (or lack of it) can
be seen directly. Does not modify any existing files or retrain anything.

Root cause of the original bug: np.abs(lead) treats T-wave bumps and
R-peak spikes as equally valid "peaks" in absolute-value space, so
find_peaks sometimes locks onto the T-wave instead of the R-peak.

Fix: detect peaks on the SIGNED signal (not abs), and require a minimum
HEIGHT (not just prominence) so only genuinely large positive deflections
(true QRS spikes) qualify -- T-waves are consistently smaller than R-peaks
in these examples, so a height threshold separates them reliably.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

X_PATH = "processed/X.npy"
Y_PATH = "processed/y.npy"

FS = 500
BEFORE_R = int(0.20 * FS)   # 100
AFTER_R = int(0.40 * FS)    # 200
PEAK_DISTANCE = int(0.20 * FS)
PEAK_PROMINENCE = 0.5

# NEW: height threshold as a fraction of this signal's own max amplitude.
# R-peaks are the dominant positive deflection in Lead 2 for these
# examples (T-waves sit at roughly 1.0-1.5 vs R-peaks at 4-6 in the
# samples already reviewed), so 0.5x max reliably excludes T-waves.
HEIGHT_FRACTION = 0.5

CLASS_NAMES = {
    0: "Atrial Fibrillation",
    1: "Sinus Bradycardia",
    2: "Sinus Rhythm",
    3: "Sinus Tachycardia",
    4: "Supraventricular Tachycardia",
}

SAMPLES_PER_CLASS = 2


def detect_peaks_OLD(lead2_signal):
    """Original (buggy) method: abs(signal), no height threshold."""
    detection_signal = np.abs(lead2_signal)
    peaks, _ = find_peaks(detection_signal, distance=PEAK_DISTANCE, prominence=PEAK_PROMINENCE)
    valid_peaks = [p for p in peaks if (p - BEFORE_R) >= 0 and (p + AFTER_R) <= len(lead2_signal)]
    return peaks, valid_peaks


def detect_peaks_NEW(lead2_signal):
    """
    Corrected method: signed signal (not abs), with a height threshold
    scaled to this signal's own amplitude, so only true QRS spikes
    qualify -- T-waves are excluded by height, not just prominence.
    """
    height_threshold = HEIGHT_FRACTION * lead2_signal.max()

    peaks, _ = find_peaks(
        lead2_signal,  # signed, not abs
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
        height=height_threshold,
    )
    valid_peaks = [p for p in peaks if (p - BEFORE_R) >= 0 and (p + AFTER_R) <= len(lead2_signal)]
    return peaks, valid_peaks


def plot_comparison(ecg, class_id, sample_idx, save_path):
    lead2 = ecg[1]
    time_axis = np.arange(len(lead2)) / FS

    old_all, old_valid = detect_peaks_OLD(lead2)
    new_all, new_valid = detect_peaks_NEW(lead2)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # --- OLD method ---
    ax = axes[0]
    ax.plot(time_axis, lead2, color="black", linewidth=0.8)
    ax.scatter(np.array(old_valid) / FS, lead2[old_valid], color="red", s=45, zorder=4, label="Detected (OLD: abs, no height)")
    ax.set_title(f"OLD method (buggy) — {len(old_valid)} valid peaks detected")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("Amplitude")

    # --- NEW method ---
    ax = axes[1]
    ax.plot(time_axis, lead2, color="black", linewidth=0.8)
    ax.scatter(np.array(new_valid) / FS, lead2[new_valid], color="green", s=45, zorder=4, label="Detected (NEW: signed + height)")
    ax.set_title(f"NEW method (corrected) — {len(new_valid)} valid peaks detected")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    fig.suptitle(f"Class: {CLASS_NAMES[class_id]}  |  Sample #{sample_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}  (OLD: {len(old_valid)} peaks, NEW: {len(new_valid)} peaks)")


def main():
    print("Loading original ECG data...")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    np.random.seed(42)  # reproducible sample selection

    for class_id in sorted(CLASS_NAMES.keys()):
        indices = np.where(y == class_id)[0]
        if len(indices) == 0:
            continue
        chosen = np.random.choice(indices, size=min(SAMPLES_PER_CLASS, len(indices)), replace=False)

        print(f"Class {class_id} ({CLASS_NAMES[class_id]}):")
        for sample_idx in chosen:
            ecg = np.asarray(X[sample_idx], dtype=np.float32)
            save_path = f"peak_comparison_class{class_id}_sample{sample_idx}.png"
            plot_comparison(ecg, class_id, sample_idx, save_path)
        print()

    print("Done. Compare OLD (top, red) vs NEW (bottom, green) on each image.")
    print("Check specifically: does green now sit on the tall sharp R-peak spikes")
    print("instead of the smaller rounded T-wave bumps?")


if __name__ == "__main__":
    main()