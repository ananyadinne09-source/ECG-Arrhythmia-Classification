"""
build_heartbeat_dataset_v2.py

Rebuilds the heartbeat-extraction dataset using the CORRECTED R-peak
detection method (signed signal + height threshold), validated in
validate_heartbeat_extraction_v2.py against 10 samples across all 5
classes.

Does NOT touch or overwrite:
  - processed/X_heartbeat.npy / y_heartbeat.npy  (old, buggy version)
  - best_inception.pth                            (existing baseline)
  - processed/X.npy / y.npy                        (source data, read-only)

Saves NEW files:
  - processed/X_heartbeat_v2.npy   shape (M, 12, 300)
  - processed/y_heartbeat_v2.npy   shape (M,)
  - processed/heartbeat_v2_skipped_indices.npy   (which original indices were dropped, and why)

M <= 31449 because some records will have zero valid peaks under the
stricter corrected detection and must be skipped (logged, not silently
dropped).
"""

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

X_PATH = "processed/X.npy"
Y_PATH = "processed/y.npy"

OUT_X_PATH = "processed/X_heartbeat_v2.npy"
OUT_Y_PATH = "processed/y_heartbeat_v2.npy"
OUT_SKIPPED_PATH = "processed/heartbeat_v2_skipped_indices.npy"

FS = 500
BEFORE_R = int(0.20 * FS)   # 100
AFTER_R = int(0.40 * FS)    # 200
PEAK_DISTANCE = int(0.20 * FS)
PEAK_PROMINENCE = 0.5
HEIGHT_FRACTION = 0.5       # validated fix: same as v2 validation script

CLASS_NAMES = {
    0: "Atrial Fibrillation",
    1: "Sinus Bradycardia",
    2: "Sinus Rhythm",
    3: "Sinus Tachycardia",
    4: "Supraventricular Tachycardia",
}


def detect_peaks_NEW(lead2_signal):
    """Validated corrected detection: signed signal + height threshold."""
    height_threshold = HEIGHT_FRACTION * lead2_signal.max()

    peaks, _ = find_peaks(
        lead2_signal,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
        height=height_threshold,
    )
    valid_peaks = [p for p in peaks if (p - BEFORE_R) >= 0 and (p + AFTER_R) <= len(lead2_signal)]
    return valid_peaks


def extract_heartbeat(ecg, valid_peaks):
    """Same selection rule as original: pick the middle valid peak."""
    selected_peak = valid_peaks[len(valid_peaks) // 2]
    start = selected_peak - BEFORE_R
    end = selected_peak + AFTER_R
    return ecg[:, start:end]  # (12, 300)


def main():
    print("Loading original ECG data...")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)
    n_total = X.shape[0]
    print(f"X shape: {X.shape}, y shape: {y.shape}\n")

    heartbeats = []
    labels = []
    skipped_indices = []
    skipped_per_class = {c: 0 for c in CLASS_NAMES}

    for i in tqdm(range(n_total), desc="Extracting heartbeats (v2, corrected)"):
        ecg = np.asarray(X[i], dtype=np.float32)
        lead2 = ecg[1]

        valid_peaks = detect_peaks_NEW(lead2)

        if len(valid_peaks) == 0:
            skipped_indices.append(i)
            skipped_per_class[int(y[i])] += 1
            continue

        heartbeat = extract_heartbeat(ecg, valid_peaks)
        heartbeats.append(heartbeat.astype(np.float32))
        labels.append(y[i])

    X_hb = np.stack(heartbeats, axis=0)
    y_hb = np.array(labels)

    np.save(OUT_X_PATH, X_hb)
    np.save(OUT_Y_PATH, y_hb)
    np.save(OUT_SKIPPED_PATH, np.array(skipped_indices))

    print("\n" + "=" * 60)
    print("HEARTBEAT EXTRACTION SUMMARY (v2, corrected detection)")
    print("=" * 60)
    print(f"Total input records   : {n_total}")
    print(f"Successfully extracted: {X_hb.shape[0]}")
    print(f"Skipped (no valid peak): {len(skipped_indices)}")
    print(f"\nX_heartbeat_v2 shape  : {X_hb.shape}")
    print(f"y_heartbeat_v2 shape  : {y_hb.shape}")

    print("\nFinal class distribution (kept):")
    unique, counts = np.unique(y_hb, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"   {CLASS_NAMES[int(u)]:<32} {c:>6}")

    print("\nSkipped per class:")
    for c, name in CLASS_NAMES.items():
        print(f"   {name:<32} {skipped_per_class[c]:>6}")

    print(f"\nSaved to:")
    print(f"   {OUT_X_PATH}")
    print(f"   {OUT_Y_PATH}")
    print(f"   {OUT_SKIPPED_PATH}  (original indices that had to be skipped)")


if __name__ == "__main__":
    main()