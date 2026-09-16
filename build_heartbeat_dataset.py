import numpy as np
import pickle
from scipy.signal import find_peaks
from tqdm import tqdm


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"processed\X.npy"
LABEL_PATH = r"processed\y.npy"

OUTPUT_X = r"processed\X_heartbeat.npy"
OUTPUT_Y = r"processed\y_heartbeat.npy"

FS = 500

BEFORE_R = 0.20
AFTER_R = 0.40

BEFORE_SAMPLES = int(BEFORE_R * FS)
AFTER_SAMPLES = int(AFTER_R * FS)

HEARTBEAT_LENGTH = BEFORE_SAMPLES + AFTER_SAMPLES


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING ECG DATA")
print("=" * 60)

X = np.load(DATA_PATH, mmap_mode="r")
y = np.load(LABEL_PATH)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# OUTPUT ARRAYS
# ============================================================

heartbeat_X = []
heartbeat_y = []


# ============================================================
# PROCESS EACH ECG
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING ONE HEARTBEAT PER ECG")
print("=" * 60)

for index in tqdm(range(len(X))):

    ecg = np.array(
        X[index],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Use Lead 2 for R-peak detection
    # --------------------------------------------------------

    lead = ecg[1]

    detection_signal = np.abs(lead)

    peaks, properties = find_peaks(
        detection_signal,
        distance=int(0.20 * FS),
        prominence=0.5
    )

    # --------------------------------------------------------
    # Find valid peaks
    # --------------------------------------------------------

    valid_peaks = []

    for peak in peaks:

        start = peak - BEFORE_SAMPLES
        end = peak + AFTER_SAMPLES

        if start >= 0 and end <= ecg.shape[1]:

            valid_peaks.append(peak)

    # --------------------------------------------------------
    # Skip ECG if no valid heartbeat
    # --------------------------------------------------------

    if len(valid_peaks) == 0:
        continue

    # --------------------------------------------------------
    # Choose middle heartbeat
    # --------------------------------------------------------

    peak = valid_peaks[len(valid_peaks) // 2]

    start = peak - BEFORE_SAMPLES
    end = peak + AFTER_SAMPLES

    heartbeat = ecg[:, start:end]

    # Safety check
    if heartbeat.shape != (12, HEARTBEAT_LENGTH):
        continue

    heartbeat_X.append(heartbeat)
    heartbeat_y.append(y[index])


# ============================================================
# CONVERT TO NUMPY
# ============================================================

heartbeat_X = np.asarray(
    heartbeat_X,
    dtype=np.float32
)

heartbeat_y = np.asarray(
    heartbeat_y,
    dtype=np.int64
)


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 60)
print("HEARTBEAT DATASET")
print("=" * 60)

print("X heartbeat shape:", heartbeat_X.shape)
print("y heartbeat shape:", heartbeat_y.shape)

np.save(
    OUTPUT_X,
    heartbeat_X
)

np.save(
    OUTPUT_Y,
    heartbeat_y
)

print("\nSaved:")
print(OUTPUT_X)
print(OUTPUT_Y)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

unique, counts = np.unique(
    heartbeat_y,
    return_counts=True
)

for cls, count in zip(unique, counts):

    print(
        f"Class {cls}: {count}"
    )

print("\nDONE")