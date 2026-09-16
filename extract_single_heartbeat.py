import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"processed\X.npy"

ECG_INDEX = 0          # ECG to examine
FS = 500               # Sampling frequency

BEFORE_R = 0.20        # seconds before R-peak
AFTER_R = 0.40         # seconds after R-peak

BEFORE_SAMPLES = int(BEFORE_R * FS)
AFTER_SAMPLES = int(AFTER_R * FS)

HEARTBEAT_LENGTH = BEFORE_SAMPLES + AFTER_SAMPLES


# ============================================================
# LOAD ECG
# ============================================================

print("=" * 60)
print("LOADING ECG")
print("=" * 60)

X = np.load(DATA_PATH, mmap_mode="r")

print("Dataset shape:", X.shape)

ecg = np.array(X[ECG_INDEX], dtype=np.float32)

print("Selected ECG:", ECG_INDEX)
print("ECG shape:", ecg.shape)


# ============================================================
# R-PEAK DETECTION
# ============================================================

# Use Lead 2 for R-peak detection
lead = ecg[1]

# Make the signal positive so both polarities can be detected
detection_signal = np.abs(lead)

# Minimum distance between two R-peaks
min_distance = int(0.20 * FS)

peaks, properties = find_peaks(
    detection_signal,
    distance=min_distance,
    prominence=0.5
)

print("\nDetected R-peaks:", len(peaks))

print("R-peak times:")
for i, peak in enumerate(peaks):
    print(f"Peak {i + 1}: {peak / FS:.3f} seconds")


# ============================================================
# SELECT A VALID HEARTBEAT
# ============================================================

# Find peaks where the complete heartbeat fits inside the ECG
valid_peaks = []

for peak in peaks:

    start = peak - BEFORE_SAMPLES
    end = peak + AFTER_SAMPLES

    if start >= 0 and end <= ecg.shape[1]:
        valid_peaks.append(peak)

print("\nValid R-peaks:", len(valid_peaks))


# ============================================================
# SELECT ONE HEARTBEAT
# ============================================================

# Select the middle valid heartbeat for now
selected_peak = valid_peaks[len(valid_peaks) // 2]

start = selected_peak - BEFORE_SAMPLES
end = selected_peak + AFTER_SAMPLES

heartbeat = ecg[:, start:end]

print("\n" + "=" * 60)
print("SINGLE HEARTBEAT")
print("=" * 60)

print("R-peak:", selected_peak)
print("R-peak time:", selected_peak / FS, "seconds")
print("Heartbeat shape:", heartbeat.shape)
print("Expected shape:", (12, HEARTBEAT_LENGTH))


# ============================================================
# SAVE HEARTBEAT
# ============================================================

np.save(
    "single_heartbeat.npy",
    heartbeat
)

print("\nSaved:")
print("single_heartbeat.npy")


# ============================================================
# VISUALIZE ONE HEARTBEAT
# ============================================================

time = np.arange(HEARTBEAT_LENGTH) / FS

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 16),
    sharex=True
)

for i in range(12):

    axes[i].plot(time, heartbeat[i])

    axes[i].axvline(
        BEFORE_R,
        linestyle="--",
        linewidth=1
    )

    axes[i].set_ylabel(f"Lead {i + 1}")

    axes[i].grid(alpha=0.2)


axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "Single Heartbeat - 12 Lead ECG",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "single_heartbeat.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print("single_heartbeat.png")

plt.show()