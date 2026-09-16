import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SAMPLING_RATE = 500

X_PATH = r"C:\Users\anany\Documents\major project\processed\X.npy"

# Heartbeat window
BEFORE_R = 0.20   # 200 ms
AFTER_R = 0.40    # 400 ms

# --------------------------------------------------
# LOAD ECG
# --------------------------------------------------

X = np.load(X_PATH, mmap_mode="r")

print("Dataset shape:", X.shape)

# Select one ECG recording
signal = X[0]

print("ECG shape:", signal.shape)

# --------------------------------------------------
# R-PEAK DETECTION
# --------------------------------------------------

# Lead II
ecg = signal[1]

peaks, properties = find_peaks(
    ecg,
    distance=int(0.25 * SAMPLING_RATE),
    prominence=0.5
)

print("\nDetected R-peaks:")
print(peaks)

print("\nR-peak times:")
print(peaks / SAMPLING_RATE)

# --------------------------------------------------
# SELECT ONE GOOD R-PEAK
# --------------------------------------------------

# Choose a peak near the middle of the recording
target_time = 7.8

selected_peak = peaks[
    np.argmin(np.abs(peaks / SAMPLING_RATE - target_time))
]

selected_time = selected_peak / SAMPLING_RATE

print("\nSelected heartbeat:")
print("R-peak sample:", selected_peak)
print("R-peak time:", round(selected_time, 3), "seconds")

# --------------------------------------------------
# EXTRACT ONE HEARTBEAT
# --------------------------------------------------

start = selected_peak - int(BEFORE_R * SAMPLING_RATE)
end = selected_peak + int(AFTER_R * SAMPLING_RATE)

# Make sure we stay inside the recording
start = max(0, start)
end = min(signal.shape[1], end)

heartbeat = signal[:, start:end]

print("\nHeartbeat shape:", heartbeat.shape)
print("Expected approximately: (12, 300)")

# --------------------------------------------------
# TIME AXIS
# --------------------------------------------------

time = (
    np.arange(start, end) - selected_peak
) / SAMPLING_RATE

# --------------------------------------------------
# PLOT ONE HEARTBEAT - ALL 12 LEADS
# --------------------------------------------------

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 18),
    sharex=True
)

fig.suptitle(
    "Single Heartbeat - 12 Lead ECG",
    fontsize=18,
    fontweight="bold"
)

for lead in range(12):

    axes[lead].plot(
        time,
        heartbeat[lead],
        linewidth=1
    )

    axes[lead].axvline(
        0,
        linestyle="--",
        linewidth=1
    )

    axes[lead].set_ylabel(
        f"Lead {lead + 1}",
        fontsize=9
    )

    axes[lead].grid(
        True,
        alpha=0.3
    )

axes[-1].set_xlabel(
    "Time relative to R-peak (seconds)"
)

plt.tight_layout()

plt.show()