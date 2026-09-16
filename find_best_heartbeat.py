import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"processed\X.npy"

ECG_INDEX = 0
FS = 500

BEFORE_R = 0.20
AFTER_R = 0.40

BEFORE_SAMPLES = int(BEFORE_R * FS)
AFTER_SAMPLES = int(AFTER_R * FS)


# ============================================================
# LOAD ECG
# ============================================================

X = np.load(DATA_PATH, mmap_mode="r")

ecg = np.array(X[ECG_INDEX], dtype=np.float32)

print("ECG shape:", ecg.shape)


# ============================================================
# R-PEAK DETECTION
# ============================================================

lead = ecg[1]

detection_signal = np.abs(lead)

peaks, properties = find_peaks(
    detection_signal,
    distance=int(0.20 * FS),
    prominence=0.5
)


# ============================================================
# EXTRACT VALID HEARTBEATS
# ============================================================

heartbeats = []
peak_times = []

for peak in peaks:

    start = peak - BEFORE_SAMPLES
    end = peak + AFTER_SAMPLES

    if start >= 0 and end <= ecg.shape[1]:

        beat = ecg[:, start:end]

        heartbeats.append(beat)
        peak_times.append(peak / FS)


heartbeats = np.array(heartbeats)

print("Number of heartbeats:", len(heartbeats))
print("Heartbeat shape:", heartbeats.shape)


# ============================================================
# SCORE EACH HEARTBEAT
# ============================================================

scores = []

for i, beat in enumerate(heartbeats):

    # Use Lead 2 to judge beat quality
    signal = beat[1]

    # Amplitude range
    amplitude = np.max(signal) - np.min(signal)

    # Smoothness / high-frequency noise
    difference = np.diff(signal)
    noise = np.std(difference)

    # Higher amplitude + lower noise = better score
    score = amplitude / (noise + 1e-6)

    scores.append(score)


scores = np.array(scores)

# Get top 5
best_indices = np.argsort(scores)[::-1][:5]


# ============================================================
# PRINT BEST HEARTBEATS
# ============================================================

print("\n" + "=" * 60)
print("TOP 5 HEARTBEATS")
print("=" * 60)

for rank, index in enumerate(best_indices):

    print(
        f"Rank {rank + 1} | "
        f"Heartbeat {index + 1} | "
        f"R-peak time: {peak_times[index]:.3f}s | "
        f"Quality score: {scores[index]:.2f}"
    )


# ============================================================
# DISPLAY BEST HEARTBEAT
# ============================================================

best_index = best_indices[0]

best_heartbeat = heartbeats[best_index]

np.save(
    "best_single_heartbeat.npy",
    best_heartbeat
)

print("\nBest heartbeat saved as:")
print("best_single_heartbeat.npy")


# ============================================================
# PLOT BEST HEARTBEAT
# ============================================================

time = np.arange(
    BEFORE_SAMPLES + AFTER_SAMPLES
) / FS

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 16),
    sharex=True
)

for i in range(12):

    axes[i].plot(
        time,
        best_heartbeat[i]
    )

    axes[i].axvline(
        BEFORE_R,
        linestyle="--",
        linewidth=1
    )

    axes[i].set_ylabel(
        f"Lead {i + 1}"
    )

    axes[i].grid(alpha=0.2)


axes[-1].set_xlabel(
    "Time (seconds)"
)

fig.suptitle(
    "Best Single Heartbeat - 12 Lead ECG",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "best_single_heartbeat.png",
    dpi=300,
    bbox_inches="tight"
)

print("Saved:")
print("best_single_heartbeat.png")

plt.show()