import numpy as np
import torch
from scipy.signal import find_peaks

from train_inception import load_data, build_model, DEVICE


# ============================================================
# SETTINGS
# ============================================================

SAMPLING_RATE = 500

CHECKPOINT_PATH = "best_inception.pth"

# Same heartbeat window
BEFORE_R = 0.20
AFTER_R = 0.40


# ============================================================
# LOAD CLASS NAMES
# ============================================================

import joblib

encoder = joblib.load(
    r"C:\Users\anany\Documents\major project\processed\label_encoder.pkl"
)

class_names = list(encoder.classes_)

print("Classes:")
for i, name in enumerate(class_names):
    print(i, ":", name)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    r"C:\Users\anany\Documents\major project\processed\X.npy",
    mmap_mode="r"
)

y = np.load(
    r"C:\Users\anany\Documents\major project\processed\y.npy"
)

print("\nDataset shape:", X.shape)


# ============================================================
# LOAD MODEL
# ============================================================

(
    train_loader,
    val_loader,
    test_loader,
    class_weights,
    num_classes
) = load_data()

model, criterion, optimizer, scheduler, scaler = build_model(
    num_classes,
    class_weights
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# SELECT ONE ECG
# ============================================================

sample_index = 0

signal = np.array(X[sample_index], dtype=np.float32)

actual_id = int(y[sample_index])

print("\nSelected ECG:", sample_index)
print("Actual class:", class_names[actual_id])


# ============================================================
# R-PEAK DETECTION
# ============================================================

# Lead II
ecg = signal[1]

peaks, _ = find_peaks(
    ecg,
    distance=int(0.25 * SAMPLING_RATE),
    prominence=0.5
)

print("\nDetected R-peaks:", len(peaks))


# ============================================================
# TEST ALL DETECTED HEARTBEATS
# ============================================================

print("\n")
print("=" * 70)
print("TESTING INDIVIDUAL HEARTBEATS")
print("=" * 70)

results = []

for beat_number, peak in enumerate(peaks):

    # --------------------------------------------
    # Extract heartbeat window
    # --------------------------------------------

    start = peak - int(
        BEFORE_R * SAMPLING_RATE
    )

    end = peak + int(
        AFTER_R * SAMPLING_RATE
    )

    # Skip beats too close to recording boundaries
    if start < 0 or end > signal.shape[1]:
        continue

    # --------------------------------------------
    # Create heartbeat-only signal
    # --------------------------------------------

    heartbeat_only = np.zeros_like(signal)

    heartbeat_only[:, start:end] = signal[:, start:end]

    # --------------------------------------------
    # Model prediction
    # --------------------------------------------

    heartbeat_tensor = torch.from_numpy(
        heartbeat_only
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(
            heartbeat_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        prediction = int(
            logits.argmax(dim=1).item()
        )

        confidence = probabilities[
            prediction
        ].item()

    # --------------------------------------------
    # Store result
    # --------------------------------------------

    results.append({
        "beat": beat_number + 1,
        "peak": peak,
        "time": peak / SAMPLING_RATE,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities.cpu().numpy()
    })


# ============================================================
# PRINT ALL RESULTS
# ============================================================

print("\nIndividual heartbeat predictions:\n")

for result in results:

    print(
        f"Beat {result['beat']:2d} | "
        f"Time: {result['time']:.3f}s | "
        f"Prediction: "
        f"{class_names[result['prediction']]:30s} | "
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )


# ============================================================
# FIND HIGHEST-CONFIDENCE HEARTBEAT
# ============================================================

best_result = max(
    results,
    key=lambda x: x["confidence"]
)


print("\n")
print("=" * 70)
print("BEST SINGLE HEARTBEAT")
print("=" * 70)

print(
    "\nHeartbeat number:",
    best_result["beat"]
)

print(
    "R-peak time:",
    f"{best_result['time']:.3f} seconds"
)

print(
    "Prediction:",
    class_names[
        best_result["prediction"]
    ]
)

print(
    "Confidence:",
    f"{best_result['confidence'] * 100:.2f}%"
)

print("\nClass probabilities:")

for i, probability in enumerate(
    best_result["probabilities"]
):

    print(
        f"{class_names[i]:35s}"
        f"{probability * 100:6.2f}%"
    )

print("\n" + "=" * 70)