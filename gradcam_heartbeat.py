import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import joblib


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = r"C:\Users\anany\Documents\major project\processed"
CHECKPOINT = r"C:\Users\anany\Documents\major project\best_inception_heartbeat.pth"
OUTPUT_DIR = r"C:\Users\anany\Documents\major project\xai_heartbeat_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "Atrial Fibrillation",
    "Sinus Bradycardia",
    "Sinus Rhythm",
    "Sinus Tachycardia",
    "Supraventricular Tachycardia"
]

NUM_CLASSES = 5


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y


class InceptionBlock(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        out_channels = 64

        self.branch1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=9,
            padding=4,
            bias=False
        )

        self.branch2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=19,
            padding=9,
            bias=False
        )

        self.branch3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=39,
            padding=19,
            bias=False
        )

        self.branch4_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.branch4_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.bn = nn.BatchNorm1d(256)

        self.relu = nn.ReLU(inplace=True)

        self.se = SEBlock(256)

    def forward(self, x):

        b1 = self.branch1(x)

        b2 = self.branch2(x)

        b3 = self.branch3(x)

        b4 = self.branch4_pool(x)
        b4 = self.branch4_conv(b4)

        out = torch.cat(
            [b1, b2, b3, b4],
            dim=1
        )

        out = self.bn(out)

        out = self.relu(out)

        out = self.se(out)

        return out


class ResidualInception(nn.Module):

    def __init__(self, channels=256):

        super().__init__()

        self.block1 = InceptionBlock(channels)
        self.block2 = InceptionBlock(channels)
        self.block3 = InceptionBlock(channels)

    def forward(self, x):

        identity = x

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x + identity

        x = F.relu(x)

        return x


class InceptionTime(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.stem = InceptionBlock(12)

        self.res1 = ResidualInception(256)

        self.res2 = ResidualInception(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.res1(x)

        x = self.res2(x)

        x = self.global_pool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.classifier(x)

        return x


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HEARTBEAT GRAD-CAM")
print("=" * 70)

print("Device:", DEVICE)

print("\nLoading heartbeat dataset...")

X = np.load(
    os.path.join(
        DATA_DIR,
        "X_heartbeat_v2.npy"
    )
)

y = np.load(
    os.path.join(
        DATA_DIR,
        "y_heartbeat_v2.npy"
    )
)

test_idx = np.load(
    os.path.join(
        DATA_DIR,
        "heartbeat_test_idx.npy"
    )
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Test samples:", len(test_idx))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained heartbeat model...")

model = InceptionTime(
    num_classes=NUM_CLASSES
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Checkpoint loaded successfully.")
print("NO TRAINING WILL BE PERFORMED.")


# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):

    global activations

    activations = output


def backward_hook(module, grad_input, grad_output):

    global gradients

    gradients = grad_output[0]


target_layer = model.res2.block3

target_layer.register_forward_hook(
    forward_hook
)

target_layer.register_full_backward_hook(
    backward_hook
)


# ============================================================
# FIND HIGH-CONFIDENCE CORRECT HEARTBEAT
# ============================================================

print("\nSearching test set for a high-confidence correct heartbeat...")

best_confidence = -1
best_index = None
best_prediction = None
best_true = None

with torch.no_grad():

    for idx in test_idx:

        signal = torch.tensor(
            X[idx],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        output = model(signal)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        confidence = confidence.item()
        prediction = prediction.item()
        true_label = int(y[idx])

        if prediction == true_label:

            if confidence > best_confidence:

                best_confidence = confidence
                best_index = idx
                best_prediction = prediction
                best_true = true_label


print("\nBest heartbeat found.")

print(
    "True class:",
    CLASS_NAMES[best_true]
)

print(
    "Predicted class:",
    CLASS_NAMES[best_prediction]
)

print(
    "Confidence:",
    f"{best_confidence * 100:.2f}%"
)

print(
    "Dataset index:",
    best_index
)


# ============================================================
# COMPUTE GRAD-CAM
# ============================================================

print("\nComputing Grad-CAM...")

signal = torch.tensor(
    X[best_index],
    dtype=torch.float32
).unsqueeze(0).to(DEVICE)

signal.requires_grad_(True)

model.zero_grad()

output = model(signal)

score = output[0, best_prediction]

score.backward()


# ============================================================
# GRAD-CAM CALCULATION
# ============================================================

# Activations:
# (1, channels, time)

# Gradients:
# (1, channels, time)

weights = gradients.mean(
    dim=2,
    keepdim=True
)

cam = (
    weights * activations
).sum(
    dim=1
).squeeze(0)


cam = F.relu(cam)

cam = cam.detach().cpu().numpy()


# Normalize 0–1

if cam.max() > cam.min():

    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min()
    )

else:

    cam = np.zeros_like(cam)


# ============================================================
# UPSAMPLE CAM TO 300 SAMPLES
# ============================================================

cam_tensor = torch.tensor(
    cam,
    dtype=torch.float32
).view(
    1, 1, -1
)

cam_tensor = F.interpolate(
    cam_tensor,
    size=300,
    mode="linear",
    align_corners=False
)

cam = cam_tensor.squeeze().numpy()


# ============================================================
# SMOOTH CAM
# ============================================================

window = 15

kernel = np.ones(window) / window

cam_smooth = np.convolve(
    cam,
    kernel,
    mode="same"
)

cam_smooth = (
    cam_smooth - cam_smooth.min()
) / (
    cam_smooth.max() - cam_smooth.min() + 1e-8
)


# ============================================================
# FIND MOST IMPORTANT TIME
# ============================================================

peak_sample = np.argmax(
    cam_smooth
)

peak_time = peak_sample / 500.0

print(
    "\nMost important temporal region:",
    f"{peak_time:.3f} seconds"
)


# ============================================================
# PLOT
# ============================================================

time = np.arange(300) / 500.0

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 18),
    sharex=True
)

signal_np = X[best_index]

for lead in range(12):

    ax = axes[lead]

    ecg = signal_np[lead]

    ax.plot(
        time,
        ecg,
        linewidth=1
    )

    # Grad-CAM heatmap

    ax.imshow(
        cam_smooth[np.newaxis, :],
        aspect="auto",
        extent=[
            time[0],
            time[-1],
            np.min(ecg),
            np.max(ecg)
        ],
        alpha=0.35,
        origin="lower"
    )

    ax.set_ylabel(
        f"Lead {lead + 1}"
    )


axes[-1].set_xlabel(
    "Time (seconds)"
)

fig.suptitle(
    "Heartbeat-Level Grad-CAM",
    fontsize=16
)

fig.text(
    0.5,
    0.955,
    f"True: {CLASS_NAMES[best_true]} | "
    f"Predicted: {CLASS_NAMES[best_prediction]} | "
    f"Confidence: {best_confidence * 100:.2f}%",
    ha="center",
    fontsize=12
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)


# ============================================================
# SAVE
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "one_heartbeat_gradcam.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nGrad-CAM saved to:")

print(output_file)

print("\nDONE.")