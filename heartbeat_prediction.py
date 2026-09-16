import numpy as np
import torch
import joblib

# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_PATH = "processed/X_heartbeat_v2.npy"
Y_PATH = "processed/y_heartbeat_v2.npy"
CHECKPOINT_PATH = "best_inception_heartbeat.pth"
ENCODER_PATH = "processed/label_encoder.pkl"


# ============================================================
# MODEL
# ============================================================

import torch.nn as nn


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
        b, c, _ = x.shape

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

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = x

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = out + identity
        out = self.relu(out)

        return out


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
# LOAD CLASS NAMES
# ============================================================

encoder = joblib.load(ENCODER_PATH)

class_names = list(encoder.classes_)

print("Classes:")

for i, name in enumerate(class_names):
    print(i, ":", name)


# ============================================================
# LOAD HEARTBEAT DATASET
# ============================================================

X = np.load(
    X_PATH,
    mmap_mode="r"
)

y = np.load(Y_PATH)

print("\nHeartbeat dataset shape:", X.shape)
print("Labels shape:", y.shape)


# ============================================================
# LOAD HEARTBEAT MODEL
# ============================================================

model = InceptionTime(
    num_classes=len(class_names)
).to(DEVICE)

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

print("\nHeartbeat model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# SELECT ONE HEARTBEAT
# ============================================================

sample_index = 0

heartbeat = np.array(
    X[sample_index],
    dtype=np.float32
)

actual_id = int(y[sample_index])

print("\n" + "=" * 70)
print("SINGLE HEARTBEAT PREDICTION")
print("=" * 70)

print("\nSample index:", sample_index)

print(
    "Input shape:",
    heartbeat.shape
)

print(
    "Actual class:",
    class_names[actual_id]
)


# ============================================================
# PREDICTION
# ============================================================

heartbeat_tensor = torch.from_numpy(
    heartbeat
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


# ============================================================
# RESULT
# ============================================================

print("\nPredicted class:")

print(
    class_names[prediction]
)

print(
    "Confidence:",
    f"{confidence * 100:.2f}%"
)

print("\nClass probabilities:")

for i, probability in enumerate(
    probabilities.cpu().numpy()
):

    print(
        f"{class_names[i]:35s}"
        f"{probability * 100:6.2f}%"
    )

print("\n" + "=" * 70)