import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast, GradScaler

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\anany\Documents\major project"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

X_PATH = os.path.join(PROCESSED_DIR, "X_heartbeat_v2.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y_heartbeat_v2.npy")

TRAIN_IDX_PATH = os.path.join(PROCESSED_DIR, "heartbeat_train_idx.npy")
VAL_IDX_PATH = os.path.join(PROCESSED_DIR, "heartbeat_val_idx.npy")
TEST_IDX_PATH = os.path.join(PROCESSED_DIR, "heartbeat_test_idx.npy")

CHECKPOINT_PATH = os.path.join(BASE_DIR, "best_inception_heartbeat.pth")
HISTORY_PATH = os.path.join(BASE_DIR, "heartbeat_training_history.json")

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
PATIENCE = 6

NUM_CLASSES = 5
NUM_LEADS = 12
HEARTBEAT_LENGTH = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Atrial Fibrillation",
    "Sinus Bradycardia",
    "Sinus Rhythm",
    "Sinus Tachycardia",
    "Supraventricular Tachycardia"
]

print("=" * 70)
print("HEARTBEAT INCEPTIONTIME TRAINING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Input shape: ({NUM_LEADS}, {HEARTBEAT_LENGTH})")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()


# ============================================================
# DATASET
# ============================================================

class ECGDataset(Dataset):

    def __init__(self, X, y, indices, augment=False):
        self.X = X
        self.y = y
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        signal = self.X[real_idx].copy()
        label = int(self.y[real_idx])

        signal = torch.tensor(signal, dtype=torch.float32)

        # ----------------------------------------------------
        # Training augmentation
        # ----------------------------------------------------

        if self.augment:

            # Gaussian noise
            if np.random.rand() < 0.15:
                noise = torch.randn_like(signal) * 0.01
                signal = signal + noise

            # Amplitude scaling
            if np.random.rand() < 0.15:
                scale = np.random.uniform(0.9, 1.1)
                signal = signal * scale

        return signal, label


# ============================================================
# LOAD DATA
# ============================================================

print("Loading heartbeat dataset...")

X = np.load(X_PATH)
y = np.load(Y_PATH)

train_idx = np.load(TRAIN_IDX_PATH)
val_idx = np.load(VAL_IDX_PATH)
test_idx = np.load(TEST_IDX_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Train samples: {len(train_idx)}")
print(f"Validation samples: {len(val_idx)}")
print(f"Test samples: {len(test_idx)}")
print()


# ============================================================
# DATASETS
# ============================================================

train_dataset = ECGDataset(
    X,
    y,
    train_idx,
    augment=True
)

val_dataset = ECGDataset(
    X,
    y,
    val_idx,
    augment=False
)

test_dataset = ECGDataset(
    X,
    y,
    test_idx,
    augment=False
)


# ============================================================
# WEIGHTED RANDOM SAMPLER
# ============================================================

train_labels = y[train_idx]

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("Training class distribution:")

for i in range(NUM_CLASSES):
    print(
        f"{CLASS_NAMES[i]:30s}: {class_counts[i]}"
    )

class_weights = 1.0 / class_counts

sample_weights = class_weights[train_labels]

sample_weights = torch.tensor(
    sample_weights,
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# SQUEEZE-AND-EXCITATION BLOCK
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

        batch, channels, _ = x.size()

        y = self.pool(x).view(batch, channels)

        y = self.fc(y).view(
            batch,
            channels,
            1
        )

        return x * y


# ============================================================
# INCEPTION BLOCK
# ============================================================

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


# ============================================================
# RESIDUAL INCEPTION BLOCK
# ============================================================

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


# ============================================================
# INCEPTIONTIME MODEL
# ============================================================

class InceptionTime(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        self.stem = InceptionBlock(
            NUM_LEADS
        )

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
# MODEL
# ============================================================

model = InceptionTime(
    NUM_CLASSES
).to(DEVICE)

print("Model created.")

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print()


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.02
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

scaler = GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    all_preds = []
    all_labels = []

    for signals, labels in train_loader:

        signals = signals.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(signals)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item() * signals.size(0)
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.detach().cpu().numpy()
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

    epoch_loss = (
        running_loss / len(train_dataset)
    )

    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )

    epoch_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    return epoch_loss, epoch_acc, epoch_f1


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(loader, dataset):

    model.eval()

    running_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for signals, labels in loader:

            signals = signals.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(signals)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item() * signals.size(0)
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = (
        running_loss / len(dataset)
    )

    epoch_acc = accuracy_score(
        all_labels,
        all_preds
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro"
    )

    return (
        epoch_loss,
        epoch_acc,
        weighted_f1,
        macro_f1
    )


# ============================================================
# TRAINING LOOP
# ============================================================

history = []

best_val_f1 = 0.0
best_epoch = 0
patience_counter = 0

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = (
        train_one_epoch()
    )

    (
        val_loss,
        val_acc,
        val_f1,
        val_macro_f1
    ) = evaluate(
        val_loader,
        val_dataset
    )

    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Train F1: {train_f1 * 100:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}% | "
        f"Val F1: {val_f1 * 100:.2f}% | "
        f"Val Macro F1: {val_macro_f1 * 100:.2f}%"
    )

    print(
        f"Learning Rate: {current_lr:.8f}"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_f1": val_f1,
        "val_macro_f1": val_macro_f1,
        "learning_rate": current_lr
    })

    # --------------------------------------------------------
    # Save best checkpoint
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch + 1
        patience_counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1
            },
            CHECKPOINT_PATH
        )

        print(
            f"*** BEST MODEL SAVED "
            f"(Val F1: {best_val_f1 * 100:.2f}%) ***"
        )

    else:

        patience_counter += 1

        print(
            f"No improvement. "
            f"Patience: {patience_counter}/{PATIENCE}"
        )

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")

            break


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST HEARTBEAT MODEL")
print("=" * 70)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: {checkpoint['epoch'] + 1}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['best_val_f1'] * 100:.2f}%"
)


# ============================================================
# FINAL TEST
# ============================================================

(
    test_loss,
    test_acc,
    test_weighted_f1,
    test_macro_f1
) = evaluate(
    test_loader,
    test_dataset
)

print()
print("=" * 70)
print("FINAL HEARTBEAT TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   {test_acc * 100:.2f}%"
)

print(
    f"Weighted F1:     {test_weighted_f1 * 100:.2f}%"
)

print(
    f"Macro F1:        {test_macro_f1 * 100:.2f}%"
)

print()
print(f"Best checkpoint saved at:")
print(CHECKPOINT_PATH)

print()
print("Training completed.")
print("=" * 70)