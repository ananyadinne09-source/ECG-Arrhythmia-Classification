import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from train_inception import InceptionTime, ECGDataset, DEVICE, BATCH_SIZE

DATA_DIR = r"C:\Users\anany\Documents\major project\processed"

print("Loading processed dataset...")

X = np.load(os.path.join(DATA_DIR, "X.npy"), mmap_mode="r")
y = np.load(os.path.join(DATA_DIR, "y.npy"))

with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", encoder.classes_)
print("Number of classes:", len(encoder.classes_))

# Recreate the EXACT same split used during training
indices = np.arange(len(y))

idx_train, idx_test, y_train, y_test = train_test_split(
    indices,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

idx_train, idx_val, y_train, y_val = train_test_split(
    idx_train,
    y_train,
    test_size=0.10,
    random_state=42,
    stratify=y_train
)

print("\nSplit sizes:")
print("Train:", len(idx_train))
print("Validation:", len(idx_val))
print("Test:", len(idx_test))

# Validation loader
val_loader = DataLoader(
    ECGDataset(X, idx_val, y_val, train=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load existing best model
print("\nLoading best_inception.pth...")

checkpoint = torch.load(
    "best_inception.pth",
    map_location=DEVICE
)

print("Checkpoint epoch:", checkpoint["epoch"] + 1)
print("Stored best validation F1:", checkpoint["best_f1"])

model = InceptionTime(len(encoder.classes_)).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

predictions = []
labels = []

print("\nEvaluating validation set...")

with torch.no_grad():

    for signals, targets in val_loader:

        signals = signals.to(DEVICE)
        targets = targets.to(DEVICE)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):
            outputs = model(signals)

        preds = outputs.argmax(dim=1)

        predictions.extend(preds.cpu().numpy())
        labels.extend(targets.cpu().numpy())

val_accuracy = accuracy_score(labels, predictions)
val_f1 = f1_score(labels, predictions, average="weighted")

print("\n" + "=" * 60)
print("EXACT VALIDATION RESULTS")
print("=" * 60)

print(f"Validation Accuracy : {val_accuracy:.4f}")
print(f"Validation Accuracy : {val_accuracy * 100:.2f}%")

print(f"Validation F1       : {val_f1:.4f}")
print(f"Validation F1       : {val_f1 * 100:.2f}%")

print("=" * 60)