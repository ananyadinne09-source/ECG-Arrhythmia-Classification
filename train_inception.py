import os
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\anany\Documents\major project\processed"

BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading processed dataset...")

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

print("X Shape :", X.shape)
print("y Shape :", y.shape)
print("Classes :", len(label_encoder.classes_))

# ----------------------------------------------------
# Remove classes having fewer than 2 samples
# ----------------------------------------------------
unique, counts = np.unique(y, return_counts=True)

valid_classes = unique[counts >= 2]

mask = np.isin(y, valid_classes)

X = X[mask]
y = y[mask]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nRemoved extremely rare classes")
print("New X shape:", X.shape)
print("New y shape:", y.shape)
print("Remaining classes:", len(np.unique(y)))
# ============================================================
# TRAIN TEST SPLIT
# ============================================================
def create_chunks(X, y, chunk_size=1000, stride=500):

    X_chunks = []
    y_chunks = []

    for signal, label in zip(X, y):

        total_length = signal.shape[1]

        for start in range(0, total_length - chunk_size + 1, stride):

            chunk = signal[:, start:start+chunk_size]

            X_chunks.append(chunk)
            y_chunks.append(label)

    return np.array(X_chunks), np.array(y_chunks)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train,
)

print("\nTrain :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

print("\nCreating 2-second ECG chunks...")

X_train, y_train = create_chunks(X_train, y_train)
X_val, y_val = create_chunks(X_val, y_val)
X_test, y_test = create_chunks(X_test, y_test)

print("\nAfter Chunking")
print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)
# ============================================================
# CLASS WEIGHTS
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
)

weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

# ============================================================
# DATASET
# ============================================================
def augment(signal):

    # Gaussian noise
    if np.random.rand() < 0.5:
        signal = signal + np.random.normal(
            0,
            0.01,
            signal.shape
        ).astype(np.float32)

    # Random scaling
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.9,1.1)
        signal = signal * scale
        signal = signal.astype(np.float32)
    return signal

class ECGDataset(Dataset):
    def __init__(self,X,y,train=False):

        self.X = X
        self.y = y
        self.train = train

    def __len__(self):

        return len(self.X)

    def __getitem__(self,idx):

        signal = self.X[idx]

        if self.train:
            signal = augment(signal)

        signal = torch.tensor(signal,dtype=torch.float32)
        label = torch.tensor(self.y[idx],dtype=torch.long)

        return signal,label


train_loader = DataLoader(
    ECGDataset(X_train, y_train, train=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    ECGDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    ECGDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print("\nDataLoaders Ready.")

# ============================================================
# INCEPTION BLOCK
# ============================================================

class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels=32):

        super().__init__()

        self.b1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=9,
            padding=4,
        )

        self.b2 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=19,
            padding=9,
        )

        self.b3 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=39,
            padding=19,
        )

        self.pool = nn.Sequential(

            nn.MaxPool1d(
                kernel_size=3,
                stride=1,
                padding=1,
            ),

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)

        self.relu = nn.ReLU()

    def forward(self, x):

        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.pool(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.bn(x)

        return self.relu(x)

# ============================================================
# MODEL
# ============================================================

class InceptionTime(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.block1 = InceptionBlock(12)

        self.block2 = InceptionBlock(128)

        self.block3 = InceptionBlock(128)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.gap(x)

        x = x.squeeze(-1)

        return self.fc(x)


num_classes = len(np.unique(y_train))

model = InceptionTime(num_classes).to(DEVICE)

print(model)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5,
)
# ============================================================
# TRAINING
# ============================================================

best_val_f1 = 0.0
counter = 0
patience = 7

for epoch in range(EPOCHS):

    # ---------------- TRAIN ----------------
    model.train()

    train_loss = 0
    train_preds = []
    train_labels = []

    for batch_idx, (x, y_batch) in enumerate(train_loader):

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)}")

        x = x.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y_batch.cpu().numpy())

    scheduler.step()

    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average="weighted")

    # ---------------- VALIDATION ----------------
    model.eval()

    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():

        for x, y_batch in val_loader:

            x = x.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(x)

            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

            preds = outputs.argmax(dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y_batch.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="weighted")

    print(
    f"Epoch {epoch+1}/{EPOCHS} | "
    f"Train Loss {train_loss/len(train_loader):.4f} | "
    f"Val Loss {val_loss/len(val_loader):.4f} | "
    f"Train Acc {train_acc:.4f} | "
    f"Val Acc {val_acc:.4f} | "
    f"Val F1 {val_f1:.4f}"
)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "best_inception.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break

print("\nTraining Complete!")

# ============================================================
# TEST
# ============================================================

model.load_state_dict(torch.load("best_inception.pth", map_location=DEVICE))

model.eval()

preds = []
labels = []

with torch.no_grad():

    for x, y_batch in test_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        pred = outputs.argmax(dim=1)

        preds.extend(pred.cpu().numpy())
        labels.extend(y_batch.cpu().numpy())

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print("Accuracy :", accuracy_score(labels, preds))
print("Weighted F1 :", f1_score(labels, preds, average="weighted"))