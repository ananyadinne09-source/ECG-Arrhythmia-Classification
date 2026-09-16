import numpy as np
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

X_PATH = r"processed\X_heartbeat_v2.npy"
Y_PATH = r"processed\y_heartbeat_v2.npy"

TRAIN_IDX_PATH = r"processed\heartbeat_train_idx.npy"
VAL_IDX_PATH = r"processed\heartbeat_val_idx.npy"
TEST_IDX_PATH = r"processed\heartbeat_test_idx.npy"


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 60)
print("LOADING HEARTBEAT DATASET")
print("=" * 60)

y = np.load(Y_PATH)

print("Total samples:", len(y))
print("Labels shape:", y.shape)


# ============================================================
# FIRST SPLIT
# 80% train+validation
# 20% test
# ============================================================

indices = np.arange(len(y))

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# SECOND SPLIT
# 90% train
# 10% validation
#
# Overall:
# Train      = 72%
# Validation = 8%
# Test       = 20%
# ============================================================

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.10,
    random_state=42,
    stratify=y[train_val_idx]
)


# ============================================================
# SAVE INDICES
# ============================================================

np.save(TRAIN_IDX_PATH, train_idx)
np.save(VAL_IDX_PATH, val_idx)
np.save(TEST_IDX_PATH, test_idx)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("HEARTBEAT DATASET SPLIT")
print("=" * 60)

print("Train      :", len(train_idx))
print("Validation :", len(val_idx))
print("Test       :", len(test_idx))

print("\nTotal:", len(train_idx) + len(val_idx) + len(test_idx))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_names = {
    0: "Atrial Fibrillation",
    1: "Sinus Bradycardia",
    2: "Sinus Rhythm",
    3: "Sinus Tachycardia",
    4: "Supraventricular Tachycardia"
}


print("\n" + "=" * 60)
print("TRAIN CLASS DISTRIBUTION")
print("=" * 60)

unique, counts = np.unique(
    y[train_idx],
    return_counts=True
)

for cls, count in zip(unique, counts):
    print(f"{class_names[int(cls)]:<35} {count}")


print("\n" + "=" * 60)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 60)

unique, counts = np.unique(
    y[val_idx],
    return_counts=True
)

for cls, count in zip(unique, counts):
    print(f"{class_names[int(cls)]:<35} {count}")


print("\n" + "=" * 60)
print("TEST CLASS DISTRIBUTION")
print("=" * 60)

unique, counts = np.unique(
    y[test_idx],
    return_counts=True
)

for cls, count in zip(unique, counts):
    print(f"{class_names[int(cls)]:<35} {count}")


print("\n" + "=" * 60)
print("SPLIT FILES SAVED")
print("=" * 60)

print(TRAIN_IDX_PATH)
print(VAL_IDX_PATH)
print(TEST_IDX_PATH)

print("\nDONE")