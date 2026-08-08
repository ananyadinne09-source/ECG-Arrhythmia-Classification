"""
confusion_matrix.py
Generates a labeled, annotated confusion matrix for the best trained
checkpoint on the test set.
Run independently: python confusion_matrix.py
"""

import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from train_inception import load_data, build_model, DEVICE

CHECKPOINT_PATH = "best_inception.pth"
OUTPUT_IMG = "confusion_matrix.png"


def get_predictions(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for signals, targets in loader:
            signals = signals.to(DEVICE, non_blocking=True)
            outputs = model(signals)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.numpy())

    return np.array(all_labels), np.array(all_preds)


def load_class_names():
    """Maps new (0..num_classes-1) indices back to readable diagnosis names."""
    with open("class_names.pkl", "rb") as f:
        mapping = pickle.load(f)

    old_encoder = mapping["old_encoder"]
    new_encoder = mapping["new_encoder"]

    names = []
    for new_idx in range(len(new_encoder.classes_)):
        original_label_int = new_encoder.classes_[new_idx]
        name = old_encoder.inverse_transform([original_label_int])[0]
        names.append(str(name))

    return names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plots and saves a normalized confusion matrix with numbers in each cell."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (row-normalized) - InceptionTime ECG Classifier")

    # Annotate each cell with its value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            if value > 0.01:  # skip near-zero cells to reduce clutter
                text_color = "white" if value > 0.5 else "black"
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center",
                    fontsize=6, color=text_color
                )

    fig.colorbar(im, ax=ax, label="Proportion")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    print(f"\nSaved confusion matrix to: {save_path}")


def main():
    print("Loading data and model...")
    train_loader, val_loader, test_loader, class_weights, num_classes = load_data()

    model, criterion, optimizer, scheduler, scaler = build_model(num_classes, class_weights)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (best F1: {checkpoint['best_f1']:.4f})")

    y_true, y_pred = get_predictions(model, test_loader)

    class_names = load_class_names()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    plot_confusion_matrix(y_true, y_pred, class_names, OUTPUT_IMG)


if __name__ == "__main__":
    main()