import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from train_inception import load_data, build_model


if __name__ == "__main__":

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_weights, num_classes = load_data()

    # ============================================================
    # BUILD MODEL
    # ============================================================

    model, criterion, optimizer, scheduler, scaler = build_model(
        num_classes=num_classes,
        class_weights=class_weights
    )

    # ============================================================
    # LOAD BEST CHECKPOINT
    # ============================================================

    print("\n" + "=" * 60)
    print("LOADING BEST CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        "best_inception.pth",
        map_location=next(model.parameters()).device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best model loaded successfully.")

    # ============================================================
    # CLASS NAMES
    # ============================================================

    with open("processed/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    class_names = list(label_encoder.classes_)

    print("\nClass order:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    # ============================================================
    # TEST MODEL
    # ============================================================

    predictions = []
    labels = []

    device = next(model.parameters()).device

    with torch.no_grad():

        for signals, targets in test_loader:

            signals = signals.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):
                outputs = model(signals)

            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)

    # ============================================================
    # ACCURACY & F1
    # ============================================================

    accuracy = accuracy_score(labels, predictions)

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted F1   : {weighted_f1:.4f} ({weighted_f1 * 100:.2f}%)")
    print(f"Macro F1      : {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    report = classification_report(
        labels,
        predictions,
        labels=np.arange(num_classes),
        target_names=class_names,
        digits=4
    )

    print(report)

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(num_classes)
    )

    print(cm)

    # ============================================================
    # RAW CONFUSION MATRIX
    # ============================================================

    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45
    )

    plt.title("Confusion Matrix - InceptionTime ECG Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("\nRaw confusion matrix saved as:")
    print("confusion_matrix.png")

    plt.show()

    # ============================================================
    # NORMALIZED CONFUSION MATRIX
    # ============================================================

    cm_normalized = (
        cm.astype(float)
        / cm.sum(axis=1, keepdims=True)
    ) * 100

    print("\n" + "=" * 60)
    print("NORMALIZED CONFUSION MATRIX (%)")
    print("=" * 60)

    np.set_printoptions(
        precision=2,
        suppress=True
    )

    print(cm_normalized)

    # ============================================================
    # PLOT NORMALIZED CONFUSION MATRIX
    # ============================================================

    fig, ax = plt.subplots(figsize=(10, 8))

    disp_norm = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized,
        display_labels=class_names
    )

    disp_norm.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f",
        xticks_rotation=45
    )

    plt.title(
        "Normalized Confusion Matrix - InceptionTime ECG Classifier"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix_normalized.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("\nNormalized confusion matrix saved as:")
    print("confusion_matrix_normalized.png")

    plt.show()

    # ============================================================
    # FIND MISCLASSIFIED SAMPLES
    # ============================================================

    misclassified = np.where(labels != predictions)[0]

    print("\n" + "=" * 60)
    print("MISCLASSIFIED TEST SAMPLES")
    print("=" * 60)

    print(f"Total misclassified samples: {len(misclassified)}")

    print("\nFirst 30 misclassified samples:")

    for idx in misclassified[:30]:

        actual = class_names[labels[idx]]
        predicted = class_names[predictions[idx]]

        print(
            f"Test index {idx:4d} | "
            f"Actual: {actual:35s} | "
            f"Predicted: {predicted}"
        )

    # ============================================================
    # SAVE MISCLASSIFIED INDICES
    # ============================================================

    np.save(
        "misclassified_test_indices.npy",
        misclassified
    )

    print("\nMisclassified indices saved as:")
    print("misclassified_test_indices.npy")

    # ============================================================
    # DONE
    # ============================================================

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)