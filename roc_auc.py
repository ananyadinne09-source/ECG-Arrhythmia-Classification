import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc

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

    device = next(model.parameters()).device

    # ============================================================
    # LOAD BEST CHECKPOINT
    # ============================================================

    print("\n" + "=" * 60)
    print("LOADING BEST CHECKPOINT")
    print("=" * 60)

    checkpoint = torch.load(
        "best_inception.pth",
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Best model loaded successfully.")

    # ============================================================
    # LOAD CLASS NAMES
    # ============================================================

    with open("processed/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    class_names = list(label_encoder.classes_)

    print("\nClass order:")
    for i, name in enumerate(class_names):
        print(f"{i}: {name}")

    # ============================================================
    # GET PREDICTION PROBABILITIES
    # ============================================================

    all_probabilities = []
    all_labels = []

    print("\nRunning inference on test set...")

    with torch.no_grad():

        for signals, targets in test_loader:

            signals = signals.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):
                outputs = model(signals)

            probabilities = torch.softmax(outputs, dim=1)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                targets.cpu().numpy()
            )

    probabilities = np.concatenate(all_probabilities, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Test samples: {len(labels)}")
    print(f"Probability shape: {probabilities.shape}")

    # ============================================================
    # ROC-AUC FOR EACH CLASS
    # ============================================================

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):

        # One-vs-rest labels
        binary_labels = (labels == i).astype(int)

        fpr[i], tpr[i], _ = roc_curve(
            binary_labels,
            probabilities[:, i]
        )

        roc_auc[i] = auc(
            fpr[i],
            tpr[i]
        )

    # ============================================================
    # MACRO AUC
    # ============================================================

    macro_auc = np.mean(
        list(roc_auc.values())
    )

    # ============================================================
    # WEIGHTED AUC
    # ============================================================

    class_support = np.array([
        np.sum(labels == i)
        for i in range(num_classes)
    ])

    weighted_auc = np.average(
        list(roc_auc.values()),
        weights=class_support
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print("\n" + "=" * 60)
    print("ROC-AUC RESULTS")
    print("=" * 60)

    for i in range(num_classes):

        print(
            f"{class_names[i]:35s}: "
            f"{roc_auc[i]:.4f}"
        )

    print("-" * 60)

    print(f"Macro ROC-AUC    : {macro_auc:.4f}")
    print(f"Weighted ROC-AUC : {weighted_auc:.4f}")

    # ============================================================
    # PLOT ROC CURVES
    # ============================================================

    plt.figure(figsize=(10, 8))

    for i in range(num_classes):

        plt.plot(
            fpr[i],
            tpr[i],
            linewidth=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})"
        )

    # Random classifier
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curves - InceptionTime ECG Classifier"
    )

    plt.legend(
        loc="lower right",
        fontsize=9
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    plt.savefig(
        "roc_auc_curves.png",
        dpi=300,
        bbox_inches="tight"
    )

    print("\nROC curves saved as:")
    print("roc_auc_curves.png")

    plt.show()

    print("\n" + "=" * 60)
    print("ROC-AUC EVALUATION COMPLETE")
    print("=" * 60)