# ============================================================
# XAI - GRAD-CAM FOR INCEPTIONTIME ECG ARRHYTHMIA CLASSIFIER
# ============================================================
#
# Uses:
#   - Existing InceptionTime model
#   - Existing best_inception.pth checkpoint
#   - Existing Chapman-Shaoxing processed test data
#
# Target layer:
#   model.res2.block3
#
# Output:
#   xai_results/
#       gradcam_Atrial_Fibrillation.png
#       gradcam_Sinus_Bradycardia.png
#       gradcam_Sinus_Rhythm.png
#       gradcam_Sinus_Tachycardia.png
#       gradcam_Supraventricular_Tachycardia.png
#
# ============================================================

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

# Import your existing model/data functions
from train_inception import (
    load_data,
    build_model,
    DEVICE
)


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT_PATH = "best_inception.pth"

OUTPUT_DIR = "xai_results"

# Chapman ECG recordings are 10 seconds at 500 Hz
SAMPLING_RATE = 500

# Number of classes/examples to explain
NUM_CLASSES_TO_EXPLAIN = 5

# Save figures at high resolution
DPI = 300


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # Register hooks
        self.forward_handle = target_layer.register_forward_hook(
            self.forward_hook
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self.backward_hook
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(self, module, input, output):

        self.activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    def generate(self, signal, target_class):

        self.model.zero_grad(set_to_none=True)

        # Forward pass
        output = self.model(signal)

        # Selected class score
        class_score = output[:, target_class]

        # Backward pass
        class_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ----------------------------------------------------
        # Activations shape:
        #
        #     [batch, channels, time]
        #
        # Expected:
        #
        #     [1, 256, 5000]
        # ----------------------------------------------------

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        # ----------------------------------------------------
        # Grad-CAM weights
        #
        # Average gradients across time
        # ----------------------------------------------------

        weights = gradients.mean(dim=2, keepdim=True)

        # ----------------------------------------------------
        # Weighted combination of feature maps
        # ----------------------------------------------------

        cam = (weights * activations).sum(dim=1)

        # ReLU
        cam = torch.relu(cam)

        # Move to CPU
        cam = cam.squeeze(0).cpu().numpy()

        # ----------------------------------------------------
        # Normalize to 0-1
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (cam - cam_min) / (cam_max - cam_min)

        else:

            cam = np.zeros_like(cam)

        return cam, output.detach()

    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# SMOOTH GRAD-CAM
# ============================================================

def smooth_cam(cam, window=25):

    """
    Smooth Grad-CAM to make important ECG regions easier
    to visualize.

    window=25 corresponds to 50 ms at 500 Hz.
    """

    if window <= 1:
        return cam

    kernel = np.ones(window) / window

    smoothed = np.convolve(
        cam,
        kernel,
        mode="same"
    )

    # Normalize again
    minimum = smoothed.min()
    maximum = smoothed.max()

    if maximum - minimum > 1e-8:

        smoothed = (
            smoothed - minimum
        ) / (
            maximum - minimum
        )

    return smoothed


# ============================================================
# LOAD LABEL NAMES
# ============================================================

def load_class_names():

    encoder_path = os.path.join(
        "processed",
        "label_encoder.pkl"
    )

    if not os.path.exists(encoder_path):

        raise FileNotFoundError(
            f"Could not find label encoder:\n"
            f"{encoder_path}"
        )

    import joblib

    encoder = joblib.load(encoder_path)

    class_names = list(
        encoder.classes_
    )

    return class_names


# ============================================================
# CREATE ECG + GRAD-CAM FIGURE
# ============================================================

def plot_gradcam(
    signal,
    cam,
    actual_class,
    predicted_class,
    predicted_id,
    probabilities,
    sample_number,
    output_path
):

    """
    signal:
        ECG signal with shape [12, 5000]

    cam:
        Grad-CAM importance with shape [5000]
    """

    # --------------------------------------------------------
    # Convert signal to numpy
    # --------------------------------------------------------

    signal = np.asarray(signal)

    # --------------------------------------------------------
    # Smooth CAM for visualization
    # --------------------------------------------------------

    cam_smooth = smooth_cam(
        cam,
        window=25
    )

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    num_samples = signal.shape[1]

    time_axis = (
        np.arange(num_samples)
        / SAMPLING_RATE
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        12,
        1,
        figsize=(18, 22),
        sharex=True
    )

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        "Grad-CAM Explainability for ECG Arrhythmia Classification",
        fontsize=18,
        fontweight="bold",
        y=0.995
    )

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    confidence = probabilities[predicted_id] * 100

    fig.text(
        0.5,
        0.978,
        f"Actual: {actual_class}    |    "
        f"Predicted: {predicted_class}    |    "
        f"Confidence: {confidence:.2f}%",
        ha="center",
        fontsize=13
)

    # --------------------------------------------------------
    # Plot all 12 ECG leads
    # --------------------------------------------------------

    for lead in range(12):

        ax = axes[lead]

        ecg = signal[lead]

        # ECG waveform
        ax.plot(
            time_axis,
            ecg,
            linewidth=0.7
        )

        # ----------------------------------------------------
        # Grad-CAM importance as background
        #
        # Important regions are shown using transparency.
        # ----------------------------------------------------

        ax.imshow(
            cam_smooth[np.newaxis, :],
            aspect="auto",
            extent=[
                time_axis[0],
                time_axis[-1],
                ecg.min(),
                ecg.max()
            ],
            origin="lower",
            alpha=0.30,
            cmap="jet"
        )

        ax.set_ylabel(
            f"Lead {lead + 1}",
            fontsize=9
        )

        ax.grid(
            True,
            alpha=0.20
        )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    axes[-1].set_xlabel(
        "Time (seconds)",
        fontsize=11
    )

    # --------------------------------------------------------
    # Explanation legend
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "Higher Grad-CAM intensity indicates temporal regions "
        "that contributed more strongly to the predicted class.",
        ha="center",
        fontsize=11
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0.035,
            1,
            0.965
        ]
    )

    plt.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("XAI - GRAD-CAM ECG EXPLAINABILITY")
    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    print("\nLoading dataset...")

    (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        num_classes
    ) = load_data()

    print(
        f"Number of classes: {num_classes}"
    )

    # ========================================================
    # LOAD CLASS NAMES
    # ========================================================

    class_names = load_class_names()

    print("\nClass order:")

    for i, name in enumerate(class_names):

        print(
            f"{i}: {name}"
        )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    print("\nBuilding model...")

    (
        model,
        criterion,
        optimizer,
        scheduler,
        scaler
    ) = build_model(
        num_classes,
        class_weights
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\nCould not find checkpoint:\n"
            f"{CHECKPOINT_PATH}\n\n"
            f"Make sure xai_gradcam.py is in the same "
            f"folder as best_inception.pth."
        )

    print(
        f"\nLoading checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support checkpoint containing model_state_dict
    # --------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "\nBest checkpoint loaded successfully."
    )

    # ========================================================
    # SELECT TARGET LAYER
    # ========================================================

    target_layer = model.res2.block3

    print(
        "\nGrad-CAM target layer:"
    )

    print(
        "model.res2.block3"
    )

    print(
        "\nTarget layer contains the final "
        "Inception feature representation before "
        "global average pooling."
    )

    # ========================================================
    # CREATE GRAD-CAM
    # ========================================================

    gradcam = GradCAM(
        model,
        target_layer
    )

    # ========================================================
    # FIND ONE GOOD EXAMPLE PER CLASS
    # ========================================================

    selected_examples = {}

    # Store fallback examples
    fallback_examples = {}

    total_test_samples = 0

    print(
        "\nSearching test set for examples..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We don't need gradients during this search.
    # --------------------------------------------------------

    for signals, targets in test_loader:

        signals = signals.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        with torch.no_grad():

            outputs = model(
                signals
            )

            predictions = outputs.argmax(
                dim=1
            )

        for i in range(
            len(targets)
        ):

            actual = int(
                targets[i].item()
            )

            predicted = int(
                predictions[i].item()
            )

            total_test_samples += 1

            # ------------------------------------------------
            # Save first example as fallback
            # ------------------------------------------------

            if actual not in fallback_examples:

                fallback_examples[actual] = (
                    signals[i]
                    .detach()
                    .cpu(),
                    actual,
                    predicted
                )

            # ------------------------------------------------
            # Prefer correctly classified examples
            # ------------------------------------------------

            if (
                actual == predicted
                and actual not in selected_examples
            ):

                selected_examples[actual] = (
                    signals[i]
                    .detach()
                    .cpu(),
                    actual,
                    predicted
                )

        # ----------------------------------------------------
        # Stop once every class has a correct example
        # ----------------------------------------------------

        if len(selected_examples) == num_classes:

            break

    # ========================================================
    # FILL MISSING CLASSES WITH FALLBACKS
    # ========================================================

    for class_id in range(num_classes):

        if class_id not in selected_examples:

            if class_id in fallback_examples:

                selected_examples[class_id] = (
                    fallback_examples[class_id]
                )

    # ========================================================
    # REPORT SELECTION
    # ========================================================

    print(
        f"\nExamples selected: "
        f"{len(selected_examples)}/{num_classes}"
    )

    # ========================================================
    # GENERATE GRAD-CAM FOR EACH CLASS
    # ========================================================

    for class_id in range(num_classes):

        if class_id not in selected_examples:

            print(
                f"\nWARNING: No test example found "
                f"for class {class_id}"
            )

            continue

        (
            signal_cpu,
            actual_id,
            initial_prediction
        ) = selected_examples[class_id]

        # ----------------------------------------------------
        # Move signal to GPU
        # ----------------------------------------------------

        signal = signal_cpu.unsqueeze(0).to(
            DEVICE
        )

        # ----------------------------------------------------
        # First prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                signal
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predicted_id = int(
                logits.argmax(
                    dim=1
                ).item()
            )

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        cam, output = gradcam.generate(
            signal,
            predicted_id
        )

        # ----------------------------------------------------
        # Get probabilities
        # ----------------------------------------------------

        probabilities = torch.softmax(
            output,
            dim=1
        )[0].cpu().numpy()

        # ----------------------------------------------------
        # Class names
        # ----------------------------------------------------

        actual_name = class_names[
            actual_id
        ]

        predicted_name = class_names[
            predicted_id
        ]

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)

        print(
            f"Example {class_id + 1}"
        )

        print(
            f"Actual    : {actual_name}"
        )

        print(
            f"Predicted : {predicted_name}"
        )

        print(
            f"Confidence: "
            f"{probabilities[predicted_id] * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Print all class probabilities
        # ----------------------------------------------------

        print("\nClass probabilities:")

        for idx, probability in enumerate(
            probabilities
        ):

            print(
                f"  {class_names[idx]:35s} "
                f"{probability * 100:6.2f}%"
            )

        # ----------------------------------------------------
        # Highest importance region
        # ----------------------------------------------------

        peak_index = int(
            np.argmax(cam)
        )

        peak_time = (
            peak_index
            / SAMPLING_RATE
        )

        print(
            f"\nPeak Grad-CAM location: "
            f"{peak_time:.3f} seconds"
        )

        # ----------------------------------------------------
        # Output filename
        # ----------------------------------------------------

        safe_name = (
            predicted_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"gradcam_{safe_name}.png"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_gradcam(
    signal_cpu.numpy(),
    cam,
    actual_name,
    predicted_name,
    predicted_id,
    probabilities,
    class_id + 1,
    output_path
)

    # ========================================================
    # CLEAN UP
    # ========================================================

    gradcam.remove_hooks()

    print("\n")
    print("=" * 70)
    print("GRAD-CAM COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved in:"
    )

    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(OUTPUT_DIR)
    ):

        print(
            f"  {filename}"
        )

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()