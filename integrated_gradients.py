# ============================================================
# XAI - INTEGRATED GRADIENTS FOR INCEPTIONTIME ECG CLASSIFIER
# ============================================================
#
# Uses:
#   - Existing InceptionTime model from train_inception.py
#   - Existing best_inception.pth checkpoint
#   - Existing Chapman-Shaoxing processed test data
#   - Captum Integrated Gradients
#
# Input:
#   ECG signal shape = [12, 5000]
#   12 leads
#   500 Hz
#   10 seconds
#
# Output:
#   xai_results/
#       integrated_gradients_Atrial_Fibrillation.png
#       integrated_gradients_Sinus_Bradycardia.png
#       integrated_gradients_Sinus_Rhythm.png
#       integrated_gradients_Sinus_Tachycardia.png
#       integrated_gradients_Supraventricular_Tachycardia.png
#
# ============================================================


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

from captum.attr import IntegratedGradients

# Import the EXACT existing model/data setup
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

SAMPLING_RATE = 500

DPI = 300

# Number of steps used by Integrated Gradients.
#
# More steps = more accurate approximation
# but slower computation.
#
# 50 is a reasonable starting point.
IG_STEPS = 10


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CLASS NAMES
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

    encoder = joblib.load(
        encoder_path
    )

    return list(
        encoder.classes_
    )


# ============================================================
# INTEGRATED GRADIENTS
# ============================================================

def generate_integrated_gradients(
    model,
    signal,
    target_class
):

    """
    Generate Integrated Gradients attribution.

    Parameters
    ----------
    model:
        Trained InceptionTime model.

    signal:
        ECG tensor with shape [1, 12, 5000].

    target_class:
        Integer class ID being explained.

    Returns
    -------
    attributions:
        Tensor with shape [1, 12, 5000].
    """

    # --------------------------------------------------------
    # Create Integrated Gradients object
    # --------------------------------------------------------

    ig = IntegratedGradients(
        model
    )

    # --------------------------------------------------------
    # Baseline
    #
    # Because the ECG signals are z-score normalized,
    # zero represents the average normalized signal.
    #
    # Therefore a zero baseline is appropriate here.
    # --------------------------------------------------------

    baseline = torch.zeros_like(
        signal
    )

    # --------------------------------------------------------
    # Calculate attribution
    # --------------------------------------------------------

    attributions = ig.attribute(
        signal,
        baselines=baseline,
        target=target_class,
        n_steps=IG_STEPS,
        method="gausslegendre"
    )

    return attributions


# ============================================================
# NORMALIZE ATTRIBUTION
# ============================================================

def normalize_attribution(
    attribution
):

    """
    Normalize attribution values to [-1, 1].

    Positive values:
        support the predicted class.

    Negative values:
        oppose the predicted class.
    """

    maximum = np.max(
        np.abs(attribution)
    )

    if maximum > 1e-12:

        attribution = (
            attribution / maximum
        )

    return attribution


# ============================================================
# SMOOTH ATTRIBUTION
# ============================================================

def smooth_attribution(
    attribution,
    window=25
):

    """
    Smooth temporal attribution.

    25 samples at 500 Hz = 50 ms.
    """

    if window <= 1:

        return attribution

    kernel = (
        np.ones(window)
        / window
    )

    smoothed = np.convolve(
        attribution,
        kernel,
        mode="same"
    )

    return smoothed


# ============================================================
# CREATE ECG + IG FIGURE
# ============================================================

def plot_integrated_gradients(
    signal,
    attributions,
    actual_class,
    predicted_class,
    predicted_id,
    probabilities,
    sample_number,
    output_path
):

    """
    Plot all 12 ECG leads with Integrated Gradients.

    signal:
        Shape [12, 5000]

    attributions:
        Shape [12, 5000]
    """

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    signal = np.asarray(
        signal
    )

    attributions = np.asarray(
        attributions
    )

    # --------------------------------------------------------
    # Temporal attribution
    #
    # We combine attribution across the 12 leads.
    #
    # Mean absolute attribution tells us:
    # "Which time regions were important overall?"
    # --------------------------------------------------------

    temporal_attribution = np.mean(
        np.abs(attributions),
        axis=0
    )

    # --------------------------------------------------------
    # Smooth temporal attribution
    # --------------------------------------------------------

    temporal_attribution = smooth_attribution(
        temporal_attribution,
        window=25
    )

    # --------------------------------------------------------
    # Normalize temporal attribution
    # --------------------------------------------------------

    temporal_attribution = normalize_attribution(
        temporal_attribution
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
        "Integrated Gradients Explainability for ECG "
        "Arrhythmia Classification",
        fontsize=18,
        fontweight="bold",
        y=0.995
    )

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    confidence = (
        probabilities[predicted_id]
        * 100
    )

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
    # Plot each lead
    # --------------------------------------------------------

    for lead in range(12):

        ax = axes[lead]

        ecg = signal[lead]

        # ----------------------------------------------------
        # ECG waveform
        # ----------------------------------------------------

        ax.plot(
            time_axis,
            ecg,
            linewidth=0.7
        )

        # ----------------------------------------------------
        # Temporal importance background
        # ----------------------------------------------------

        ax.imshow(
            temporal_attribution[
                np.newaxis,
                :
            ],
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
    # Explanation
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.015,
        "Higher intensity indicates temporal regions with "
        "greater overall Integrated Gradients attribution "
        "toward the predicted class.",
        ha="center",
        fontsize=11
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout(
        rect=[
            0,
            0.035,
            1,
            0.965
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
    print("XAI - INTEGRATED GRADIENTS ECG EXPLAINABILITY")
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

    for i, name in enumerate(
        class_names
    ):

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
    # LOAD CHECKPOINT
    # ========================================================

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            f"\nCould not find checkpoint:\n"
            f"{CHECKPOINT_PATH}"
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
    # Support both checkpoint formats
    # --------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
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
    # CREATE INTEGRATED GRADIENTS
    # ========================================================

    print(
        "\nIntegrated Gradients configuration:"
    )

    print(
        f"Baseline : zero tensor"
    )

    print(
        f"Steps    : {IG_STEPS}"
    )

    print(
        f"Method   : Gauss-Legendre"
    )

    # ========================================================
    # FIND ONE CORRECT EXAMPLE PER CLASS
    # ========================================================

    selected_examples = {}

    fallback_examples = {}

    print(
        "\nSearching test set for examples..."
    )

    for signals, targets in test_loader:

        signals = signals.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                signals
            )

            predictions = outputs.argmax(
                dim=1
            )

        # ----------------------------------------------------
        # Examine batch
        # ----------------------------------------------------

        for i in range(
            len(targets)
        ):

            actual_id = int(
                targets[i].item()
            )

            predicted_id = int(
                predictions[i].item()
            )

            # ------------------------------------------------
            # Fallback
            # ------------------------------------------------

            if actual_id not in fallback_examples:

                fallback_examples[
                    actual_id
                ] = (
                    signals[i]
                    .detach()
                    .cpu(),

                    actual_id,

                    predicted_id
                )

            # ------------------------------------------------
            # Correct example
            # ------------------------------------------------

            if (
                actual_id == predicted_id
                and actual_id
                not in selected_examples
            ):

                selected_examples[
                    actual_id
                ] = (
                    signals[i]
                    .detach()
                    .cpu(),

                    actual_id,

                    predicted_id
                )

        # ----------------------------------------------------
        # Stop when all classes found
        # ----------------------------------------------------

        if len(
            selected_examples
        ) == num_classes:

            break

    # ========================================================
    # FILL MISSING CLASSES
    # ========================================================

    for class_id in range(
        num_classes
    ):

        if (
            class_id
            not in selected_examples
        ):

            if (
                class_id
                in fallback_examples
            ):

                selected_examples[
                    class_id
                ] = fallback_examples[
                    class_id
                ]

    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"\nExamples selected: "
        f"{len(selected_examples)}/{num_classes}"
    )

    # ========================================================
    # GENERATE IG
    # ========================================================

    for class_id in range(
        num_classes
    ):

        if (
            class_id
            not in selected_examples
        ):

            print(
                f"\nWARNING: No test example "
                f"found for class {class_id}"
            )

            continue

        (
            signal_cpu,
            actual_id,
            initial_prediction
        ) = selected_examples[
            class_id
        ]

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        signal = (
            signal_cpu
            .unsqueeze(0)
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # Prediction
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
        # Generate Integrated Gradients
        # ----------------------------------------------------

        print(
            "\nGenerating Integrated Gradients..."
        )

        attributions = (
            generate_integrated_gradients(
                model,
                signal,
                predicted_id
            )
        )

        # ----------------------------------------------------
        # Convert attribution
        # ----------------------------------------------------

        attributions = (
            attributions
            .detach()
            .cpu()
            .numpy()
            .squeeze(0)
        )

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
        # Probability array
        # ----------------------------------------------------

        probabilities = (
            probabilities[0]
            .cpu()
            .numpy()
        )

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
        # Print class probabilities
        # ----------------------------------------------------

        print(
            "\nClass probabilities:"
        )

        for idx, probability in enumerate(
            probabilities
        ):

            print(
                f"  {class_names[idx]:35s} "
                f"{probability * 100:6.2f}%"
            )

        # ----------------------------------------------------
        # Temporal attribution
        # ----------------------------------------------------

        temporal_attribution = np.mean(
            np.abs(attributions),
            axis=0
        )

        temporal_attribution = smooth_attribution(
            temporal_attribution,
            window=25
        )

        # ----------------------------------------------------
        # Peak importance
        # ----------------------------------------------------

        peak_index = int(
            np.argmax(
                temporal_attribution
            )
        )

        peak_time = (
            peak_index
            / SAMPLING_RATE
        )

        print(
            f"\nPeak Integrated Gradients "
            f"location: {peak_time:.3f} seconds"
        )

        # ----------------------------------------------------
        # Total attribution
        # ----------------------------------------------------

        total_attribution = np.sum(
            np.abs(attributions)
        )

        print(
            f"Total absolute attribution: "
            f"{total_attribution:.6f}"
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
            f"integrated_gradients_{safe_name}.png"
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plot_integrated_gradients(
            signal_cpu.numpy(),
            attributions,
            actual_name,
            predicted_name,
            predicted_id,
            probabilities,
            class_id + 1,
            output_path
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INTEGRATED GRADIENTS COMPLETE")
    print("=" * 70)

    print(
        "\nResults saved in:"
    )

    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print("\nGenerated files:")

    for filename in sorted(
        os.listdir(
            OUTPUT_DIR
        )
    ):

        if filename.startswith(
            "integrated_gradients_"
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