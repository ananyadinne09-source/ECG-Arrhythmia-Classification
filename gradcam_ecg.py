"""
gradcam_ecg.py
Computes Grad-CAM heatmaps for the trained 1D InceptionTime model, showing
which time regions of the ECG signal most influenced the prediction.
Run independently: python gradcam_ecg.py
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train_inception import load_data, build_model, DEVICE

CHECKPOINT_PATH = "best_inception.pth"
NUM_SAMPLES_TO_PLOT = 4
LEAD_TO_PLOT = 1  # Lead II (index 1) -- classic ECG rhythm strip lead


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


class GradCAM1D:
    """
    Grad-CAM for 1D CNNs. Hooks into a target conv-based layer, captures
    activations (forward) and gradients (backward), and computes a
    class-discriminative importance map over the time axis.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        input_tensor: shape (1, 12, 5000)
        Returns: 1D heatmap of length matching the input's time dimension,
        normalized to [0, 1], plus the predicted class index.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # activations/gradients shape: (1, channels, time)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, channels, 1)
        cam = (weights * self.activations).sum(dim=1)       # (1, time)
        cam = F.relu(cam)

        cam = cam.squeeze(0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample CAM (from the conv layer's time resolution) to input length
        cam_upsampled = np.interp(
            np.linspace(0, len(cam) - 1, input_tensor.shape[2]),
            np.arange(len(cam)),
            cam
        )

        return cam_upsampled, class_idx


def plot_gradcam(signal, cam, lead_idx, true_name, pred_name, save_path):
    """Overlays the Grad-CAM heatmap on top of one ECG lead's waveform."""
    fig, ax = plt.subplots(figsize=(14, 4))

    lead_signal = signal[lead_idx]
    time_axis = np.arange(len(lead_signal))

    ax.plot(time_axis, lead_signal, color="black", linewidth=0.8, label=f"Lead {lead_idx + 1}")

    ax.imshow(
        cam[np.newaxis, :],
        cmap="jet",
        aspect="auto",
        alpha=0.4,
        extent=[0, len(lead_signal), lead_signal.min(), lead_signal.max()]
    )

    ax.set_title(f"Grad-CAM | True: {true_name} | Predicted: {pred_name}")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Normalized Amplitude")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    print("Loading data and model...")
    train_loader, val_loader, test_loader, class_weights, num_classes = load_data()

    model, criterion, optimizer, scheduler, scaler = build_model(num_classes, class_weights)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = load_class_names()

    # Target layer: final residual Inception stage, right before global pooling.
    # This is the last point where time-axis resolution is preserved.
    target_layer = model.res2

    gradcam = GradCAM1D(model, target_layer)

    test_iter = iter(test_loader)
    signals, labels = next(test_iter)

    for i in range(min(NUM_SAMPLES_TO_PLOT, signals.shape[0])):
        single_signal = signals[i:i+1].to(DEVICE)
        true_idx = labels[i].item()

        cam, pred_idx = gradcam.generate(single_signal)

        signal_np = signals[i].numpy()  # (12, 5000)

        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]

        plot_gradcam(
            signal_np,
            cam,
            LEAD_TO_PLOT,
            true_name,
            pred_name,
            save_path=f"gradcam_sample_{i}.png"
        )


if __name__ == "__main__":
    main()