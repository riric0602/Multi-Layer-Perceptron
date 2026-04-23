import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import load_model, close_on_key
import matplotlib.pyplot as plt


def plot_compare(history1, history2, label1="Model 1", label2="Model 2"):
    """
    PLot loss and accuracy graphs of two different models
    .
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.canvas.mpl_connect('key_press_event', close_on_key)

    # Model 1 Loss
    axes[0, 0].plot(history1['loss'], label='Train Loss')
    axes[0, 0].plot(history1['val_loss'], label='Val Loss')
    axes[0, 0].set_title(f"{label1} Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Model 1 Accuracy
    axes[0, 1].plot(history1['accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history1['val_accuracy'], label='Val Accuracy')
    axes[0, 1].set_title(f"{label1} Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Model 2 Loss
    axes[1, 0].plot(history2['loss'], label='Train Loss')
    axes[1, 0].plot(history2['val_loss'], label='Val Loss')
    axes[1, 0].set_title(f"{label2} Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Model 2 Accuracy
    axes[1, 1].plot(history2['accuracy'], label='Train Accuracy')
    axes[1, 1].plot(history2['val_accuracy'], label='Val Accuracy')
    axes[1, 1].set_title(f"{label2} Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print("Usage: python compare.py model1_path model2_path")
            sys.exit(1)

        model_1_path = f"models/{sys.argv[1]}.json"
        model_2_path = f"models/{sys.argv[2]}.json"

        # Load the 2 models' history to compare
        _, _, _, _, _, train_losses_1, val_losses_1, train_accuracies_1, val_accuracies_1 = load_model(model_1_path)
        _, _, _, _, _, train_losses_2, val_losses_2, train_accuracies_2, val_accuracies_2 = load_model(model_2_path)

        history_1 = {
            "loss": train_losses_1,
            "val_loss": val_losses_1,
            "accuracy": train_accuracies_1,
            "val_accuracy": val_accuracies_1,
        }

        history_2 = {
            "loss": train_losses_2,
            "val_loss": val_losses_2,
            "accuracy": train_accuracies_2,
            "val_accuracy": val_accuracies_2,
        }

        plot_compare(history_1, history_2, label1="Model 1", label2="Model 2")

    except Exception as e:
        print(f"Error: {e}")