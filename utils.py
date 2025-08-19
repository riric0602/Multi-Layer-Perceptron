from matplotlib import pyplot as plt
import numpy as np
import json


def one_hot_encoder(y, num_classes):
    y = y.astype(int).flatten()
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def log_loss(y_true, y_pred):
    epsilon = 1e-15
    p = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))


def close_on_key(event) -> None:
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    fig = plt.figure(figsize=(14, 5))
    fig.canvas.mpl_connect('key_press_event', close_on_key)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    if val_accuracies:
        plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution')
    plt.legend()
    plt.show()


def plot_confusion_matrix(tn, fp, fn, tp, title="Confusion Matrix"):
    cm = np.array([[tn, fp],
                   [fn, tp]])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")

    # Labels
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0", "1"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["0", "1"])
    ax.set_title(title)

    # Write numbers in cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.colorbar(im)
    plt.show()


def load_model(filepath):
    with open(filepath, "r") as f:
        model_data = json.load(f)

    weights = [np.array(w) for w in model_data["weights"]]
    biases = [np.array(b) for b in model_data["biases"]]
    activations = [a for a in model_data["activations"]]
    train_losses = [a for a in model_data["train_loss_history"]]
    val_losses = [a for a in model_data["val_loss_history"]]
    train_accuracies = [a for a in model_data["train_accuracy_history"]]
    val_accuracies = [a for a in model_data["val_accuracy_history"]]

    return weights, biases, activations, train_losses, val_losses, train_accuracies, val_accuracies

