from matplotlib import pyplot as plt
import numpy as np
import json
from pandas import DataFrame
from utils.metrics import one_hot_encoder, confusion_matrix, f1, precision, recall, mse, classification_metrics


def close_on_key(event) -> None:
    """
    Close the window when ESC key is pressed.
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def get_top_correlations(df: DataFrame, n=10):
    "Get top correlated features with Diagnosis"
    correlations = (
        df
            .corr(numeric_only=True)['Diagnosis']
            .drop('Diagnosis')
    )

    return correlations.reindex(
        correlations
            .abs()
            .sort_values(ascending=False)
            .head(n)
            .index
    )


def plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    """
    Plot loss and accuracy of the model over epochs.
    """
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
    """
    Plot confusion matrix of the model over epochs.
    """
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
    """
    Load the model from a saved json file.
    """
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


def display_bonus_metrics(model, X_train, y_train, X_val, y_val):
    """
    Display the bonus metrics of the model.
    """
    y_train_oh = one_hot_encoder(y_train, 2)
    y_val_oh = one_hot_encoder(y_val, 2)

    train_metrics = classification_metrics(model, X_train, y_train)
    val_metrics = classification_metrics(model, X_val, y_val)

    #  Train and Validation Confusion matrix
    tp_train, tn_train, fp_train, fn_train = train_metrics["tp"], train_metrics["tn"], train_metrics["fp"], train_metrics["fn"]
    tp_val, tn_val, fp_val, fn_val = val_metrics["tp"], val_metrics["tn"], val_metrics["fp"], val_metrics["fn"]

    print("\nTrain Confusion matrix:")
    print(f"[{tp_train}, {fp_train}\n{fn_train}, {tn_train}]\n")

    print("Validation Confusion matrix:")
    print(f"[{tp_val}, {fp_val}\n{fn_val}, {tn_val}]\n")

    # Compute Precision of model
    train_precision = train_metrics["precision"]
    val_precision = val_metrics["precision"]
    print(f"\nTrain Precision: {train_precision}")
    print(f"Validation Precision: {val_precision}")

    # Compute Recall of model
    train_recall = train_metrics["recall"]
    val_recall = val_metrics["recall"]
    print(f"\nTrain Recall: {train_recall}")
    print(f"Validation Recall: {val_recall}")

    # Compute F1 of model
    train_f1 = train_metrics["f1"]
    val_f1 = val_metrics["f1"]
    print(f"\nTrain F1: {train_f1}")
    print(f"Validation F1: {val_f1}")

    # Compute MSE of model
    train_mse = mse(model, X_train, y_train_oh)
    val_mse = mse(model, X_val, y_val_oh)
    print(f"\nTrain MSE: {train_mse}")
    print(f"Validation MSE: {val_mse}")


def display_history(model):
    """
    Display the history of the metrics of the model.
    """
    # History of training metrics
    min_train_loss = min(model.train_losses)
    min_train_loss_epoch = model.train_losses.index(min_train_loss) + 1
    max_train_acc = max(model.train_accuracies)
    max_train_acc_epoch = model.train_accuracies.index(max_train_acc) + 1

    # History of validation metrics
    min_val_loss = min(model.val_losses)
    min_val_loss_epoch = model.val_losses.index(min_val_loss) + 1
    max_val_acc = max(model.val_accuracies)
    max_val_acc_epoch = model.val_accuracies.index(max_val_acc) + 1
    max_val_f1 = max(model.val_f1_scores)
    max_val_f1_epoch = model.val_f1_scores.index(max_val_f1) + 1
    max_val_precision = max(model.val_precisions)
    max_val_precision_epoch = model.val_precisions.index(max_val_precision) + 1
    max_val_recall = max(model.val_recalls)
    max_val_recall_epoch = model.val_recalls.index(max_val_recall) + 1

    print("\n=== Training Metrics ===")
    print(f"Minimum Train Loss: {min_train_loss:.4f} at epoch {min_train_loss_epoch}")
    print(f"Maximum Train Accuracy: {max_train_acc:.4f} at epoch {max_train_acc_epoch}\n")

    print("\n=== Validation Metrics ===")
    print(f"Minimum Validation Loss: {min_val_loss:.4f} at epoch {min_val_loss_epoch}")
    print(f"Maximum Validation Accuracy: {max_val_acc:.4f} at epoch {max_val_acc_epoch}")
    print(f"Maximum Validation Precision: {max_val_precision:.4f} at epoch {max_val_precision_epoch}")
    print(f"Maximum Validation Recall: {max_val_recall:.4f} at epoch {max_val_recall_epoch}")
    print(f"Maximum Validation F1: {max_val_f1:.4f} at epoch {max_val_f1_epoch}")
