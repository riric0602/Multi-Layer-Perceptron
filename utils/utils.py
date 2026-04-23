from matplotlib import pyplot as plt
import numpy as np
import json
from pandas import DataFrame
from utils.metrics import one_hot_encoder, confusion_matrix, f1, precision, recall, mse, classification_metrics


class COLOR:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"


def c(text, color):
    return f"{color}{text}{COLOR.RESET}"


def close_on_key(event) -> None:
    """
    Close the window when ESC key is pressed.
    """
    if event.key == 'escape':
        plt.close(event.canvas.figure)


def get_top_correlations(df: DataFrame, n=10):
    """
    Get top correlated features with Diagnosis
    """
    correlations = (
        df
        .corr(numeric_only=True)['Diagnosis_num']
        .drop('Diagnosis_num')
    )

    return correlations.reindex(
        correlations.abs().sort_values(ascending=False).head(n).index
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
    activations = model_data["activations"]
    train_losses = model_data["train_loss_history"]
    val_losses = model_data["val_loss_history"]
    train_accuracies = model_data["train_accuracy_history"]
    val_accuracies = model_data["val_accuracy_history"]
    scaler_mean = np.array(model_data["scaler_mean"])
    scaler_std = np.array(model_data["scaler_std"])

    return weights, biases, activations, scaler_mean, scaler_std, train_losses, val_losses, train_accuracies, val_accuracies


def pct(x):
    """
    Percentage function
    """
    return f"{x * 100:.5f}%"


def print_metrics_block(title, metrics, color):
    """
    Print bonus metrics with colors
    """
    print(f"\n{c(f'=== {title} ===', color)}")
    print(f"{c('Accuracy', COLOR.GREEN)}:  {pct(metrics['accuracy'])}")
    print(f"{c('Precision', COLOR.CYAN)}: {pct(metrics['precision'])}")
    print(f"{c('Recall', COLOR.CYAN)}:    {pct(metrics['recall'])}")
    print(f"{c('F1', COLOR.CYAN)}:        {pct(metrics['f1'])}")
    print(f"{c('Loss', COLOR.YELLOW)}:      {metrics['loss']:.6f}")


def print_confusion(title, tp, tn, fp, fn, color):
    """
    Print confusion Matrix
    """
    print(f"\n{c(f'=== {title} Confusion Matrix ===', color)}")
    print(f"{c('TP', COLOR.GREEN)}: {tp} | {c('FP', COLOR.RED)}: {fp}")
    print(f"{c('FN', COLOR.RED)}: {fn} | {c('TN', COLOR.GREEN)}: {tn}")


def display_bonus_metrics(model, X_train, y_train, X_val, y_val):
    """
    Display the bonus metrics of the model.
    """
    y_train_oh = one_hot_encoder(y_train, 2)
    y_val_oh = one_hot_encoder(y_val, 2)

    train_metrics = classification_metrics(model, X_train, y_train)
    val_metrics = classification_metrics(model, X_val, y_val)

    # Confusion matrix
    tp, tn, fp, fn = train_metrics["tp"], train_metrics["tn"], train_metrics["fp"], train_metrics["fn"]
    print_confusion("Train", tp, tn, fp, fn, COLOR.BOLD)

    tp, tn, fp, fn = val_metrics["tp"], val_metrics["tn"], val_metrics["fp"], val_metrics["fn"]
    print_confusion("Validation", tp, tn, fp, fn, COLOR.BOLD)

    # Metrics
    print_metrics_block("Train Metrics", train_metrics, COLOR.BOLD)
    print_metrics_block("Validation Metrics", val_metrics, COLOR.BOLD)

    # MSE
    train_mse = mse(model, X_train, y_train_oh)
    val_mse = mse(model, X_val, y_val_oh)

    print(f"\n{c('=== MSE ===', COLOR.BOLD)}")
    print(f"{c('Train MSE', COLOR.BLUE)}: {train_mse:.6f}")
    print(f"{c('Val MSE', COLOR.MAGENTA)}:   {val_mse:.6f}")


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

    print(f"\n{c('=== TRAINING HISTORY ===', COLOR.BOLD)}")
    print(f"{c('Min Loss:', COLOR.YELLOW)} {min_train_loss:.4f} {c('(epoch ' + str(min_train_loss_epoch) + ')', COLOR.DIM)}")
    print(f"{c('Max Accuracy:', COLOR.GREEN)} {pct(max_train_acc)} {c('(epoch ' + str(max_train_acc_epoch) + ')', COLOR.DIM)}")

    print(f"\n{c('=== VALIDATION HISTORY ===', COLOR.BOLD)}")
    print(f"{c('Min Loss:', COLOR.YELLOW)} {min_val_loss:.4f} {c('(epoch ' + str(min_val_loss_epoch) + ')', COLOR.DIM)}")
    print(f"{c('Max Accuracy:', COLOR.GREEN)} {pct(max_val_acc)} {c('(epoch ' + str(max_val_acc_epoch) + ')', COLOR.DIM)}")
    print(f"{c('Max Precision:', COLOR.CYAN)} {pct(max_val_precision)} {c('(epoch ' + str(max_val_precision_epoch) + ')', COLOR.DIM)}")
    print(f"{c('Max Recall:', COLOR.CYAN)} {pct(max_val_recall)} {c('(epoch ' + str(max_val_recall_epoch) + ')', COLOR.DIM)}")
    print(f"{c('Max F1:', COLOR.CYAN)} {pct(max_val_f1)} {c('(epoch ' + str(max_val_f1_epoch) + ')', COLOR.DIM)}")
