from matplotlib import pyplot as plt
import numpy as np


def one_hot_encoder(y, num_classes):
    y = y.astype(int).flatten()
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def log_loss(y_true, y_pred):
    epsilon = 1e-15
    p = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))


def metrics(model, X, y_true):
    y_true_oh = one_hot_encoder(y_true, 2)
    y_pred = model.feedforward(X)
    loss = log_loss(y_true_oh, y_pred)

    train_preds = np.argmax(y_pred, axis=1)
    acc = np.mean(train_preds == y_true)
    return loss, acc


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
