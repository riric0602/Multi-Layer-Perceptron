import numpy as np


def one_hot_encoder(y, num_classes):
    """
    One hot encode the labels using a one-hot vector.
    """
    y = y.astype(int).flatten()
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def log_loss(y_true, y_pred):
    """
    Compute the log loss for a binary classification problem.
    """
    epsilon = 1e-15
    p = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))


def loss_and_accuracy(model, X, y_true):
    """
    Compute the loss and accuracy of the model.
    """
    y_true_oh = one_hot_encoder(y_true, 2)
    y_pred = model.feedforward(X, model.weights, model.biases)
    loss = log_loss(y_true_oh, y_pred)

    preds = np.argmax(y_pred, axis=1)
    acc = np.mean(preds == y_true)
    return loss, acc


def confusion_matrix(model, X, y_true):
    """
    Compute the confusion matrix of the model.
    """
    y_pred = model.feedforward(X, model.weights, model.biases)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.astype(int).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return int(tp), int(tn), int(fp), int(fn)


def precision(tp, fp):
    """
    Compute the precision of the model.
    """
    return tp / (tp + fp)


def recall(tp, fn):
    """
    Compute the recall of the model.
    """
    return tp / (tp + fn)


def f1(precision, recall):
    """
    Compute the F1 score of the model.
    """
    return 2 * precision * recall / (precision + recall)


def mse(model, X, y_true):
    """
    Compute the mean squared error (MSE) of the model.
    """
    y_pred = model.feedforward(X, model.weights, model.biases)
    return np.mean((y_true - y_pred)**2)

