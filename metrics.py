from utils import one_hot_encoder, log_loss
import numpy as np


def loss_and_accuracy(model, X, y_true):
    y_true_oh = one_hot_encoder(y_true, 2)
    y_pred = model.feedforward(X, model.weights, model.biases)
    loss = log_loss(y_true_oh, y_pred)

    preds = np.argmax(y_pred, axis=1)
    acc = np.mean(preds == y_true)
    return loss, acc


def confusion_matrix(model, X, y_true):
    y_pred = model.feedforward(X, model.weights, model.biases)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = y_true.astype(int).ravel()

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return int(tp), int(tn), int(fp), int(fn)


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall)


def mse(model, X, y_true):
    y_pred = model.feedforward(X, model.weights, model.biases)
    return np.mean((y_true - y_pred)**2)

