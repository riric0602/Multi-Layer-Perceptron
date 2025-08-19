import json
import numpy as np
from utils import one_hot_encoder, plot_loss_and_accuracy
from metrics import *

class MLP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []              # list of neuron counts per layer
        self.activation_funcs = []    # activation function names per layer
        self.weights = []              # weight matrices
        self.biases = []               # bias vectors
        self.z_s = []  # pre-activation values
        self.a_s = []  # activation outputs
        self.velocities_w = [] # weights velocity
        self.velocities_b = [] # biases velocity
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        np.random.seed(1)


    def add_layer(self, num_neurons, activation='sigmoid'):
        # Define input dimension of layer for initialization of weights / biases
        input_dim = self.input_size if not self.layers else self.layers[-1]

        # Weight and bias initialization
        bias = np.zeros(num_neurons)

        if activation in ['sigmoid', 'softmax']:
            # Random initialization
            weight = np.random.randn(input_dim, num_neurons)
        elif activation == 'tanh':
            # Xavier Initialization
            limit = np.sqrt(1. / input_dim)
            weight = np.random.uniform(-limit, limit, (input_dim, num_neurons))
        elif activation == 'relu':
            # He initialization
            weight = np.random.randn(input_dim, num_neurons) * np.sqrt(2.0 / input_dim)
        else:
            raise ValueError("Unsupported activation: choose from ['sigmoid','tanh','relu']")

        self.layers.append(num_neurons)
        self.activation_funcs.append(activation)
        self.weights.append(weight)
        self.biases.append(bias)


    def neuron_activation(self, z, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation.")


    def activation_derivative(self, a, activation):
        if activation == 'sigmoid':
            return a * (1 - a)
        elif activation == 'tanh':
            return 1 - a ** 2
        elif activation == 'relu':
            return (a > 0).astype(float)
        elif activation == 'softmax':
            raise ValueError("Softmax derivative is handled implicitly with cross-entropy.")
        else:
            raise ValueError("Unsupported activation.")


    def feedforward(self, X, weights, biases):
        self.z_s = []
        self.a_s = [X]

        a = X
        for w, b, act_func in zip(weights, biases, self.activation_funcs):
            z = np.dot(a, w) + b
            a = self.neuron_activation(z, act_func)
            self.z_s.append(z)
            self.a_s.append(a)
        return a


    def backpropagation(self, y, lr, momentum):
        num_layers = len(self.layers)
        m = y.shape[0]
        X = self.a_s[-1]

        # Chain Rule Computation with derivatives
        delta = X - y
        for i in reversed(range(num_layers)):
            # Compute Weights / Biases Gradient Descent
            dw = np.dot(self.a_s[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.a_s[i], self.activation_funcs[i - 1])

            if momentum is not None:
                # Momentum update
                self.velocities_w[i] = momentum * self.velocities_w[i] + lr * dw
                self.velocities_b[i] = momentum * self.velocities_b[i] + lr * db

                # Update weights and biases
                self.weights[i] -= self.velocities_w[i]
                self.biases[i] -= self.velocities_b[i]
            else:
                # Update weights and biases
                self.weights[i] -= lr * dw
                self.biases[i] -= lr * db


    def learning_algorithm(self, X_train, y_train_oh, lr, momentum):
        # Nesterov Optimization lookahead parameters
        if momentum is not None:
            weights = [w - momentum * vw for w, vw in zip(self.weights, self.velocities_w)]
            biases = [b - momentum * vb for b, vb in zip(self.biases, self.velocities_b)]
            self.feedforward(X_train, weights, biases)
        else:
            self.feedforward(X_train, self.weights, self.biases)

        self.backpropagation(y_train_oh, lr, momentum)


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.001, patience=None, momentum=None, metrics=None):
        # One hot encode the true results
        y_train_oh = one_hot_encoder(y_train, 2)
        y_val_oh = one_hot_encoder(y_val, 2)

        # Early stopping and Nesterov Optimization variables
        min_delta = 0.01
        patience_counter = 0
        best_loss = float('inf')
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]

        for epoch in range(epochs):
            self.learning_algorithm(X_train, y_train_oh, lr, momentum)

            # Compute training and validation metrics
            train_loss, train_acc = loss_and_accuracy(self, X_train, y_train)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            val_loss, val_acc = loss_and_accuracy(self, X_val, y_val)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if patience is not None:
                if val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            print(f"Epoch: {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - accuracy: {train_acc:.4f} - val_accuracy: {val_acc:.4f}")

        plot_loss_and_accuracy(self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies)

        if metrics is True:
            #  Train and Validation Confusion matrix
            tp_train, tn_train, fp_train, fn_train = confusion_matrix(self, X_train, y_train)
            tp_val, tn_val, fp_val, fn_val = confusion_matrix(self, X_val, y_val)

            print("\nTrain Confusion matrix:")
            print(f"[{tp_train}, {tn_train}\n{fp_train}, {fn_train}]\n")

            print("Validation Confusion matrix:")
            print(f"[{tp_val}, {tn_val}\n{fp_val}, {fn_val}]\n")

            # Compute Precision of model
            train_precision = precision(tp_train, fp_train)
            val_precision = precision(tp_val, fp_val)
            print(f"\nTrain Precision: {train_precision}")
            print(f"Validation Precision: {val_precision}")

            # Compute Recall of model
            train_recall = recall(tp_train, fn_train)
            val_recall = recall(tp_val, fn_val)
            print(f"\nTrain Recall: {train_recall}")
            print(f"Validation Recall: {val_recall}")

            # Compute F1 of model
            train_f1 = f1(train_precision, train_recall)
            val_f1 = f1(val_precision, val_recall)
            print(f"\nTrain F1: {train_f1}")
            print(f"Validation F1: {val_f1}")

            # Compute MSE of model
            train_mse = mse(self, X_train, y_train_oh)
            val_mse = mse(self, X_val, y_val_oh)
            print(f"\nTrain MSE: {train_mse}")
            print(f"Validation MSE: {val_mse}")


    def save_model(self, filepath):
        model_data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": [a for a in self.activation_funcs],
            "train_loss_history": self.train_losses,
            "val_loss_history": self.val_losses,
            "train_accuracy_history": self.train_accuracies,
            "val_accuracy_history": self.val_accuracies,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)
