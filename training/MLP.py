import json
import os
import copy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import COLOR, plot_loss_and_accuracy, display_bonus_metrics, display_history
from utils.metrics import *

class MLP:
    def __init__(self, input_size):
        """
        Initialization of the MLP object with input size.
        """
        self.input_size = input_size
        self.layers = []              # list of neuron counts per layer
        self.activation_funcs = []    # activation function names per layer
        self.weights = []             # weight matrices
        self.biases = []              # bias vectors
        self.z_s = []                 # pre-activation values
        self.a_s = []                 # activation outputs
        self.velocities_w = []        # weights velocity
        self.velocities_b = []        # biases velocity
        self.scaler_mean = 0
        self.scaler_std = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        np.random.seed(1)


    def add_layer(self, num_neurons, activation='relu'):
        """
        Add a neural network layer to the MLP object.
        """
        # Define input dimension of layer for initialization of weights / biases
        input_dim = self.input_size if not self.layers else self.layers[-1]

        # Weight and bias initialization
        bias = np.zeros(num_neurons)

        if activation in ['softmax']:
            # Random initialization
            weight = np.random.randn(input_dim, num_neurons)
        elif activation in ['sigmoid']:
            # Xavier initialization
            limit = np.sqrt(6. / (input_dim + num_neurons))
            weight = np.random.uniform(-limit, limit, (input_dim, num_neurons))
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
        """
        Activation function for a neuron in the neural network.
        """
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
        """
        Partial Derivative of an activation function for a neuron.
        """
        if activation == 'sigmoid':
            return a * (1 - a)
        elif activation == 'tanh':
            return 1 - a ** 2
        elif activation == 'relu':
            return (a > 0).astype(float)
        else:
            raise ValueError("Unsupported activation.")


    def feedforward(self, X, weights, biases):
        """
        Feedforward a neural network using weights and biases.
        """
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

        p = self.a_s[-1]
        delta = (p - y)

        for i in reversed(range(num_layers)):
            dw = np.dot(self.a_s[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(
                    self.a_s[i],
                    self.activation_funcs[i - 1]
                )

            # momentum / weight update
            if momentum is not None:
                self.velocities_w[i] = momentum * self.velocities_w[i] + lr * dw
                self.velocities_b[i] = momentum * self.velocities_b[i] + lr * db
                self.weights[i] -= self.velocities_w[i]
                self.biases[i] -= self.velocities_b[i]
            else:
                self.weights[i] -= lr * dw
                self.biases[i] -= lr * db


    def learning_algorithm(self, X_train, y_train_oh, lr, momentum):
        """
        Training logic using feedforward and backpropagation.
        """
        # Nesterov Optimization lookahead parameters
        if momentum is not None:
            weights = [w - momentum * vw for w, vw in zip(self.weights, self.velocities_w)]
            biases = [b - momentum * vb for b, vb in zip(self.biases, self.velocities_b)]
            self.feedforward(X_train, weights, biases)
        else:
            self.feedforward(X_train, self.weights, self.biases)

        self.backpropagation(y_train_oh, lr, momentum)


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, lr=0.001, patience=None, momentum=None, metrics=None, history=None):
        """
        Main training function for the neural network in the MLP object.
        """
        # One hot encode the true results
        if isinstance(y_train[0], str):
            y_train = np.array([1 if y == "M" else 0 for y in y_train])

        y_train_oh = one_hot_encoder(y_train, 2)

        # Early stopping and Nesterov Optimization variables
        min_delta = 1e-6
        patience_counter = 0
        best_f1 = 0
        best_weights = copy.deepcopy(self.weights)
        best_biases = copy.deepcopy(self.biases)
        self.velocities_w = [np.zeros_like(w) for w in self.weights]
        self.velocities_b = [np.zeros_like(b) for b in self.biases]

        for epoch in range(epochs):
            self.learning_algorithm(X_train, y_train_oh, lr, momentum)

            # Compute training and validation metrics
            train_metrics = classification_metrics(self, X_train, y_train)
            self.train_losses.append(train_metrics["loss"])
            self.train_accuracies.append(train_metrics["accuracy"])
            self.train_precisions.append(train_metrics["precision"])
            self.train_recalls.append(train_metrics["recall"])
            self.train_f1_scores.append(train_metrics["f1"])

            val_metrics = classification_metrics(self, X_val, y_val)
            self.val_losses.append(val_metrics["loss"])
            self.val_accuracies.append(val_metrics["accuracy"])
            self.val_precisions.append(val_metrics["precision"])
            self.val_recalls.append(val_metrics["recall"])
            self.val_f1_scores.append(val_metrics["f1"])

            if patience is not None:
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    best_weights = copy.deepcopy(self.weights)
                    best_biases = copy.deepcopy(self.biases)
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.weights = best_weights
                        self.biases = best_biases
                        print(f"Early stopping at {COLOR.BOLD}epoch {epoch + 1}{COLOR.RESET}. Restored best validation-F1 weights.")
                        break

            print(
                f"{COLOR.YELLOW}Epoch {epoch+1}/{epochs}{COLOR.RESET} | "
                f"{COLOR.BOLD}train{COLOR.RESET}: "
                f"{COLOR.BLUE}loss {train_metrics['loss']:.4f}{COLOR.RESET}, "
                f"{COLOR.GREEN}acc {train_metrics['accuracy']:.4f}{COLOR.RESET} | "
                f"{COLOR.BOLD}val{COLOR.RESET}: "
                f"{COLOR.BLUE}loss {val_metrics['loss']:.4f}{COLOR.RESET}, "
                f"{COLOR.GREEN}acc {val_metrics['accuracy']:.4f}{COLOR.RESET}, "
                f"{COLOR.CYAN}f1 {val_metrics['f1']:.4f}{COLOR.RESET}"
            )

        if patience is not None and self.val_losses:
            self.weights = best_weights
            self.biases = best_biases
            print(f"{COLOR.GREEN}Best validation F1 kept from epoch {best_epoch + 1}.{COLOR.RESET}")

        plot_loss_and_accuracy(self.train_losses, self.train_accuracies, self.val_losses, self.val_accuracies)

        if metrics is True:
            display_bonus_metrics(self, X_train, y_train, X_val, y_val)

        if history is True:
            display_history(self)


    def save_model(self, filepath):
        """
        Save the trained model as a json file.
        """
        model_data = {
            "input_size": self.input_size,
            "layers": self.layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": [a for a in self.activation_funcs],
            "scaler_mean": self.scaler_mean.tolist(),
            "scaler_std": self.scaler_std.tolist(),
            "train_loss_history": self.train_losses,
            "val_loss_history": self.val_losses,
            "train_accuracy_history": self.train_accuracies,
            "val_accuracy_history": self.val_accuracies,
            "train_precision_history": self.train_precisions,
            "val_precision_history": self.val_precisions,
            "train_recall_history": self.train_recalls,
            "val_recall_history": self.val_recalls,
            "train_f1_history": self.train_f1_scores,
            "val_f1_history": self.val_f1_scores,
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)
