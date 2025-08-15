import json
import numpy as np
from utils import one_hot_encoder, metrics, plot_loss_and_accuracy

class MLP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []              # list of neuron counts per layer
        self.activation_funcs = []    # activation function names per layer
        self.weights = []              # weight matrices
        self.biases = []               # bias vectors
        self.z_s = []  # pre-activation values
        self.a_s = []  # activation outputs

    def add_layer(self, num_neurons, activation='sigmoid'):
        # Define input dimension of layer for initialization of weights / biases
        input_dim = self.input_size if not self.layers else self.layers[-1]

        # Weight and bias initialization
        bias = np.zeros(num_neurons)

        if activation in ['sigmoid', 'tanh', 'softmax']:
            # Xavier initialization
            limit = np.sqrt(1. / input_dim)
            weight = np.random.uniform(-limit, limit, (input_dim, num_neurons))
        elif activation == 'relu':
            # He initialization
            weight = np.random.randn(input_dim, num_neurons) * np.sqrt(2. / input_dim)
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

    def activation_derivative(self, z, activation):
        if activation == 'sigmoid':
            sig = self.neuron_activation(z, 'sigmoid')
            return sig * (1 - sig)
        elif activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif activation == 'relu':
            return (z > 0).astype(float)
        elif activation == 'softmax':
            raise ValueError("Softmax derivative is handled implicitly with cross-entropy.")
        else:
            raise ValueError("Unsupported activation.")

    def feedforward(self, X):
        self.z_s = []
        self.a_s = [X]

        for w, b, act_func in zip(self.weights, self.biases, self.activation_funcs):
            z = np.dot(X, w) + b
            X = self.neuron_activation(z, act_func)
            self.z_s.append(z)
            self.a_s.append(X)
        return X

    def backpropagation(self, y, learning_rate):
        m = y.shape[0]
        X = self.a_s[-1]

        # Chain Rule Computation with derivatives
        delta = X - y
        for i in reversed(range(len(self.layers))):
            # Compute Weights / Biases Gradient Descent
            dw = np.dot(self.a_s[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activation_derivative(self.z_s[i - 1],
                                                                                 self.activation_funcs[i - 1])

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, learning_rate=0.001):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        y_train_oh = one_hot_encoder(y_train, 2)

        for epoch in range(epochs):
            self.feedforward(X_train)
            self.backpropagation(y_train_oh, learning_rate)

            # Compute training and validation metrics
            train_loss, train_acc = metrics(self, X_train, y_train)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            val_loss, val_acc = metrics(self, X_val, y_val)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            print(f"Epoch: {epoch + 1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - accuracy: {train_acc:.4f} - val_accuracy: {val_acc:.4f}")

        plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies)

    def save_model(self, filepath):
        model_data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": [a for a in self.activation_funcs]
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)
