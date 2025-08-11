import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []              # list of neuron counts per layer
        self.activation_funcs = []    # activation function names per layer
        self.weights = []              # weight matrices
        self.biases = []               # bias vectors

        # Will be filled during feedforward
        self.z_s = []  # pre-activation values
        self.a_s = []  # activation outputs

    def add_layer(self, num_neurons, activation='sigmoid'):
        input_dim = self.input_size if not self.layers else self.layers[-1]
        self.layers.append(num_neurons)
        self.activation_funcs.append(activation)

        # Weight and bias initialization
        if activation in ['sigmoid', 'tanh']:
            # Xavier initialization
            limit = np.sqrt(1. / input_dim)
            weight = np.random.uniform(-limit, limit, (input_dim, num_neurons))
        elif activation == 'relu':
            # He initialization
            weight = np.random.randn(input_dim, num_neurons) * np.sqrt(2. / input_dim)
        else:
            raise ValueError("Unsupported activation: choose from ['sigmoid','tanh','relu']")

        bias = np.zeros(num_neurons)
        self.weights.append(weight)
        self.biases.append(bias)

    def neuron_activation(self, x, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
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
        else:
            raise ValueError("Unsupported activation.")

    def feedforward(self, X):
        self.z_s = []
        self.a_s = [X]

        a = X
        for w, b, act_func in zip(self.weights, self.biases, self.activation_funcs):
            z = np.dot(a, w) + b
            a = self.neuron_activation(z, act_func)
            self.z_s.append(z)
            self.a_s.append(a)
        return a

    def backpropagation(self, y, learning_rate):
        y = y.reshape(-1, 1)
        m = y.shape[0]
        output = self.a_s[-1]

        # Chain Rule Computation with derivatives
        delta = (output - y)
        for i in reversed(range(len(self.layers))):
            delta *= self.activation_derivative(self.z_s[i], self.activation_funcs[i])
            dw = np.dot(self.a_s[i].T, delta) / m
            db = np.sum(delta, axis=0) / m

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            # Compute delta for previous layer (if not at input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, learning_rate=0.001):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        y_train = y_train.reshape(-1, 1)
        if y_val is not None:
            y_val = y_val.reshape(-1, 1)

        for epoch in range(epochs):
            # Full batch gradient descent
            self.feedforward(X_train)
            self.backpropagation(y_train, learning_rate)

            # Training metrics
            output_train = self.feedforward(X_train)
            train_loss = np.mean((output_train - y_train) ** 2)
            train_losses.append(train_loss)

            train_preds = (output_train >= 0.5).astype(int)
            train_accuracy = np.mean(train_preds == y_train)
            train_accuracies.append(train_accuracy)

            if X_val is not None and y_val is not None:
                output_val = self.feedforward(X_val)
                val_loss = np.mean((output_val - y_val) ** 2)
                val_losses.append(val_loss)

                val_preds = (output_val >= 0.5).astype(int)
                val_accuracy = np.mean(val_preds == y_val)
                val_accuracies.append(val_accuracy)

        plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies)

    def save_model(self, filepath):
        np.savez(
            filepath,
            **{f'weight{i}': w for i, w in enumerate(self.weights)},
            **{f'bias{i}': b for i, b in enumerate(self.biases)}
        )


# Utility functions :

def plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(14, 5))

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