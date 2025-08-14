import numpy as np
import json
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
        if activation in ['sigmoid', 'tanh', 'softmax']:
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

    def one_hot_encoder(self, y, num_classes):
        y = y.astype(int).flatten()
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def neuron_activation(self, x, activation):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        elif activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
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

        a = X
        for w, b, act_func in zip(self.weights, self.biases, self.activation_funcs):
            z = np.dot(a, w) + b
            a = self.neuron_activation(z, act_func)
            self.z_s.append(z)
            self.a_s.append(a)
        return a

    def backpropagation(self, y, learning_rate):
        # y = y.reshape(-1, 1)
        m = y.shape[0]
        output = self.a_s[-1]

        # Chain Rule Computation with derivatives
        delta = output - y
        for i in reversed(range(len(self.layers))):
            dw = np.dot(self.a_s[i].T, delta) / m
            db = np.sum(delta, axis=0) / m
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                if self.activation_funcs[i - 1] != 'softmax':
                    delta *= self.activation_derivative(self.z_s[i - 1], self.activation_funcs[i - 1])

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, learning_rate=0.001):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        # One-hot encode manually
        y_train = self.one_hot_encoder(y_train, 2)
        if y_val is not None:
            y_val = self.one_hot_encoder(y_val, 2)

        # Add output layer containing 2 neurons
        self.add_layer(2, activation='softmax')

        for epoch in range(epochs):
            # Full batch gradient descent
            self.feedforward(X_train)
            self.backpropagation(y_train, learning_rate)

            # Training metrics
            output_train = self.feedforward(X_train)
            train_loss = -np.mean(np.sum(y_train * np.log(output_train + 1e-15), axis=1))
            train_losses.append(train_loss)

            train_preds = np.argmax(output_train, axis=1)
            y_true = np.argmax(y_train, axis=1)
            train_accuracy = np.mean(train_preds == y_true)
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
        model_data = {
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": [a for a in self.activation_funcs]
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f)


# Utility functions :

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