import numpy as np

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

        bias = np.zeros((1, num_neurons))
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
        m = y.shape[0]
        output = self.a_s[-1]
        delta = (output - y)

        for i in reversed(range(len(self.layers))):
            delta *= self.activation_derivative(self.z_s[i], self.activation_funcs[i])
            dw = np.dot(self.a_s[i], delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Update weights and biases
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            # Compute delta for previous layer (if not at input layer)
            if i > 0:
                delta = delta @ self.weights[i].T

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.feedforward(X_train)
            self.backpropagation(y_train, learning_rate)

            if epoch % 100 == 0:
                train_loss = np.mean((self.a_s[-1] - y_train) ** 2)
                msg = f"Epoch {epoch}, Train Loss: {train_loss:.4f}"

                if X_val is not None and y_val is not None:
                    val_output = self.feedforward(X_val)
                    val_loss = np.mean((val_output - y_val) ** 2)
                    msg += f", Val Loss: {val_loss:.4f}"

                print(msg)
