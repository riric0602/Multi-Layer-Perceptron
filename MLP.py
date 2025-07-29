import numpy as np

class MLP:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []        # list of output sizes (neurons per layer)
        self.activations = []   # activation function names
        self.weights = []       # list of weight matrices
        self.biases = []        # list of bias vectors
        self.z_s = []  # pre-activation values (linear function)
        self.activations = []  # activations (starts with input)


    def add_layer(self, num_neurons, activation='sigmoid'):
        input_dim = self.input_size if not self.layers else self.layers[-1]
        self.layers.append(num_neurons)
        self.activations.append(activation)

        if activation in ['sigmoid', 'tanh']:
            # Xavier initialization
            limit = np.sqrt(1. / input_dim)
            weight = np.random.uniform(-limit, limit, (input_dim, num_neurons))
        elif activation == 'relu':
            # He initialization
            weight = np.random.randn(input_dim, num_neurons) * np.sqrt(2. / input_dim)
        else:
            raise ValueError("Unsupported activation. You must choose between ['sigmoid', 'tanh', 'relu']")

        bias = np.zeros((1, num_neurons))

        self.weights.append(weight)
        self.biases.append(bias)


    def neuron_activation(self, x, activation='sigmoid'):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return np.tanh(x)
        elif activation == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation. You must choose between ['sigmoid', 'tanh', 'relu']")


    def feedforward(self, X):
        self.z_s = []
        self.activations = [X]

        a = X
        for w, b, act in zip(self.weights, self.biases, self.activations):
            z = a @ w + b
            a = self.neuron_activation(z, act)
            self.z_s.append(z)
            self.activations.append(a)
        return a


    def activation_derivative(self, z, activation):
        if activation == 'sigmoid':
            sig = self.neuron_activation(z, 'sigmoid')
            return sig * (1 - sig)
        elif activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif activation == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unsupported activation: {activation}")


    def backpropagation(self, X, y, lr):
        m = y.shape[0]
        output = self.activations[-1]
        delta = (output - y)  # dL/dA for MSE

        for i in reversed(range(len(self.layers))):
            z = self.z_s[i]
            a_prev = self.activations[i]
            act = self.activations[i]

            delta *= self.activation_derivative(z, act)

            dw = a_prev.T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

            delta = delta @ self.weights[i].T


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            self.feedforward(X_train)
            self.backpropagation(X_train, y_train, lr)

            if epoch % 100 == 0:
                train_loss = np.mean((self.activations[-1] - y_train) ** 2)
                msg = f"Epoch {epoch}, Train Loss: {train_loss:.4f}"

                if X_val is not None and y_val is not None:
                    val_output = self.feedforward(X_val)
                    val_loss = np.mean((val_output - y_val) ** 2)
                    msg += f", Val Loss: {val_loss:.4f}"

                print(msg)
