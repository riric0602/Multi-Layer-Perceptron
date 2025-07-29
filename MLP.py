import numpy as np

class MLP:
    def __init__(self, input_size, output_size):
        self.layers = []
        self.activations = []
        self.weights = []
        self.biases = []
        self.input_size = input_size
        self.output_size = output_size

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
            raise ValueError("Unknown activation function. You must choose between ['sigmoid', 'tanh', 'relu']")

        bias = np.zeros((1, num_neurons))

        self.weights.append(weight)
        self.biases.append(bias)
