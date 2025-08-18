import numpy as np
from MLP import MLP
import argparse
from separate import preprocess_and_split_data

def parse_parameters():
    parser = argparse.ArgumentParser(
        description="Train a Multi-Layer Perceptron model for binary classification of breast cancer."
    )

    parser.add_argument("-l", '--add_layers', type=int, nargs="+", default=[24, 24], help="Layers' sizes to add.")
    parser.add_argument('-a', '--activations', type=str, nargs="+", help="Learning rate for training.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.05, help="Learning rate for training.")
    parser.add_argument('-e', '--epochs', type=int, default=1000, help="Number of epochs to train.")
    parser.add_argument('-es', '--early_stop', type=int, default=None, help="Number of epochs to stop before overfitting.")

    return parser.parse_args()


def get_parameters(params):
    layers = params.add_layers
    learning_rate = params.learning_rate
    epochs = params.epochs
    activations = [None] * len(layers)
    early_stop = params.early_stop

    if params.activations is not None:
        activations = [act.lower() for act in params.activations]

        if len(activations) != len(layers):
            raise ValueError("Error: activation functions mismatch with number of layers.")

        for act in activations:
            if act not in {'relu', 'tanh', 'sigmoid'}:
                raise ValueError("Error: activation function must be 'relu', 'tanh', or 'sigmoid'.")

    return layers, activations, learning_rate, epochs, early_stop


if __name__ == "__main__":
    # try:
        # Get parameters from the command line
        params = parse_parameters()
        layers, activations, learning_rate, epochs, early_stop = get_parameters(params)

        # Retrieve training and validation sets
        X_train, X_val, y_train, y_val = preprocess_and_split_data()

        # Make sure data is numpy arrays and floats
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        mean = np.mean(X_train, axis=0, keepdims=True)
        std = np.std(X_train, axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std

        input_size = X_train.shape[1]  # number of features

        # Initialize the model
        model = MLP(input_size)

        for layer, activation in zip(layers, activations):
            if activation:
                model.add_layer(layer, activation=activation)
            else:
                model.add_layer(layer)

        # Add output layer containing 2 neurons
        model.add_layer(2, activation='softmax')

        # Train the model
        model.fit(X_train, y_train, X_val, y_val, epochs=epochs, lr=learning_rate, early_stop=early_stop)

        # Save trained MLP model
        model.save_model("cancer_detection.json")
        print("Model 'cancer_detection.json' saved locally.")

    # except Exception as e:
    #     print(f"Error: {e}")