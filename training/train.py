import numpy as np
import pandas as pd
from MLP import MLP
import argparse
import os

def parse_parameters():
    """
    Parse command line arguments for training options.
    """
    parser = argparse.ArgumentParser(
        description="Train a Multi-Layer Perceptron model for binary classification of breast cancer."
    )

    parser.add_argument("-l", '--add_layers', type=int, nargs="+", default=[24, 24, 24], help="Layers' sizes to add.")
    parser.add_argument('-a', '--activations', type=str, nargs="+", help="Learning rate for training.")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help="Learning rate for training.")
    parser.add_argument('-e', '--epochs', type=int, default=500, help="Number of epochs to train.")
    parser.add_argument('-es', '--early_stop', type=int, default=None, help="Number of epochs to stop before overfitting.")
    parser.add_argument('-m', '--nesterov', type=float, default=None,
                        help="Momentum lookahead for the nesterov optimization.")
    parser.add_argument('-n', '--model_name', type=str, default="cancer_detection",
                        help="Name of the model to be saved after training.")
    parser.add_argument('-s', '--bonus_metrics', action="store_true",
                        help="Additional metrics for the learning phase.")
    parser.add_argument('-H', '--history', action="store_true",
                        help="History of metrics obtained during training.")

    return parser.parse_args()


def get_parameters(params):
    """
    Parse command line arguments for training options.
    """
    layers = params.add_layers
    learning_rate = params.learning_rate
    epochs = params.epochs
    activations = [None] * len(layers)
    early_stop = params.early_stop
    momentum = params.nesterov
    name = params.model_name
    metrics = params.bonus_metrics
    history = params.history

    if params.activations is not None:
        activations = [act.lower() for act in params.activations]

        if len(activations) != len(layers):
            raise ValueError("Error: activation functions mismatch with number of layers.")

        for act in activations:
            if act not in {'relu', 'tanh', 'sigmoid'}:
                raise ValueError("Error: activation function must be 'relu', 'tanh', or 'sigmoid'.")

    return layers, activations, learning_rate, epochs, early_stop, momentum, name, metrics, history


def get_train_validation_sets():
    """
    Retrieve training and validation sets.
    """

    train_path = os.path.join("datasets", "train.csv")
    val_path = os.path.join("datasets", "val.csv")
    if os.path.exists(train_path) and os.path.exists(val_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        X_train = train_df.drop(columns=["Diagnosis"])
        y_train = train_df["Diagnosis"]
        X_val = val_df.drop(columns=["Diagnosis"])
        y_val = val_df["Diagnosis"]

        return standarize_datasets(X_train, y_train, X_val, y_val)
    else:
        raise ValueError("Train and Validation datasets do not exist. Run separate.py script.")


def standarize_datasets(X_train, y_train, X_val, y_val):
    """
    Standardize the training and validation sets.
    """
    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train, y_train, X_val, y_val, mean, std


if __name__ == "__main__":
    try:
        # Get parameters from the command line
        params = parse_parameters()
        layers, activations, learning_rate, epochs, early_stop, momentum, name, metrics, history = get_parameters(params)

        # Retrieve training and validation sets
        X_train, y_train, X_val, y_val, mean, std = get_train_validation_sets()

        # Initialize the model
        input_size = X_train.shape[1]
        model = MLP(input_size)
        model.scaler_mean = mean
        model.scaler_std = std

        for layer, activation in zip(layers, activations):
            if activation:
                model.add_layer(layer, activation=activation)
            else:
                model.add_layer(layer)

        # Add output layer containing 2 neurons
        model.add_layer(2, activation='softmax')

        # Train the model
        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            lr=learning_rate,
            patience=early_stop,
            momentum=momentum,
            metrics=metrics,
            history=history
        )

        # Save trained MLP model in 'models' folder
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", f"{name}.json")
        model.save_model(model_path)
        print(f"\nModel '{model_path}' saved locally.")

    except Exception as e:
        print(f"Error: {e}")