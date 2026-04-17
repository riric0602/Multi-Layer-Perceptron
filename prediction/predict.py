import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.MLP import MLP
from data_processing.separate import preprocess_dataset, COLUMN_NAMES
from utils.utils import load_model


DATA_FILE = "datasets/data.csv"


def read_and_scale_data(mean, std):
    """
    Read and scale the given dataset to be predicted.
    """
    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
    dataset = preprocess_dataset(df)

    X = dataset.drop(columns=['Diagnosis']).values.astype(np.float32)
    y = dataset['Diagnosis'].values.reshape(-1, 1)

    mean = np.array(mean)
    std = np.array(std)
    std = np.where(std == 0, 1.0, std)

    mean = mean.flatten()
    std = std.flatten()

    X_scaled = (X - mean) / std

    return X_scaled, y


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross entropy loss for the results.
    """
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def parse_parameters():
    """
    Parse the command line parameters.
    """
    parser = argparse.ArgumentParser(
        description="Predict if a given dataset's cells are malignant or benign."
    )

    parser.add_argument("model_name", nargs="?", default="cancer_detection", help="Name of the model to use for prediction (positional argument).")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Retrieve trained model to use for prediction
        params = parse_parameters()
        name = params.model_name

        # Load weights and biases
        weights, biases, activations, scaler_mean, scaler_std, _, _, _, _ = load_model(f'models/{name}.json')

        # Create model instance with input size based on first layer weights shape
        model = MLP(input_size=weights[0].shape[0])

        # Assign loaded weights and biases to the model
        model = MLP(input_size=weights[0].shape[0])
        model.weights = weights
        model.biases = biases
        model.activation_funcs = activations
        model.scaler_mean = scaler_mean
        model.scaler_std = scaler_std 

        print("Model loaded successfully !")

        X_scaled, y_true = read_and_scale_data(mean=model.scaler_mean, std=model.scaler_std)
        probs = model.feedforward(X_scaled, model.weights, model.biases)  # shape (n, 2)
        y_pred = np.argmax(probs, axis=1)  # predicted class

        # One-hot encode y_true to match probs shape (n, 2)
        y_true_oh = np.zeros((len(y_true), 2))
        y_true_oh[np.arange(len(y_true)), y_true.flatten().astype(int)] = 1

        # Compute metrics
        loss = binary_cross_entropy(y_true_oh, probs)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        print("\n--------------Prediction Metrics:--------------")
        print(f"Binary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
    except Exception as e:
        print(f"Error: {e}")
