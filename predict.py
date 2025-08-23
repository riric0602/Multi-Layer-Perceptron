import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
from MLP import MLP
import pandas as pd
import argparse
from separate import preprocess_dataset, COLUMN_NAMES
from utils import load_model


DATA_FILE = "data.csv"


def read_and_scale_data():
    """
    Read and scale the given dataset to be predicted.
    """
    # Load and preprocess the dataset
    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
    dataset = preprocess_dataset(df)

    X_predict = dataset.drop(columns=['Diagnosis'])
    y_true = dataset['Diagnosis'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_predict)

    return X_scaled, y_true


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross entropy loss for the results.
    """
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    bce = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(bce)


def parse_parameters():
    """
    Parse the command line parameters.
    """
    parser = argparse.ArgumentParser(
        description="Predict if a given dataset's cells are malignant or benign."
    )

    parser.add_argument("-n", '--model_name', type=str, default="cancer_detection", help="Name of the model to use for prediction.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        # Retrieve trained model to use for prediction
        params = parse_parameters()
        name = params.model_name

        # Load weights and biases
        weights, biases, activations, _, _, _, _ = load_model(f'{name}.json')

        # Create model instance with input size based on first layer weights shape
        model = MLP(input_size=weights[0].shape[0])

        # Assign loaded weights and biases to the model
        model.weights = weights
        model.biases = biases
        model.activation_funcs = activations

        print("Model loaded successfully !")

        X_scaled, y_true = read_and_scale_data()
        probs = model.feedforward(X_scaled, model.weights, model.biases)
        probs_binary = probs[:, 1].reshape(-1, 1)

        y_pred = (probs_binary > 0.5).astype(int).flatten()

        # Compute metrics
        loss = binary_cross_entropy(y_true.reshape(-1, 1), probs_binary)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        print("\n--------------Prediction Metrics:--------------")
        print(f"Binary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
    except Exception as e:
        print(f"Error: {e}")
