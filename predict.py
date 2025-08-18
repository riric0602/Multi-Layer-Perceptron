import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MLP import MLP
import pandas as pd
import json
from separate import preprocess_dataset, DATA_FILE, COLUMN_NAMES


def load_model(filepath):
    with open(filepath, "r") as f:
        model_data = json.load(f)

    weights = [np.array(w) for w in model_data["weights"]]
    biases = [np.array(b) for b in model_data["biases"]]
    activations = [a for a in model_data["activations"]]

    return weights, biases, activations


def read_and_scale_data():
    # Load and preprocess the dataset
    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
    dataset = preprocess_dataset(df)

    X_predict = dataset.drop(columns=['Diagnosis'])
    y_true = dataset['Diagnosis'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_predict)

    return X_scaled, y_true


def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    bce = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(bce)


if __name__ == "__main__":
    try:
        # Load weights and biases
        weights, biases, activations = load_model('cancer_detection.json')

        # Create model instance with input size based on first layer weights shape
        model = MLP(input_size=weights[0].shape[0])

        # Assign loaded weights and biases to the model
        model.weights = weights
        model.biases = biases
        model.activation_funcs = activations

        print("Model loaded successfully !")

        X_scaled, y_true = read_and_scale_data()
        probs = model.feedforward(X_scaled)
        y_pred = np.argmax(probs, axis=1)

        # Compute metrics
        loss = binary_cross_entropy(y_true.reshape(-1, 1), probs)
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)

        print("\n--------------Prediction Metrics:--------------")
        print(f"Binary Cross-Entropy Loss: {loss:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
    except Exception as e:
        print(f"Error: {e}")
