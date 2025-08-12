import numpy as np
from sklearn.preprocessing import StandardScaler

from MLP import MLP
import pandas as pd
import json
from separate import preprocess_dataset, DATA_FILE, COLUMN_NAMES


def load_model(filepath):
    with open(filepath, "r") as f:
        model_data = json.load(f)

    weights = [np.array(w) for w in model_data["weights"]]
    biases = [np.array(b) for b in model_data["biases"]]
    return weights, biases

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)

    bce = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(bce)


def predict(model, X):
    output = model.feedforward(X)  # output shape (num_samples, 1)
    preds = (output >= 0.5).astype(int)  # threshold at 0.5: 1 = malignant, 0 = benign
    return preds

if __name__ == "__main__":
    # Load weights and biases
    weights, biases = load_model('cancer_detection.json')

    # Create model instance with input size based on first layer weights shape
    model = MLP(input_size=weights[0].shape[0])

    # Assign loaded weights and biases to the model
    model.weights = list(weights)
    model.biases = list(biases)
    model.activation_funcs = ['relu', 'relu', 'sigmoid']

    print("Model loaded successfully")

    # Load and preprocess the dataset
    df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
    dataset = preprocess_dataset(df)

    X_predict = dataset.drop(columns=['Diagnosis'])
    y_true = dataset['Diagnosis'].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_predict)

    # Get raw predicted probabilities
    probs = model.feedforward(X_scaled)

    # Calculate and print binary cross-entropy loss
    print("y_true shape:", y_true.shape)
    print("probs shape:", probs.shape)
    print(probs)

    loss = binary_cross_entropy(y_true.reshape(-1, 1), probs)
    print(f"Binary Cross-Entropy Loss: {loss:.4f}")

    # predictions = predict(model, X_predict)
    #
    # for i, pred in enumerate(predictions):
    #     label = "Malignant" if pred == 1 else "Benign"
    #     print(f"Sample {i}: {label}")