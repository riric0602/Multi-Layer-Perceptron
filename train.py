import numpy as np
from MLP import MLP
import argparse
from separate import preprocess_and_split_data

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Train a Multi-Layer Perceptron model."
#     )
#
#     parser.add_argument(
#         "--l",
#         '--add_layers',
#         type=str,
#         nargs="+",
#         default='24 24',
#         help="Layers' sizes to add."
#     )
#     parser.add_argument(
#         '-a',
#         '--activations',
#         type=str,
#         default=0.001,
#         help="Learning rate for training."
#     )
#     parser.add_argument(
#         '-lr',
#         '--learning_rate',
#         type=int,
#         default=0.001,
#         help="Learning rate for training."
#     )
#     parser.add_argument(
#         'e',
#         '--epochs',
#         type=int,
#         default=100,
#         help="Number of epochs to train."
#     )

if __name__ == "__main__":
    try:
        # Retrieve training and validation sets
        X_train, X_val, y_train, y_val = preprocess_and_split_data()

        # Make sure data is numpy arrays and floats
        X_train = np.array(X_train, dtype=np.float32)
        X_val = np.array(X_val, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.float32)

        input_size = X_train.shape[1]  # number of features

        # Initialize the model
        model = MLP(input_size)

        # Add 3 layers
        model.add_layer(24, activation='relu')
        model.add_layer(24, activation='relu')
        model.add_layer(1, activation='sigmoid')

        # Train the model
        model.fit(X_train, y_train, X_val, y_val, epochs=600, learning_rate=0.05)

        # Save trained MLP model
        model.save_model("cancer_detection.json")

    except Exception as e:
        print(f"Error: {e}")