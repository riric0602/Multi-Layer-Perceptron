import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import close_on_key

COLUMN_NAMES = [
    'Id',
    'Diagnosis',
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
    'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst',
]


def preprocess_dataset(df: DataFrame):
    """
    Preprocess dataset by converting diagnosis into 0/1 (Benign/Malignant).
    Remove any null values and drop the ID column.
    """
    df.drop(columns=['Id'], inplace=True, errors='ignore')
    df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

    if df.isnull().values.any():
        print("Warning: Dataset contains missing values.")
        df.dropna(inplace=True)

    return df


def plot_diagnosis_distribution(df: DataFrame):
    """
    Plot distribution of diagnosis values in the main dataset.
    """
    plot = sns.countplot(
        x='Diagnosis',
        hue='Diagnosis',
        data=df)
    plot.figure.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Diagnosis Class Distribution (0 = Benign, 1 = Malignant)")
    plt.show()


def plot_correlation_heatmap(df: DataFrame):
    """
    Plot correlation heatmap of the 10 middle features of the dataset.
    This plot shows how proportionate some variables are with the diagnosis.
    """
    # Select the columns to be used in the HeatMap Plot
    middle_10_features = df.columns[12:22]
    selected_columns = ['Diagnosis'] + list(middle_10_features)

    # Create the correlation matrix
    df_corr = df[selected_columns]
    corr_matrix = df_corr.corr()

    fig = plt.figure(figsize=(12, 5))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Correlation Heatmap of Features")
    plt.show()


def plot_top_correlated_features(df):
    """
    Plot the most correlated features of the dataset with the Malignant diagnosis.
    """
    # Get positive correlations with target -> Malignant correlations
    correlations = df.corr()['Diagnosis'].drop('Diagnosis').sort_values()
    positive_values = correlations[correlations > 0].sort_values(ascending=False)

    fig = plt.figure(figsize=(8, 5))
    sns.barplot(
        x=positive_values.values,
        y=positive_values.index,
        palette='Reds',
        hue=positive_values.index,
        legend=False,
    )
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Top Features Associated with Malignant Diagnosis")
    plt.xlabel("Correlation with Diagnosis (1 = Malignant)")
    plt.tight_layout()
    plt.show()


def plot_pair_plot(df: DataFrame):
    """
    Plot the pair plot of the dataset with 5 chosen variables.
    """
    plot = sns.pairplot(
        df[
            ['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Diagnosis']
        ],
        height=2.5,
        aspect = 1.5,
        palette='Set2',
        hue="Diagnosis"
    )
    plot.figure.canvas.mpl_connect('key_press_event', close_on_key)
    plt.suptitle("Feature Pairplot (subset)", y=1.02)
    plt.show()


def plot_split_distribution(y_train, y_val):
    """
    Plot the distribution of the diagnosis in the train and validation datasets.
    """
    split_data = pd.DataFrame({
        'Diagnosis': list(y_train) + list(y_val),
        'Split': ['Train'] * len(y_train) + ['Validation'] * len(y_val)
    })

    fig = plt.figure(figsize=(6, 4))
    sns.countplot(data=split_data, x='Split', hue='Diagnosis', palette='Set2')
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Class Distribution in Train vs Validation")
    plt.xlabel("Dataset Split")
    plt.ylabel("Count")
    plt.legend(title="Diagnosis", labels=["Benign (0)", "Malignant (1)"])
    plt.tight_layout()
    plt.show()


def train_test_split_custom(X, y, test_size=0.2, random_state=None):
    """
    Split dataset into train and validation sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    train_indices = []
    val_indices = []

    # Iterate over Malignant/Benign classes
    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_val = int(len(cls_indices) * test_size)

        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    # Shuffle final indices to avoid class ordering
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return (
        X[train_indices], X[val_indices],
        y[train_indices], y[val_indices]
    )


def split_dataset(df: DataFrame):
    """
    Split the dataset into train and validation datasets.
    """
    # Convert dataframe into input and output
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    X_train, X_val, y_train, y_val = train_test_split_custom(
        X_scaled, y, test_size=0.2, random_state=42
    )

    X_train = pd.DataFrame(X_train)
    X_train["Diagnosis"] = y_train
    X_val = pd.DataFrame(X_val)
    X_val["Diagnosis"] = y_val

    return X_train, X_val, y_train, y_val


def save_split_dataset(X_train, X_val, y_train, y_val):
    """
    Save the train and validation datasets into .csv files.
    """
    train_df = X_train.copy()
    train_df["Diagnosis"] = y_train

    val_df = X_val.copy()
    val_df["Diagnosis"] = y_val

    os.makedirs("datasets", exist_ok=True)
    train_df.to_csv(os.path.join("datasets", "train.csv"), index=False)
    val_df.to_csv(os.path.join("datasets", "val.csv"), index=False)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            data_path = sys.argv[1]
        else:
            raise ValueError("Please provide dataset path.")

        if not os.path.exists(data_path):
            raise ValueError("Dataset file does not exist. Provide it in directory and try again.")

        # Load and clean the data
        df = pd.read_csv(data_path, names=COLUMN_NAMES, header=0)
        df = preprocess_dataset(df)

        # Visualize DataFrame on the terminal
        print("\n🔹 First 5 samples of DataFrame: 🔹", df.head())

        # Dataset Visualization
        plot_diagnosis_distribution(df)
        plot_correlation_heatmap(df)
        plot_top_correlated_features(df)
        plot_pair_plot(df)

        # Split into train/validation sets and save them as .csv
        X_train, X_val, y_train, y_val = split_dataset(df)
        save_split_dataset(X_train, X_val, y_train, y_val)

        plot_split_distribution(y_train, y_val)

    except Exception as e:
        print(f"Error: {e}")