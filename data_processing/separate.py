import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import close_on_key, get_top_correlations

COLUMN_NAMES = [
    'Id',
    'Diagnosis',
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
    'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst',
]


def load_dataset(data_path: str) -> DataFrame:
    df = pd.read_csv(data_path, header=None)

    if df.iloc[0, 0] == 'Id' or df.iloc[0, 1] == 'Diagnosis':
        df = df.iloc[1:].reset_index(drop=True)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected {len(COLUMN_NAMES)} columns, found {df.shape[1]}.\n"
            f"Actual columns: {df.shape[1]}"
        )

    df.columns = COLUMN_NAMES
    return df


def preprocess_dataset(df: DataFrame):
    """
    Preprocess dataset by converting diagnosis into 0/1 (Benign/Malignant),
    converting features to numeric values, and removing invalid rows.
    """
    df = df.copy()

    df['Diagnosis'] = df['Diagnosis'].astype(str).str.strip()

    df['Diagnosis_raw'] = df['Diagnosis']
    df['Diagnosis_num'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    feature_columns = [c for c in df.columns if c not in ['Id', 'Diagnosis', 'Diagnosis_raw', 'Diagnosis_num']]

    for c in feature_columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df.dropna(inplace=True)

    print(df['Diagnosis_raw'].value_counts()) 

    return df


def plot_diagnosis_distribution(df: DataFrame):
    """
    Plot distribution of diagnosis values in the main dataset.
    """
    plot = sns.countplot(
        x='Diagnosis_raw',
        data=df,
        palette={'B': 'skyblue', 'M': 'salmon'}
    )

    plot.figure.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Diagnosis Class Distribution (B = Benign, M = Malignant)")
    plt.show()


def plot_correlation_heatmap(df: DataFrame, top_features):
    """
    Plot a correlation heatmap of the features most related to diagnosis.
    """
    selected_columns = ['Diagnosis_num'] + list(top_features.index)

    corr_matrix = df[selected_columns].corr(numeric_only=True)

    corr_matrix = corr_matrix.rename(columns={'Diagnosis_num': 'Diagnosis'},
                                     index={'Diagnosis_num': 'Diagnosis'})

    fig = plt.figure(figsize=(12, 5))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Correlation Heatmap of Features")
    plt.show()


def plot_top_correlated_features(df: DataFrame, top_features):
    """
    Plot the features most strongly correlated with the diagnosis.
    """
    strongest_values = top_features.sort_values()

    fig = plt.figure(figsize=(8, 5))
    sns.barplot(
        x=strongest_values.values,
        y=strongest_values.index,
        palette='coolwarm',
        hue=strongest_values.index,
        legend=False,
    )
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Features Most Correlated with Diagnosis")
    plt.xlabel("Correlation with Diagnosis")
    plt.tight_layout()
    plt.show()


def plot_pair_plot(df: DataFrame, top_features):
    """
    Plot the pair plot of the dataset with 5 chosen variables.
    """
    top_feature_names = list(top_features.index[:2])

    correlations = (
        df.corr(numeric_only=True)['Diagnosis_num']
        .drop('Diagnosis_num')
        .abs()
    )

    low_features = list(
        correlations
        .drop(top_feature_names)
        .sort_values(ascending=True)
        .head(2)
        .index
    )

    selected_columns = top_feature_names + low_features + ['Diagnosis_raw']

    plot = sns.pairplot(
        df[selected_columns],
        hue="Diagnosis_raw",
        height=2.5,
        aspect=1.5
    )

    plot.figure.canvas.mpl_connect('key_press_event', close_on_key)
    plt.suptitle("Feature Pairplot (subset)", y=1.02)
    plt.show()


def plot_split_distribution(y_train, y_val):
    """
    Plot the distribution of the diagnosis in the train and validation datasets.
    """
    y_train = pd.Series(y_train).map({1: 'M', 0: 'B'})
    y_val = pd.Series(y_val).map({1: 'M', 0: 'B'})

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

    classes = np.unique(y)

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_val = max(1, round(len(cls_indices) * test_size))

        val_indices.extend(cls_indices[:n_val])
        train_indices.extend(cls_indices[n_val:])

    # Final shuffle to mix classes
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    return (
        X[train_indices],
        X[val_indices],
        y[train_indices],
        y[val_indices]
    )


def split_dataset(df: DataFrame, test_size=0.2):
    """
    Split the dataset into train and validation datasets.
    """
    X = df.drop(columns=['Diagnosis', 'Diagnosis_raw', 'Diagnosis_num'])
    y = df['Diagnosis_num']

    X_train, X_val, y_train, y_val = train_test_split_custom(
        X, y, test_size=test_size, random_state=42
    )

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)

    return X_train, X_val, y_train, y_val


def save_split_dataset(X_train, X_val, y_train, y_val):
    """
    Save the train and validation datasets into .csv files.
    """
    os.makedirs("datasets", exist_ok=True)

    def build_df(X, y):
        df = X.copy()

        # convert back to label
        y = pd.Series(y).map({1: 'M', 0: 'B'}).values

        # insert as SECOND column (after Id if exists)
        if 'Id' in df.columns:
            cols = df.columns.tolist()
            cols.remove('Id')
            df = df[['Id'] + cols]

        df.insert(1, 'Diagnosis', y)

        return df

    train_df = build_df(X_train, y_train)
    val_df = build_df(X_val, y_val)

    train_df.to_csv("datasets/data_training.csv", index=False, header=False)
    val_df.to_csv("datasets/data_test.csv", index=False, header=False)


if __name__ == "__main__":
    try:
        if len(sys.argv) == 2:
            data_path = sys.argv[1]
        else:
            raise ValueError("Please provide dataset path.")

        if not os.path.exists(data_path):
            raise ValueError("Dataset file does not exist. Provide it in directory and try again.")

        # Load and clean the data
        df = load_dataset(data_path)
        df = preprocess_dataset(df)

        # Visualize DataFrame on the terminal
        print("\n🔹 First 5 samples of DataFrame: 🔹", df.head())

        # Dataset Visualization
        plot_diagnosis_distribution(df)

        top_features = get_top_correlations(df, 10)
        print("\nTop 10 features most correlated with Diagnosis:")
        for feature, value in top_features.items():
            print(f"- {feature}: {value:.4f}")

        plot_correlation_heatmap(df, top_features)
        plot_top_correlated_features(df, top_features)
        plot_pair_plot(df, top_features)
        
        # Split into train/validation sets and save them as .csv
        X_train, X_val, y_train, y_val = split_dataset(df)
        save_split_dataset(X_train, X_val, y_train, y_val)

        print("\nTrain and Validation datasets saved in datasets directory.")

        plot_split_distribution(y_train, y_val)

    except Exception as e:
        print(f"Error: {e}")
