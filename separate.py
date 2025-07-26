import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils import close_on_key

DATA_FILE = 'data.csv'
COLUMN_NAMES = [
    'Id',
    'Diagnosis',
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
    'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst',
]


def preprocess_dataset(df: DataFrame):
    df.drop(columns=['Id'], inplace=True, errors='ignore')
    df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

    if df.isnull().values.any():
        print("Warning: Dataset contains missing values.")
        df.dropna(inplace=True)

    return df


def plot_diagnosis_distribution(df: DataFrame):
    plot = sns.countplot(
        x='Diagnosis',
        hue='Diagnosis',
        data=df)
    plot.figure.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Diagnosis Class Distribution (0 = Benign, 1 = Malignant)")
    plt.show()


def plot_correlation_heatmap(df: DataFrame):
    corr_matrix = df.corr()

    fig = plt.figure(figsize=(12, 5))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    fig.canvas.mpl_connect('key_press_event', close_on_key)
    plt.title("Correlation Heatmap of Features")
    plt.show()


def plot_top_correlated_features(df):
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


def split_dataset(df: DataFrame):
    # Convert dataframe into input and output
    X = df.drop(columns=['Diagnosis']).values
    y = df['Diagnosis'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    try:
        if not os.path.exists(DATA_FILE):
            print("Dataset file does not exist. Provide it in directory and try again.")
            sys.exit()

        # Load and clean the data
        df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
        df = preprocess_dataset(df)

        # Visualize DataFrame on the terminal
        print("\n🔹 First 5 samples of DataFrame:", df.head())

        # Select the columns to be used in the HeatMap Plot
        middle_10_features = df.columns[12:22]
        selected_columns =  ['Diagnosis'] + list(middle_10_features)

        # Dataset Visualization
        plot_diagnosis_distribution(df)
        plot_correlation_heatmap(df[selected_columns])
        plot_top_correlated_features(df)
        plot_pair_plot(df)

        X_train, X_val, y_train, y_val = split_dataset(df)
        plot_split_distribution(y_train, y_val)

    except Exception as e:
        print(f"Error: {e}")