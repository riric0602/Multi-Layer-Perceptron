import os
import sys
import pandas as pd
from pandas import DataFrame
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


def plot_diagnosis_distribution(df: DataFrame):
    plot = sns.countplot(x='Diagnosis', data=df)
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


if __name__ == "__main__":
    try:
        if not os.path.exists(DATA_FILE):
            print("Dataset file does not exist. Provide it in directory and try again.")
            sys.exit()

        # Load and clean the data
        df = pd.read_csv(DATA_FILE, names=COLUMN_NAMES, header=0)
        df.drop(columns=['Id'], inplace=True, errors='ignore')
        df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

        # Visualize DataFrame on the terminal
        print("\n🔹 First 5 samples of DataFrame:", df.head())

        # Select the columns to be used in the HeatMap Plot
        middle_10_features = df.columns[12:22]
        selected_columns =  ['Diagnosis'] + list(middle_10_features)

        # Dataset Visualization
        plot_diagnosis_distribution(df)
        plot_correlation_heatmap(df[selected_columns])
        plot_pair_plot(df)

        # Convert dataframe to Numpy with input and output
        X = df.drop(columns=['Diagnosis']).values
        y = df['Diagnosis'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    except Exception as e:
        print(f"Error: {e}")