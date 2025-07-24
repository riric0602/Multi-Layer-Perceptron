import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils import close_on_key

if __name__ == "__main__":
    try:
        data_file = 'data.csv'

        if not os.path.exists(data_file):
            print("Dataset file does not exist. Provide it in directory and try again.")
            sys.exit()

        # Define column names in dataset
        column_names = [
            'Id',
            'Diagnosis',
            'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
            'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
            'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst',
        ]

        df = pd.read_csv(data_file, names=column_names, header=0)

        # Clean the data
        df.drop(columns=['Id'], inplace=True, errors='ignore')
        df['Diagnosis'] = df['Diagnosis'].map({'B': 0, 'M': 1})

        # Visualize DataFrame on the terminal
        print("\n🔹 First 5 samples:", df.head())
        print("\n🔹 Class distribution:", df['Diagnosis'].value_counts())

        # Plot Class Distribution of Diagnosis
        sns.countplot(x='Diagnosis', data=df)
        plt.title("Diagnosis Class Distribution (0 = Benign, 1 = Malignant)")
        plt.show()

        fig = plt.figure(figsize=(16, 12))
        corr_matrix = df.corr()

        sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
        fig.canvas.mpl_connect('key_press_event', close_on_key)
        plt.title("Correlation Heatmap of Features")
        plt.show()

        sns.pairplot(df[['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Diagnosis']], hue="Diagnosis")
        plt.suptitle("Feature Pairplot (subset)", y=1.02)
        plt.show()

        fig = plt.figure(figsize=(14, 6))
        sns.boxplot(x='Diagnosis', y='Area Mean', data=df)
        fig.canvas.mpl_connect('key_press_event', close_on_key)
        plt.title("Boxplot of Area Mean by Diagnosis")
        plt.show()

        correlation_with_target = df.corr()['Diagnosis'].drop('Diagnosis').sort_values(ascending=False)
        print("\n🔹 Top correlated features with Diagnosis:")
        print(correlation_with_target.head(10))

        top_features = correlation_with_target.head(10)
        fig = plt.figure(figsize=(10, 6))
        sns.barplot(x=top_features.values, y=top_features.index)
        fig.canvas.mpl_connect('key_press_event', close_on_key)
        plt.title("Top 10 Features Correlated with Diagnosis")
        plt.xlabel("Correlation Coefficient")
        plt.show()

        # Convert dataframe to Numpy with input and output
        X = df.drop(columns=['Diagnosis']).values
        y = df['Diagnosis'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

    except Exception as e:
        print(f"Error: {e}")