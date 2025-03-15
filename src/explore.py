import unittest
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

from main import X, y, adult


class TestDataVisualization(unittest.TestCase):
    def test_metadata(self):
        """
        Displays the metadata and variable information of the dataset.
        The metadata includes details of the dataset's structure, and the
        variable information provides insights into the target variable.
        """

        print(f"""
Metadata:
{adult.metadata}

Variable Information:
{adult.variables}
        """)

    def test_summary(self):
        """
        Prints a summary of the dataset, including:
        - The first few rows of the dataset
        - The count of each unique value in the target variable
        - The count of missing values in the feature set
        """

        print(f"""
Summary of dataset:
{X.head()}

Target variable: 
{y.value_counts()}

Missing values: 
{X.isnull().sum()}
        """)

    def test_visualize_numerical(self):
        """
        Visualizes the distributions of numerical features in the dataset.
        This includes displaying:
        - Histograms for numerical features with bin counts
        - Boxplots for selected numerical features to identify outliers
        """

        sns.set(style = "whitegrid")

        X.hist(figsize = (12, 10), bins = 30, edgecolor = "black")
        plt.suptitle("Distributions of Numerical Features", fontsize = 16)
        plt.show()

        plt.figure(figsize = (12, 6))
        sns.boxplot(data = X[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]])
        plt.title("Boxplot of Numerical Features")
        plt.xticks(rotation = 45)
        plt.show()

    def test_visualize_categorical(self):
        """
        Visualizes the distribution of categorical features in the dataset.
        This includes displaying:
        - Countplots for each categorical feature, showing the number of occurrences for each category
        """

        plt.figure(figsize = (15, 12))

        for i, col in enumerate(["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]):
            plt.subplot(4, 2, i + 1)
            sns.countplot(y = X[col], order = X[col].value_counts().index, palette = "viridis")
            plt.title(f"Distribution of {col}")
            plt.xlabel("Count")

        plt.tight_layout()
        plt.show()

    def test_visualize_income_correlation(self):
        """
        Visualizes the correlation between income and other features in the dataset.
        This includes displaying:
        - Age distribution by income class using a histogram
        - Distribution of categorical features (e.g., education, marital-status) by income class using countplots
        """

        # Combine X and y into a single DataFrame for easier plotting
        df = X.copy()
        df['income'] = y

        # Remove trailing dots and strip leading/trailing spaces from 'income' column
        df['income'] = df['income'].str.replace(r'\.$', '', regex=True).str.strip()

        # Ensure 'income' column is of string type (if not already)
        df['income'] = df['income'].astype(str)

        # Define a custom color palette for income (adjust to match cleaned labels)
        income_palette = {"<=50K": "darkblue", ">50K": "lightblue"}

        # Plot age distribution by income class
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x="age", bins=30, kde=True, hue="income", element="step", palette=income_palette)
        plt.title("Age Distribution by Income Class")
        plt.show()

        # Plot categorical feature distribution by income class
        categorical_features = ["education", "marital-status", "occupation", "race", "sex"]

        plt.figure(figsize=(15, 10))
        for i, col in enumerate(categorical_features):
            plt.subplot(3, 2, i + 1)
            sns.countplot(data=df, x=col, hue="income", palette=income_palette)
            plt.xticks(rotation=45)
            plt.title(f"Income Distribution by {col}")

        plt.tight_layout()
        plt.show()

    def test_visualize_numerical_correlations(self):
        """
        Visualizes the correlations between numerical features in the dataset.
        This includes displaying a heatmap that shows the correlation coefficients
        between numerical features in the dataset.
        """

        numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

        corr_matrix = X[numerical_columns].corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        plt.show()
