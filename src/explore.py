"""
Various functions used to explain & visualize the dataset.
Written as unittests for easy GUI execution in pycharm.
"""


import unittest
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif

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
        - The count and percentage of each unique value in the target variable
        - The total number of data points (rows)
        - The count of missing values in the feature set
        """

        total_rows = len(X)
        y_counts = y.value_counts()
        y_percentages = (y_counts / total_rows) * 100

        print(f"""
    Summary of dataset:
    {X.head()}

    Total data points: {total_rows}

    Target variable distribution: 
    {y_counts}

    Percentage of each class:
    {y_percentages}

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

        # Apply a logarithmic transformation for features with positive skewness
        transformed_X = X.copy()

        # List of features to transform (you can adjust based on the skewness you observe)
        skewed_features = ['capital-gain', 'capital-loss', 'fnlwgt']

        # Apply log transformation to skewed features
        for feature in skewed_features:
            transformed_X[feature] = transformed_X[feature].apply(
                lambda x: np.log(x + 1) if x > 0 else 0)

        # Plot histograms of transformed features
        transformed_X.hist(figsize = (12, 10), bins = 30, edgecolor = "black")
        plt.suptitle("Transformed Distributions of Numerical Features", fontsize = 16)
        plt.show()

        # Plot boxplots of selected numerical features (log-transformed ones included)
        plt.figure(figsize = (12, 10))
        sns.boxplot(data = transformed_X[
            ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]])
        plt.title("Boxplot of Transformed Numerical Features")
        plt.xticks(rotation = 45)
        plt.show()

    def test_visualize_categorical(self):
        """
        Visualizes the distribution of categorical features in the dataset.
        This includes displaying:
        - Countplots for each categorical feature, showing the number of occurrences for each category
        """

        plt.figure(figsize = (15, 12))

        for i, col in enumerate(["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]):
            plt.subplot(4, 2, i + 1)
            sns.countplot(y = X[col], order = X[col].value_counts().index, palette = "viridis")
            plt.title(f"Distribution of {col}")
            plt.xlabel("Count")

        plt.tight_layout()
        plt.show()

    def test_visualize_income_correlation(self):
        """
        Visualizes the correlation between income and numerical and categorical features in the dataset.
        This includes:
        - Histograms for all numerical features, separated by income class
        - Countplots for categorical features, showing income distribution
        """

        # Combine X and y into a single DataFrame for easier plotting
        df = X.copy()
        df['income'] = y

        # Ensure 'income' column is of string type (if not already)
        df['income'] = df['income'].astype(str)

        # Define a custom color palette for income (adjust to match cleaned labels)
        income_palette = {"<=50K": "darkblue", ">50K": "lightblue"}

        # Plot all numerical variables by income class
        numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                              "hours-per-week"]

        plt.figure(figsize = (12, 12))
        for i, col in enumerate(numerical_features):
            plt.subplot(3, 2, i + 1)
            sns.histplot(data = df, x = col, bins = 30, kde = True, hue = "income",
                         element = "step", palette = income_palette)
            plt.title(f"{col.replace('-', ' ').title()} Distribution by Income Class")

        plt.tight_layout()
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

    def test_visualize_capital_distributions(self):
        """
        Visualizes the distribution of capital-gain and capital-loss, separated by income class.
        This includes:
        - Histograms in linear scale (including zeros)
        - Histograms in log scale (excluding zeros)
        """

        # Combine X and y into a single DataFrame for easier plotting
        df = X.copy()
        df["income"] = y.astype(str)  # Ensure 'income' is a string for hue mapping

        # Define a custom color palette for income classes
        income_palette = {"<=50K": "darkblue", ">50K": "lightblue"}

        plt.figure(figsize = (12, 10))

        # Capital Gain (Linear Scale) - Keep all values
        plt.subplot(2, 2, 1)
        sns.histplot(data = df, x = "capital-gain", bins = 30, kde = False, hue = "income",
                     element = "step", palette = income_palette)
        plt.title("Capital Gain Distribution (Linear Scale)")

        # Capital Loss (Linear Scale) - Keep all values
        plt.subplot(2, 2, 2)
        sns.histplot(data = df, x = "capital-loss", bins = 30, kde = False, hue = "income",
                     element = "step", palette = income_palette)
        plt.title("Capital Loss Distribution (Linear Scale)")

        # Capital Gain (Log Scale) - Remove zeros
        plt.subplot(2, 2, 3)
        sns.histplot(data = df[df["capital-gain"] > 0],
                     x = np.log(df[df["capital-gain"] > 0]["capital-gain"]),
                     bins = 30, kde = True, hue = "income", element = "step",
                     palette = income_palette)
        plt.title("Capital Gain Distribution (Log Scale, No Zeros)")

        # Capital Loss (Log Scale) - Remove zeros
        plt.subplot(2, 2, 4)
        sns.histplot(data = df[df["capital-loss"] > 0],
                     x = np.log(df[df["capital-loss"] > 0]["capital-loss"]),
                     bins = 30, kde = True, hue = "income", element = "step",
                     palette = income_palette)
        plt.title("Capital Loss Distribution (Log Scale, No Zeros)")

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

    def test_income_class_distribution(self):
        """
        Visualizes the distribution of income classes in the dataset.
        This helps identify any imbalance in the dataset that may impact model performance.
        """
        # Ensure y is a Series by selecting a column if needed
        y_series = y.iloc[:, 0]  # Select the first column if y is a DataFrame

        plt.figure(figsize = (8, 4))
        sns.countplot(data = pd.DataFrame({"income": y_series}), y = "income", palette = "coolwarm")
        plt.title("Income Class Distribution")
        plt.xlabel("Count")
        plt.ylabel("Income Class")
        plt.show()

    def test_boxplots_by_income(self):
        """
        Displays boxplots of numerical features grouped by income class.
        This helps identify outliers and how numerical distributions differ between income levels.
        """

        df = X.copy()
        df["income"] = y

        numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                              "hours-per-week"]

        plt.figure(figsize = (15, 10))
        for i, col in enumerate(numerical_features):
            plt.subplot(3, 2, i + 1)
            sns.boxplot(data = df, x = "income", y = col, palette = "coolwarm")
            plt.title(f"{col.replace('-', ' ').title()} by Income Class")

        plt.tight_layout()
        plt.show()

    def test_capital_gain_loss_distribution(self):
        """
        Visualizes the distribution of capital-gain and capital-loss on both linear and log scales.
        This highlights the skewed nature of these features and the presence of extreme values.
        """

        df = X.copy()
        df["income"] = y

        plt.figure(figsize = (12, 10))

        # Capital Gain (Linear Scale)
        plt.subplot(2, 2, 1)
        sns.histplot(df[df["capital-gain"] > 0]["capital-gain"], bins = 30, kde = True)
        plt.title("Capital Gain Distribution (Linear Scale)")

        # Capital Loss (Linear Scale)
        plt.subplot(2, 2, 2)
        sns.histplot(df[df["capital-loss"] > 0]["capital-loss"], bins = 30, kde = True)
        plt.title("Capital Loss Distribution (Linear Scale)")

        # Capital Gain (Log Scale)
        plt.subplot(2, 2, 3)
        sns.histplot(df[df["capital-gain"] > 0]["capital-gain"], bins = 30, kde = True,
                     log_scale = True)
        plt.title("Capital Gain Distribution (Log Scale)")

        # Capital Loss (Log Scale)
        plt.subplot(2, 2, 4)
        sns.histplot(df[df["capital-loss"] > 0]["capital-loss"], bins = 30, kde = True,
                     log_scale = True)
        plt.title("Capital Loss Distribution (Log Scale)")

        plt.tight_layout()
        plt.show()

    def test_mutual_information(self):
        """
        Computes and visualizes the mutual information between features and income.
        This helps in understanding which features provide the most predictive power for classification.
        """

        df = X.copy()
        df["income"] = y.replace({">50K": 1, "<=50K": 0})  # Convert to binary labels

        # Encoding categorical columns using OneHotEncoder
        categorical_cols = X.select_dtypes(include = ["object"]).columns
        encoder = OneHotEncoder(sparse_output = False,
                                drop = "first")  # Use sparse_output instead of sparse
        encoded_features = encoder.fit_transform(X[categorical_cols])

        # Convert encoded features back to a DataFrame
        encoded_df = pd.DataFrame(encoded_features,
                                  columns = encoder.get_feature_names_out(categorical_cols))

        # Concatenate the encoded features with the numeric columns
        X_encoded = pd.concat([X.select_dtypes(exclude = ["object"]), encoded_df], axis = 1)

        # Compute mutual information
        mi_scores = mutual_info_classif(X_encoded, df["income"], discrete_features = "auto")
        mi_series = pd.Series(mi_scores, index = X_encoded.columns).sort_values(ascending = False)

        # Limit to top 20 most important features
        mi_series_top_20 = mi_series.head(20)

        # Visualize top 20 feature importance
        plt.figure(figsize = (12, 8))  # Increased figure size for better readability
        sns.barplot(x = mi_series_top_20, y = mi_series_top_20.index, palette = "coolwarm")

        # Increase font size for better readability
        plt.title("Top 20 Feature Importance via Mutual Information", fontsize = 16)
        plt.xlabel("Mutual Information Score", fontsize = 14)
        plt.ylabel("Feature", fontsize = 14)

        # Rotate y-axis labels for better readability
        plt.yticks(rotation = 0, fontsize = 12)  # Adjust rotation and font size

        # Adjust spacing to avoid label overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

    def test_features_to_features(self):
        """
        Visualizes the relationship between all features.
        Categorical vs Categorical is represented with a count plot
        Numerical vs Categorical is represented with a box plot
        Numerical vs Numerical is represented with a correlation heat map
        """
        df = X.copy()

        categorical_features = df.select_dtypes(include=['object', 'category']).columns
        numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

        # Categorical vs Numerical comparisons
        for cat in categorical_features:
            for num in numerical_features:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=cat, y=num, data=df)
                plt.title(f'Box Plot: {cat} vs {num}')
                plt.show()

        # Correlation Matrix for Numerical Features
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix for Numerical Features')
        plt.show()

        # Categorical vs Categorical comparisons
        for cat1 in categorical_features:
            for cat2 in categorical_features:
                if cat1 != cat2:
                    plt.figure(figsize=(6, 4))
                    sns.countplot(x=cat1, hue=cat2, data=df)
                    plt.title(f'Count Plot: {cat1} vs {cat2}')
                    plt.show()