import unittest
import matplotlib.pyplot as plt
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

        plt.figure(figsize = (8, 12))
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
        Visualizes the distribution of capital-gain and capital-loss on a log scale.
        This highlights the skewed nature of these features and the presence of extreme values.
        """

        df = X.copy()
        df["income"] = y

        plt.figure(figsize = (12, 5))

        # Capital Gain
        plt.subplot(1, 2, 1)
        sns.histplot(df[df["capital-gain"] > 0]["capital-gain"], bins = 30, kde = True,
                     log_scale = True)
        plt.title("Capital Gain Distribution (Log Scale)")

        # Capital Loss
        plt.subplot(1, 2, 2)
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
        # Convert y to binary labels
        df = X.copy()
        df["income"] = y.replace({">50K": 1, "<=50K": 0})  # Convert to binary labels

        # Initialize OneHotEncoder and fit_transform the categorical columns
        encoder = OneHotEncoder(sparse_output = False,
                                drop = 'first')  # Drop first to avoid collinearity
        categorical_columns = X.select_dtypes(include = ['object']).columns
        X_encoded = encoder.fit_transform(X[categorical_columns])

        # Combine the encoded categorical features back with the rest of the numerical features
        X_encoded_df = pd.DataFrame(X_encoded,
                                    columns = encoder.get_feature_names_out(categorical_columns))
        X_final = pd.concat([X.drop(columns = categorical_columns), X_encoded_df], axis = 1)

        # Compute mutual information
        mi_scores = mutual_info_classif(X_final, df["income"], discrete_features = "auto")
        mi_series = pd.Series(mi_scores, index = X_final.columns).sort_values(ascending = False)

        # Select top 20 features
        top_20_features = mi_series.head(20)

        # Create the plot
        plt.figure(figsize = (10, 5))

        # Use seaborn's barplot to create the plot
        sns.barplot(x = top_20_features, y = top_20_features.index,
                    palette = 'coolwarm')  # Modify palette if you want specific colors

        # Title and labels
        plt.title("Top 20 Features via Mutual Information (Colored by Target)")
        plt.xlabel("Mutual Information Score")
        plt.ylabel("Feature")
        plt.show()

