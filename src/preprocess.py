import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

def impute_missing_values(X):
    categorical_cols_with_missing = ['workclass', 'occupation', 'native-country']
    imp = SimpleImputer(strategy='most_frequent')

    X.loc[:, categorical_cols_with_missing] = imp.fit_transform(X[categorical_cols_with_missing])

def normalize_target_classes(y):
    return y.map(lambda v: v.rstrip('.'))

def preprocess(X, y):
    """
    Preprocess the data: impute missing values, encode categorical variables, and standardize numerical features.
    """
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Impute missing values separately for categorical and numerical data
    impute_missing_values(X)

    # One-hot encode categorical columns
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Standardize numerical features only
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Encode target variable
    y = y.iloc[:, 0]
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y.str.rstrip('.'))

    return X.astype(np.float64), np.array(y)
