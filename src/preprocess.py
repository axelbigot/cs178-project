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

def preprocess_neural_net(X, y):
    """
    Preprocess the data: impute missing values, encode categorical variables, and standardize features.
    """
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include = ['object']).columns.tolist()

    # Impute missing values
    impute_missing_values(X)

    # One-Hot Encode categorical columns
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output = False, drop = 'first', handle_unknown = 'ignore')
        X_encoded = encoder.fit_transform(X[categorical_cols])
        X_encoded_df = pd.DataFrame(X_encoded,
                                    columns = encoder.get_feature_names_out(categorical_cols),
                                    index = X.index)

        # Drop original categorical columns and add encoded ones
        X = X.drop(columns = categorical_cols)
        X = pd.concat([X, X_encoded_df], axis = 1)

    # Standardize numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns, index = X.index)

    # Encode target variable
    y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
    if y.dtype == 'object':
        y = y.str.rstrip('.')
        le = LabelEncoder()
        y = le.fit_transform(y)

    return X.astype(np.float64), np.array(y)  # Ensure all X values are float64