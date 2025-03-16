import pandas as pd
from keras.src.utils import to_categorical
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def impute_missing_values(X):
    categorical_cols_with_missing = ['workclass', 'occupation', 'native-country']
    imp = SimpleImputer(strategy = 'most_frequent')

    # Replace missing vals with most frequent occurrence of that feature.
    X[categorical_cols_with_missing] = imp.fit_transform(X[categorical_cols_with_missing])

def normalize_target_classes(y):
    # Normalizing some data. Some income fields have trailing period,
    # some don't. Remove trailing period for consistency.
    return y.applymap(lambda v: v.rstrip('.'))

def preprocess_neural_net(X, y):
    """
    Preprocess the data: encode categorical variables, ensure numeric values for neural network.
    """
    # Encode categorical columns in X
    categorical_cols = X.select_dtypes(include=['object']).columns  # Identify categorical columns

    encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Apply One-Hot Encoding to categorical columns
    X_encoded = encoder.fit_transform(X[categorical_cols])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Drop original categorical columns and add encoded ones
    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, X_encoded_df], axis=1)

    # Scale the numeric features
    scaler = StandardScaler()
    X[X.columns] = scaler.fit_transform(X)

    y = y.iloc[:, 0]  # Convert to Series if it's a DataFrame

    # Encode the target variable (y)
    if y.dtype == 'object':  # If target variable is categorical
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Convert to one-hot encoding for multi-class classification
        if len(set(y)) > 2:
            y = to_categorical(y)

    return X, y
