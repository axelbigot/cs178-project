from sklearn.impute import SimpleImputer


def impute_missing_values(X):
    categorical_cols_with_missing = ['workclass', 'occupation', 'native-country']
    imp = SimpleImputer(strategy = 'most_frequent')

    # Replace missing vals with most frequent occurrence of that feature.
    X[categorical_cols_with_missing] = imp.fit_transform(X[categorical_cols_with_missing])

def normalize_target_classes(y):
    # Normalizing some data. Some income fields have trailing period,
    # some don't. Remove trailing period for consistency.
    return y.applymap(lambda v: v.rstrip('.'))
