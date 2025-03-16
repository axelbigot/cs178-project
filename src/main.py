
from ucimlrepo import fetch_ucirepo

from src.preprocess import impute_missing_values, normalize_target_classes


# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = normalize_target_classes(adult.data.targets)

impute_missing_values(X)

if __name__ == '__main__':
    pass
