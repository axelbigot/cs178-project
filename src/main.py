
from ucimlrepo import fetch_ucirepo

from src.preprocess import impute_missing_values, normalize_target_classes
from src.train import neural_net
from src.validate import rank_models


# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = normalize_target_classes(adult.data.targets)

impute_missing_values(X)

if __name__ == '__main__':
    model = neural_net(X, y)[0]
    rank_models([model], X, y)
