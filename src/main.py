
from ucimlrepo import fetch_ucirepo

from src.preprocess import impute_missing_values, normalize_target_classes
from src.train import *
from src.validate import rank_models


# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = normalize_target_classes(adult.data.targets)

X, y = preprocess(X, y)

if __name__ == '__main__':
    rank_models([
        single_layer_nn,
        triple_layer_nn,
        # triple_layer_nn_iter, # Very slow and poor performance
        single_layer_nn_sgd,
        single_layer_nn_lr,
        alpha_nn,
        mass_layer_nn
    ], X, y)
