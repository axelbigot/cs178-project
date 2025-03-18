
from ucimlrepo import fetch_ucirepo

from src.preprocess import impute_missing_values, normalize_target_classes
from src.train import *
from src.validate import rank_models

# fetch dataset
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features.drop('education', axis = 1) # Dropping because of feature selection process
y = normalize_target_classes(adult.data.targets)

impute_missing_values(X)

if __name__ == '__main__':
    rank_models([
        # single_layer_nn,
        # triple_layer_nn,
        # # triple_layer_nn_iter, # Very slow and poor performance
        # single_layer_nn_sgd,
        # single_layer_nn_lr,
        # svc_linear,
        # svc_poly,
        # svc_rbf, 0.85
        # svc_sigmoid,
        catboost
    ], X, y)
