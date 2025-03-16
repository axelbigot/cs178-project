"""
Anything validation/performance/ranking related goes here.
"""
import numpy as np
from sklearn.model_selection import cross_val_score

from train import * # Training functions.


def cross_validate(model, X, y, cv = 5, scoring = 'accuracy'):
    """
    Performs cross-validation on a given model.

    :param model: Model object (e.g., a scikit-learn model or a Keras model wrapped in a compatible interface).
    :param X: Features (input data).
    :param y: Target variable (labels).
    :param cv: Number of cross-validation folds (default is 5).
    :param scoring: Scoring metric to evaluate the model (default is 'accuracy').
    :return: Mean and standard deviation of the cross-validation scores.
    """
    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv = cv, scoring = scoring)

    # Return the mean and standard deviation of the cross-validation scores
    return np.mean(scores), np.std(scores)

def rank_models(models, X, y):
    """
    Ranks training models by accuracy. Prints out as a table with the model name, parameters, and
    accuracy score along with any other relevant info.

    :param models: List of model training function references that all take in X and y.
    :param X: Features.
    :param y: Classes.
    """
    for model in models:
        print(cross_validate(model, X, y))
