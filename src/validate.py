"""
Anything validation/performance/ranking related goes here.
"""
import warnings
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_score

from train import * # Training functions.


def cross_validate(
        model: BaseEstimator, X, y, cv = 5, scoring = 'accuracy') -> tuple[float, float]:
    """
    Performs cross-validation on a given model.

    :param model: Model object (e.g., a scikit-learn model or a Keras model wrapped in a compatible interface).
    :param X: Features (input data).
    :param y: Target variable (labels).
    :param cv: Number of cross-validation folds (default is 5).
    :param scoring: Scoring metric to evaluate the model (default is 'accuracy').
    :return: Mean and standard deviation of the cross-validation scores.
    """
    X, y = preprocess_neural_net(X, y)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv = cv, scoring = scoring)

    # Return the mean and standard deviation of the cross-validation scores
    return np.mean(scores), np.std(scores)

def rank_models(models: list[Callable[..., BaseEstimator]], X, y, *, incl_params = False):
    """
    Ranks training models by accuracy. Prints out as a table with the model name, parameters, and
    accuracy score along with any other relevant info.

    :param models: List of model training function references that all take in X and y.
    :param X: Features.
    :param y: Classes.
    """
    # warnings.filterwarnings("ignore", category = ConvergenceWarning)
    performances: list[tuple[str, str, float, float, str]] = []
    for model in models:
        built_model = model(X, y)
        accuracy, accuracy_std = cross_validate(built_model, X, y)

        performances.append((
            model.__name__,
            built_model.__class__.__name__,
            accuracy,
            accuracy_std,
            str(built_model.get_params(deep = False)) if incl_params else None
        ))

    # Sort performances by accuracy in ascending order
    performances_sorted = sorted(performances, key=lambda x: x[2])

    # Print out performances as a table
    print(f"{'CV Accuracy':<15} {'CV Accuracy Std':<20} {'Model':<30} {'Model Class':<20}"
          f"{' Params' if incl_params else '':<100}")
    print("=" * (179 if incl_params else 78))
    for model_name, model_class, accuracy, accuracy_std, model_params in performances_sorted:
        print(f"{accuracy:<15.4f} {accuracy_std:<20.4f} {model_name:<30} "
              f"{model_class:<20} {model_params if model_params else '':<100}")
