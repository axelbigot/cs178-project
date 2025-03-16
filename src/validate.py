"""
Anything validation/performance/ranking related goes here.
"""
from typing import Callable

from train import * # Training functions.


def cross_validate(model, X, y):
    """
    Performs cross-validation on a model.

    I.e. cross_validate(some_model, X, y)

    :param model: Model training function reference that takes in X and y.
    :param X: Features.
    :param y: Classes.
    :return: TBD whatever makes sense. Something that conveys accuracy of the model.
    """
    pass

def rank_models(models, X, y):
    """
    Ranks training models by accuracy. Prints out as a table with the model name, parameters, and
    accuracy score along with any other relevant info.

    :param models: List of model training function references that all take in X and y.
    :param X: Features.
    :param y: Classes.
    """
    pass
