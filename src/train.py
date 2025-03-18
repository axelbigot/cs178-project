"""
Training models go here

Accept X, y as parameters to be called in main.
All validation and evaluation goes in validate.py
"""
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from preprocess import preprocess_neural_net


_SEED = 1234

def _neural_net(X, y, **kwargs):
    """
    Create a MLP neural network using scikit-learn.

    :param X: Features.
    :param y: Labels.
    :param params: Parameters.
    :return: Trained scikit-learn MLP model.
    """
    # Preprocess the data
    X, y = preprocess_neural_net(X, y)

    # Define MLP classifier
    model = MLPClassifier(
        **kwargs,
        random_state = _SEED
    )

    # Train the model
    model.fit(X, y)

    return model

def single_layer_nn(X, y):
    return _neural_net(
        X, y,
        max_iter=10
    )

def triple_layer_nn(X, y):
    return _neural_net(
        X, y,
        hidden_layer_sizes=(128, 64, 32),
        max_iter=10
    )

def triple_layer_nn_iter(X, y):
    return _neural_net(
        X, y,
        hidden_layer_sizes=(128, 64, 32),
    )

def single_layer_nn_lr(X, y):
    return _neural_net(
        X, y,
        max_iter=10,
        learning_rate_init=0.01
    )

def single_layer_nn_sgd(X, y):
    return _neural_net(
        X, y,
        max_iter=10,
        solver='sgd'
    )

def svc_rbf(X, y):
    X, y = preprocess_neural_net(X, y)
    model = SVC(
        random_state = _SEED,
        kernel = 'rbf'
    )

    return model.fit(X, y)

def svc_linear(X, y):
    X, y = preprocess_neural_net(X, y)
    model = SVC(
        random_state = _SEED,
        kernel = 'linear'
    )

    return model.fit(X, y)

def svc_poly(X, y):
    X, y = preprocess_neural_net(X, y)
    model = SVC(
        random_state = _SEED,
        kernel = 'poly'
    )

    return model.fit(X, y)

def svc_sigmoid(X, y):
    X, y = preprocess_neural_net(X, y)
    model = SVC(
        random_state = _SEED,
        kernel = 'sigmoid'
    )

    return model.fit(X, y)

