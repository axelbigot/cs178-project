"""
Training models go here

Accept X, y as parameters to be called in main.
All validation and evaluation goes in validate.py
"""
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

from preprocess import preprocess_neural_net


def neural_net(X, y):
    """
    Builds and trains a neural network for classification.

    Parameters:
    X: pandas DataFrame or numpy array, features
    y: pandas DataFrame, target variable (labels)

    Returns:
    model: Trained neural network model
    history: Training history object containing loss and accuracy for each epoch
    """

    # Preprocess data: Encode and scale the features and target variable
    X, y = preprocess_neural_net(X, y)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Define the neural network architecture
    model = Sequential()

    # Input layer (assuming X has numeric data)
    model.add(Dense(128, input_dim = X_train.shape[1], activation = 'relu'))

    # Hidden layers
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))

    # Output layer (using softmax for multi-class classification)
    if y.ndim == 1 or y.shape[1] == 1:  # For binary classification (1D or single-column array)
        model.add(Dense(1, activation = 'sigmoid'))
    else:  # For multi-class classification (more than 2 classes)
        model.add(Dense(y.shape[1], activation = 'softmax'))

    # Compile the model
    model.compile(optimizer = Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    # Train the model and capture history
    history = model.fit(X_train, y_train, epochs = 10, batch_size = 32,
                        validation_data = (X_val, y_val), verbose = 1)

    # Output the training history (loss and accuracy over epochs)
    print("Training complete.")
    print("Final Training Accuracy:", history.history['accuracy'][-1])
    print("Final Validation Accuracy:", history.history['val_accuracy'][-1])

    # Plot the training and validation loss and accuracy
    import matplotlib.pyplot as plt
    plt.figure(figsize = (12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Training Accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Training Loss')
    plt.plot(history.history['val_loss'], label = 'Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    return model, history
