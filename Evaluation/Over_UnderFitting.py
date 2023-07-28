import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_overfitting(model, X_train, y_train, X_test, y_test):
    """
    Evaluate overfitting for the given model.

    Parameters:
        model (object): The trained model.
        X_train (DataFrame): Training data features.
        y_train (Series): Training data target.
        X_test (DataFrame): Test data features.
        y_test (Series): Test data target.

    Returns:
        tuple: Mean Squared Error (MSE) for training and test datasets.
    """
    # Train the model on the training data
    model.fit(X_train, y_train)

    # Predict on training and test datasets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate MSE for training and test datasets
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    return mse_train, mse_test

def evaluate_underfitting(models, X_train, y_train, X_test, y_test):
    """
    Evaluate underfitting for the given models.

    Parameters:
        models (list): List of model objects to evaluate.
        X_train (DataFrame): Training data features.
        y_train (Series): Training data target.
        X_test (DataFrame): Test data features.
        y_test (Series): Test data target.

    Returns:
        dict: Dictionary containing Mean Squared Error (MSE) for each model on the training and test datasets.
    """
    results = {}

    for model in models:
        model_name = model.__class__.__name__
        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on training and test datasets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Calculate MSE for training and test datasets
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)

        results[model_name] = {'MSE_train': mse_train, 'MSE_test': mse_test}

    return results


#models_to_evaluate = [LinearRegression()]
#results = evaluate_underfitting(models_to_evaluate, X_train, y_train, X_test, y_test)
