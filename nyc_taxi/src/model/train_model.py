import numpy as np
import xgboost as xgb
import os
import sys

from sklearn.model_selection import train_test_split

from ny_taxi.src.utils import log_message, CustomException


def prepare_data_for_training(train, exclude_columns):
    """
    Prepares features and target variable for model training.

    Parameters:
    - train: DataFrame containing the training data.
    - exclude_columns: List of columns to exclude from features.
    - target_column: The name of the target column (default: 'trip_duration').

    Returns:
    - feature_names: List of feature columns.
    - X: Feature array for training.
    - y: Log-transformed target array for training.
    """
    try:
        feature_names = [col for col in train.columns if col not in exclude_columns]
        log_message(f'Selected {len(feature_names)} features for model training.')

        y = train['log_trip_duration']
        X = train[feature_names].values
        log_message(f'Prepared data for training with {X.shape[0]} samples and {X.shape[1]} features.')

        return feature_names, X, y

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training and validation sets.

    Parameters:
    - X: Feature array.
    - y: Target array.
    - test_size: Proportion of data to use for validation (default: 0.2).
    - random_state: Seed for reproducibility (default: 42).

    Returns:
    - X_train, X_val, y_train, y_val: Training and validation data splits.
    """
    try:
        log_message('Splitting data into training and validation sets.')
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def train_xgb_model(X_train, y_train, X_val, y_val, xgb_params, num_boost_round=60, early_stopping_rounds=50):
    """
    Trains an XGBoost model with specified parameters and early stopping.

    Parameters:
    - X_train, y_train: Training feature and target arrays.
    - X_val, y_val: Validation feature and target arrays.
    - xgb_params: Dictionary of XGBoost parameters.
    - num_boost_round: Maximum number of boosting rounds (default: 60).
    - early_stopping_rounds: Number of rounds with no improvement before stopping (default: 50).

    Returns:
    - model: Trained XGBoost model.
    """
    try:
        log_message('Initializing DMatrix for training and validation data.')
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        log_message('Starting model training with XGBoost.')
        model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round, evals=watchlist,
                          early_stopping_rounds=early_stopping_rounds, maximize=False, verbose_eval=10)
        log_message('Model training complete.')

        return model

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def predict_in_chunks(model, test, feature_names, chunk_size=10000):
    """
    Makes predictions on the test set in chunks to reduce memory usage.

    Parameters:
    - model: Trained XGBoost model.
    - test: DataFrame containing the test data.
    - feature_names: List of feature columns used for prediction.
    - chunk_size: Number of rows per chunk (default: 10000).

    Returns:
    - predictions: Array of predictions for the entire test set.
    """
    predictions = []
    num_chunks = int(np.ceil(len(test) / chunk_size))
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(test))
        chunk = test[feature_names].iloc[start:end].values
        dtest_chunk = xgb.DMatrix(chunk)
        chunk_predictions = model.predict(dtest_chunk)
        predictions.extend(chunk_predictions)
        del dtest_chunk  # Free memory after processing each chunk
    return np.array(predictions)


def make_predictions(model, test, feature_names, data_path, file_name='predictions.csv', chunk_size=10000):
    """
    Makes predictions on the test set, applies exponential transformation, and saves results.

    Parameters:
    - model: Trained XGBoost model.
    - test: DataFrame containing the test data.
    - feature_names: List of feature columns used for prediction.
    - data_path: Directory to save the prediction file.
    - file_name: Name of the prediction file (default: 'predictions.csv').

    Returns:
    - predictions_path: Full path where predictions are saved.
    """
    try:
        log_message('Preparing test data for prediction.')
        y_test = predict_in_chunks(model, test, feature_names, chunk_size=chunk_size)

        # Convert back from log-transformed predictions
        test['trip_duration'] = np.exp(y_test) - 1

        # Save predictions
        predictions_path = os.path.join(data_path, file_name)
        test[['id', 'trip_duration']].to_csv(predictions_path, index=False)
        log_message(f'Predictions saved to {predictions_path}')

        return predictions_path

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)
