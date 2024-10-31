import pandas as pd
import sys

from ny_taxi.src.utils import log_message, CustomException

def preprocess_datetime_and_flags(train: pd.DataFrame, test: pd.DataFrame) -> (
    pd.DataFrame, pd.DataFrame
):
    """
    Preprocesses the datetime columns and converts flags from Y/N to binary values.

    Steps:
    1. Convert pickup and dropoff datetime columns to proper datetime objects.
    2. Extract the date component from pickup and dropoff datetimes.
    3. Convert the 'store_and_fwd_flag' from Y/N to binary values (1/0).

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.

    Returns:
    - train: The modified training DataFrame with new columns and transformations.
    - test: The modified test DataFrame with new columns and transformations.
    """
    log_message('Starting preprocessing of datetime columns and flag conversion.')

    try:
        # Step 1: Convert pickup and dropoff datetime columns to proper datetime objects
        log_message('Converting pickup and dropoff datetime columns to datetime objects.')
        train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
        train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])

        test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

        # Step 2: Extract the date component from pickup and dropoff datetime columns
        log_message('Extracting date components from pickup and dropoff datetimes.')
        train['pickup_date'] = train['pickup_datetime'].dt.date
        train['dropoff_date'] = train['dropoff_datetime'].dt.date

        test['pickup_date'] = test['pickup_datetime'].dt.date

        # Step 3: Convert 'store_and_fwd_flag' from Y/N to binary values (1/0)
        log_message('Converting "store_and_fwd_flag" from Y/N to binary values (1/0).')
        train['store_and_fwd_flag'] = 1 * (train['store_and_fwd_flag'].values == 'Y')
        test['store_and_fwd_flag'] = 1 * (test['store_and_fwd_flag'].values == 'Y')

    except Exception as e:
        log_message(f'Error during preprocessing: {str(e)}', 'ERROR')
        raise CustomException(e, sys)

    log_message('Preprocessing of datetime columns and flag conversion completed.')

    return train, test


def remove_outliers(train, column='trip_duration', lower_quantile=0.01, upper_quantile=0.99):
    """
    Removes outliers from the specified column in the DataFrame by filtering values
    outside the 1st and 99th percentiles.

    Parameters:
    - train: The training DataFrame.
    - column: The name of the column to apply outlier removal (default: 'trip_duration').
    - lower_quantile: The lower quantile for the cutoff (default: 0.01).
    - upper_quantile: The upper quantile for the cutoff (default: 0.99).

    Returns:
    - train: The filtered DataFrame with outliers removed.
    """
    lower_bound = train[column].quantile(lower_quantile)
    upper_bound = train[column].quantile(upper_quantile)

    # Log the bounds for reference
    log_message(f'Outlier removal: {column} lower bound at {lower_bound}, upper bound at {upper_bound}.')

    # Remove outliers
    train_filtered = train[(train[column] >= lower_bound) & (train[column] <= upper_bound)]

    log_message(f'Removed outliers. Original size: {len(train)}, New size: {len(train_filtered)}.')

    return train_filtered


def filter_passenger_count(train, min_passengers=1, max_passengers=7):
    """
    Filters the training DataFrame to include only rows where 'passenger_count'
    is between the specified minimum and maximum values.

    Parameters:
    - train: The training DataFrame.
    - min_passengers: The minimum number of passengers allowed (default: 1).
    - max_passengers: The maximum number of passengers allowed (default: 7).

    Returns:
    - train: The filtered DataFrame with valid passenger counts.
    """
    log_message(f'Filtering data with "passenger_count" between {min_passengers} and {max_passengers}.')

    # Filter rows based on passenger count
    train_filtered = train[
        (train['passenger_count'] >= min_passengers) & (train['passenger_count'] <= max_passengers)]

    log_message(f'Filtering complete. Original size: {len(train)}, New size: {len(train_filtered)}.')

    return train_filtered


def process_weather_data(df, datetime_col='datetime'):
    """
    Processes a weather-related DataFrame by converting the datetime column
    to proper datetime objects and extracting the date and hour components.

    Parameters:
    - df: The weather-related DataFrame to process.
    - datetime_col: The name of the datetime column (default: 'datetime').

    Returns:
    - df: The processed DataFrame with 'pickup_date' and 'pickup_hour' columns.
    """
    try:
        log_message(f'Processing weather data: extracting date and hour from {datetime_col}.')

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df['pickup_date'] = df[datetime_col].dt.date
        df['pickup_hour'] = df[datetime_col].dt.hour

        # Drop the original datetime column
        df = df.drop(columns=[datetime_col])

        log_message('Weather data processing complete.')

        return df

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)
