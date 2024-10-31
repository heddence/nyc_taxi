import os
import sys
import gc
import numpy as np
import xgboost as xgb

from ny_taxi.src.loader.data_loader import (
    read_csv_from_tar_gz,
    downcast_numeric
)
from ny_taxi.src.preprocessor.preprocessor import (
    preprocess_datetime_and_flags,
    remove_outliers,
    filter_passenger_count,
    process_weather_data
)
from ny_taxi.src.features.feature_engineering import (
    add_log_trip_duration,
    extract_datetime_features_for_both,
    merge_weather_data,
    add_extreme_weather_feature,
    add_holiday_feature,
    add_haversine_distance_feature,
    add_manhattan_distance_feature,
    add_bearing_feature,
    add_average_speed_and_remove_outliers,
    add_cluster_features,
    add_pca_features_and_manhattan_distance,
    add_center_coordinates_and_bins,
    add_group_aggregations,
    add_multiple_column_aggregations,
    add_rolling_trip_count,
    add_dropoff_cluster_counts
)
from model.train_model import (
    prepare_data_for_training,
    split_data,
    train_xgb_model,
    make_predictions
)
from ny_taxi.src.utils import log_message, CustomException

def main():
    """
    Main function to orchestrate the loading, preprocessing, and feature engineering
    for the NYC Taxi Prediction project.
    """
    try:
        print('Starting NYC Taxi Prediction project.')
        log_message('Starting NYC Taxi Prediction project.')

        # STEP 1: Define file paths
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_dir, 'data')

        train_path = os.path.join(data_dir, 'train.tar.gz')
        test_path = os.path.join(data_dir, 'test.tar.gz')

        humidity_path = os.path.join(data_dir, 'humidity.tar.gz')
        pressure_path = os.path.join(data_dir, 'pressure.tar.gz')
        temperature_path = os.path.join(data_dir, 'temperature.tar.gz')
        wind_direction_path = os.path.join(data_dir, 'wind_direction.tar.gz')
        wind_speed_path = os.path.join(data_dir, 'wind_speed.tar.gz')

        # STEP 2: Load the data
        print(f'Loading data from {train_path} and {test_path}.')
        log_message(f'Loading data from {train_path} and {test_path}.')
        train = read_csv_from_tar_gz(train_path, 'train.csv')
        test = read_csv_from_tar_gz(test_path, 'test.csv')

        log_message('Down casting numeric columns.')
        train = downcast_numeric(train)
        test = downcast_numeric(test)

        print('Loading weather data.')
        log_message('Loading weather data.')
        humidity = (read_csv_from_tar_gz(humidity_path, 'humidity.csv')[['datetime', 'New York']]
                    .rename(columns={'New York': 'humidity'}))
        pressure = (read_csv_from_tar_gz(pressure_path, 'pressure.csv')[['datetime', 'New York']]
                    .rename(columns={'New York': 'pressure'}))
        temperature = (read_csv_from_tar_gz(temperature_path, 'temperature.csv')[['datetime', 'New York']]
                       .rename(columns={'New York': 'temperature'}))
        wind_direction = (read_csv_from_tar_gz(wind_direction_path, 'wind_direction.csv')[['datetime', 'New York']]
                          .rename(columns={'New York': 'wind_direction'}))
        wind_speed = (read_csv_from_tar_gz(wind_speed_path, 'wind_speed.csv')[['datetime', 'New York']]
                      .rename(columns={'New York': 'wind_speed'}))

        weather_dataframes = {
            'temperature_df': temperature,
            'humidity_df': humidity,
            'wind_speed_df': wind_speed,
            'wind_direction_df': wind_direction,
            'pressure_df': pressure
        }

        # STEP 3: Preprocess the data
        print('Preprocessing the data.')
        log_message('Preprocessing the data.')
        train, test = preprocess_datetime_and_flags(train, test)
        train = remove_outliers(train, column='trip_duration')
        train = filter_passenger_count(train)

        # Process each weather DataFrame
        for name, df in weather_dataframes.items():
            weather_dataframes[name] = process_weather_data(df)

        # STEP 4: Perform feature engineering
        print('Applying feature engineering.')
        log_message('Applying feature engineering.')

        # Log trip duration and datatime features
        train = add_log_trip_duration(train, column='trip_duration')
        train, test = extract_datetime_features_for_both(train, test)

        # Merge weather data to train and test
        train, test = merge_weather_data(train, test, weather_dataframes['temperature_df'], 'temperature')
        train, test = merge_weather_data(train, test, weather_dataframes['humidity_df'], 'humidity')
        train, test = merge_weather_data(train, test, weather_dataframes['wind_speed_df'], 'wind_speed')
        train, test = merge_weather_data(train, test, weather_dataframes['wind_direction_df'], 'wind_direction')
        train, test = merge_weather_data(train, test, weather_dataframes['pressure_df'], 'pressure')

        # Delete weather dataframes from memory
        del weather_dataframes
        del humidity
        del pressure
        del temperature
        del wind_speed
        del wind_direction
        gc.collect()

        # Define weather events and holidays
        weather_events = ['2016-01-10', '2016-01-13', '2016-01-17', '2016-01-23',
                          '2016-02-05', '2016-02-08', '2016-02-15', '2016-02-16',
                          '2016-02-24', '2016-02-25', '2016-03-14', '2016-03-15',
                          '2016-03-21', '2016-03-28', '2016-03-29', '2016-04-03',
                          '2016-04-04', '2016-05-30', '2016-06-28']

        holidays = ['2016-01-01', '2016-01-18', '2016-02-12', '2016-02-15',
                    '2016-05-08', '2016-05-30', '2016-06-19']

        # Add extreme weather and holiday features
        train, test = add_extreme_weather_feature(train, test, weather_events)
        train, test = add_holiday_feature(train, test, holidays)

        del weather_events, holidays
        gc.collect()

        # Add distance features
        train, test = add_haversine_distance_feature(train, test)
        train, test = add_manhattan_distance_feature(train, test)
        train, test = add_bearing_feature(train, test)

        # Speed, clustering, PCA, coordinates features
        train = add_average_speed_and_remove_outliers(train)
        train, test = add_cluster_features(train, test)
        train, test = add_pca_features_and_manhattan_distance(train, test)
        train, test = add_center_coordinates_and_bins(train, test)

        # Add group-based aggregations
        group_columns = ['pickup_hour', 'pickup_date', 'pickup_week_hour', 'pickup_cluster', 'dropoff_cluster']
        train, test = add_group_aggregations(train, test, group_columns)

        # Add multiple-column aggregations
        multi_group_columns = [
            ['center_lat_bin', 'center_lon_bin'],
            ['pickup_hour', 'center_lat_bin', 'center_lon_bin'],
            ['pickup_hour', 'pickup_cluster'],
            ['pickup_hour', 'dropoff_cluster'],
            ['pickup_cluster', 'dropoff_cluster']
        ]
        train, test = add_multiple_column_aggregations(train, test, multi_group_columns)

        # Add rolling trip count
        train, test = add_rolling_trip_count(train, test)

        # Add dropoff cluster counts with rolling average and time shift
        train, test = add_dropoff_cluster_counts(train, test, group_freq='60min',
                                                 rolling_window='240min', shift_minutes=120)

        # STEP 5: Model training
        # Define columns to exclude from features
        do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime',
                                   'trip_duration', 'check_trip_duration', 'pickup_date', 'dropoff_date',
                                   'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_lon_bin',
                                   'center_lat_bin', 'center_lon_bin', 'pickup_datetime_group']

        # Prepare data for training
        feature_names, X, y = prepare_data_for_training(train, do_not_use_for_training)
        print(f'Data preparation complete. Using {len(feature_names)} features for training.')
        del train
        gc.collect()

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)
        print(f'Data split complete: {X_train.shape[0]} samples in training, {X_val.shape[0]} in validation.')

        # Set XGBoost parameters
        xgb_params = {
            'min_child_weight': 10, 'colsample_bytree': 0.7, 'max_depth': 6,
            'subsample': 0.6, 'reg_lambda': 3, 'learning_rate': 0.15,
            'nthread': -1, 'booster': 'gbtree', 'eval_metric': 'rmse'
        }

        # Train the model
        print('Training model...')
        model = train_xgb_model(X_train, y_train, X_val, y_val, xgb_params)

        # Make predictions and save results
        predictions_path = make_predictions(model, test, feature_names, data_dir)

        print(f'Validation mean prediction: {np.exp(model.predict(xgb.DMatrix(X_val))).mean():.3f}')
        print(f'Test mean prediction saved to {predictions_path}')
        log_message(f'Validation mean prediction: {np.exp(model.predict(xgb.DMatrix(X_val))).mean():.3f}')
        log_message(f'Test mean prediction saved to {predictions_path}')

        print('Project completed successfully.')
        log_message('Project completed successfully.')

    except Exception as e:
        log_message(f'An unexpected error occurred: {e}')
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
