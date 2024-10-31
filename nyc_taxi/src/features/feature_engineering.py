import numpy as np
import pandas as pd
import sys
import gc

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from ny_taxi.src.utils import log_message, CustomException


def add_log_trip_duration(train, column='trip_duration'):
    """
    Adds a new column 'log_trip_duration' by taking the natural logarithm
    of the specified column to reduce skewness.

    Parameters:
    - train: The training DataFrame.
    - column: The name of the column to log-transform (default: 'trip_duration').

    Returns:
    - train: The DataFrame with the new 'log_trip_duration' column.
    """
    try:
        log_message(f'Applying log transformation to {column}.')
        train['log_trip_duration'] = np.log(train[column].values + 1)

        log_message('Log transformation complete.')

        return train

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)

def extract_datetime_features(df):
    """
    Extracts time-related features from the 'pickup_datetime' column in a single DataFrame.

    Features extracted:
    - 'pickup_weekday': Day of the week (0 = Monday, 6 = Sunday).
    - 'pickup_week': Week of the year.
    - 'pickup_hour': Hour of the day (0-23).
    - 'pickup_minute': Minute of the hour (0-59).
    - 'pickup_week_hour': Combined hour of the week (weekday * 24 + hour).

    Parameters:
    - df: The DataFrame (either train or test) to extract features from.

    Returns:
    - df: The DataFrame with new time-related features.
    """
    log_message('Extracting time-related features from "pickup_datetime".')
    try:
        # Extract features
        df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
        df['pickup_week'] = df['pickup_datetime'].dt.isocalendar().week
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_minute'] = df['pickup_datetime'].dt.minute
        df['pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']

        log_message('Datetime feature extraction complete.')

        return df

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def extract_datetime_features_for_both(train, test):
    """
    Extracts time-related features from the 'pickup_datetime' column for both
    train and test DataFrames by calling the helper function on both.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.

    Returns:
    - train: The training DataFrame with new time-related features.
    - test: The test DataFrame with new time-related features.
    """
    log_message('Starting datetime feature extraction for both train and test.')
    try:
        # Extract features for both DataFrames
        train = extract_datetime_features(train)
        test = extract_datetime_features(test)

        log_message('Datetime feature extraction for both train and test complete.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def merge_weather_data(train, test, weather_df, weather_column):
    """
    Merges weather data (like temperature, humidity, etc.) into the train and test DataFrames
    based on the 'pickup_date' and 'pickup_hour' columns.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - weather_df: The weather-related DataFrame to merge (e.g., temperature_df).
    - weather_column: The name of the weather column (e.g., 'temperature') to track missing values.

    Returns:
    - train: The merged training DataFrame.
    - test: The merged test DataFrame.
    """
    try:
        log_message(f'Merging {weather_column} data with train and test datasets.')

        # Merge weather data with train
        train = train.merge(weather_df, on=['pickup_date', 'pickup_hour'], how='left')

        # Merge weather data with test
        test = test.merge(weather_df, on=['pickup_date', 'pickup_hour'], how='left')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_extreme_weather_feature(train, test, weather_events):
    """
    Adds a binary 'extreme_weather' column to both the train and test DataFrames,
    indicating whether the trip occurred on an extreme weather event date.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - weather_events: A list or series of extreme weather event dates (in datetime format).

    Returns:
    - train: The modified training DataFrame with the 'extreme_weather' column.
    - test: The modified test DataFrame with the 'extreme_weather' column.
    """
    log_message('Adding extreme weather feature.')
    try:
        # Convert weather events to a set of dates
        weather_events = pd.Series(pd.to_datetime(weather_events, format='%Y-%m-%d')).dt.date

        # Add 'extreme_weather' column to train and test
        train['extreme_weather'] = train['pickup_date'].isin(weather_events).map({True: 1, False: 0})
        test['extreme_weather'] = test['pickup_date'].isin(weather_events).map({True: 1, False: 0})

        log_message('Extreme weather feature added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_holiday_feature(train, test, holidays):
    """
    Adds a binary 'is_holiday' column to both the train and test DataFrames,
    indicating whether the trip occurred on a holiday.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - holidays: A list or series of holiday dates (in datetime format).

    Returns:
    - train: The modified training DataFrame with the 'is_holiday' column.
    - test: The modified test DataFrame with the 'is_holiday' column.
    """
    log_message('Adding holiday feature.')
    try:
        # Convert holidays to a set of dates
        holidays = pd.Series(pd.to_datetime(holidays, format='%Y-%m-%d')).dt.date

        # Add 'is_holiday' column to train and test
        train['is_holiday'] = train['pickup_date'].isin(holidays).map({True: 1, False: 0})
        test['is_holiday'] = test['pickup_date'].isin(holidays).map({True: 1, False: 0})

        log_message('Holiday feature added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance (Haversine distance) between two points on the earth.

    Parameters:
    - lon1 (float): longitude of the 1st point.
    - lat1 (float): latitude of the 1st point.
    - lon2 (float): longitude of the 2nd point.
    - lat2 (float): latitude of the 2nd point.

    Returns:
    - distance (float): The Haversine distance between the two points in kilometers.
    """
    R = 6371.0  # Radius of Earth in kilometers

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Compute the differences between latitudes and longitudes
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # Haversine formula
    a = np.sin(d_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c

    return distance


def add_haversine_distance_feature(train, test):
    """
    Adds a new 'distance_km' column to both train and test DataFrames by calculating
    the Haversine distance between the pickup and dropoff coordinates.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.

    Returns:
    - train: The training DataFrame with the new 'distance_km' column.
    - test: The test DataFrame with the new 'distance_km' column.
    """
    log_message("Calculating Haversine distance for train and test datasets.")
    try:
        # Calculate distance for train
        train['distance_km'] = haversine_distance(train['pickup_longitude'].values,
                                                  train['pickup_latitude'].values,
                                                  train['dropoff_longitude'].values,
                                                  train['dropoff_latitude'].values)

        # Calculate distance for test
        test['distance_km'] = haversine_distance(test['pickup_longitude'].values,
                                                 test['pickup_latitude'].values,
                                                 test['dropoff_longitude'].values,
                                                 test['dropoff_latitude'].values)

        log_message('Haversine distance feature added successfully.')

        # Remove outliers in train based on distance
        log_message('Removing outliers in "distance_km".')
        lower_bound = train['distance_km'].quantile(0.01)
        train = train[(train['distance_km'] >= lower_bound) & (train['distance_km'] <= 56)]

        log_message(f'Outliers removed. Remaining data size: {len(train)} rows.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def manhattan_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Manhattan distance between two points on the earth.

    Parameters:
    - lon1 (float): longitude of the 1st point.
    - lat1 (float): latitude of the 1st point.
    - lon2 (float): longitude of the 2nd point.
    - lat2 (float): latitude of the 2nd point.

    Returns:
    - distance (float): The Manhattan distance between the two points in kilometers.
    """
    R = 6371.0  # Radius of Earth in kilometers

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Compute the differences in coordinates
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1

    # Calculate the Manhattan distance: sum of latitudinal and longitudinal distances
    a_lat = R * np.abs(d_lat)
    a_lon = R * np.abs(d_lon) * np.cos((lat1 + lat2) / 2)

    # Return the Manhattan distance
    return a_lat + a_lon


def add_manhattan_distance_feature(train, test):
    """
    Adds a new 'manhattan_distance_km' column to both train and test DataFrames
    by calculating the Manhattan distance between the pickup and dropoff coordinates.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.

    Returns:
    - train: The training DataFrame with the new 'manhattan_distance_km' column.
    - test: The test DataFrame with the new 'manhattan_distance_km' column.
    """
    log_message('Calculating Manhattan distance for train and test datasets.')
    try:
        # Calculate Manhattan distance for train
        train['manhattan_distance_km'] = manhattan_distance(train['pickup_longitude'].values,
                                                            train['pickup_latitude'].values,
                                                            train['dropoff_longitude'].values,
                                                            train['dropoff_latitude'].values)

        # Calculate Manhattan distance for test
        test['manhattan_distance_km'] = manhattan_distance(test['pickup_longitude'].values,
                                                           test['pickup_latitude'].values,
                                                           test['dropoff_longitude'].values,
                                                           test['dropoff_latitude'].values)

        log_message('Manhattan distance feature added successfully.')

        # Remove outliers in train based on distance
        log_message('Removing outliers in "manhattan_distance_km".')
        lower_bound = train['manhattan_distance_km'].quantile(0.01)
        train = train[(train['manhattan_distance_km'] >= lower_bound) & (train['manhattan_distance_km'] <= 56)]

        log_message(f'Outliers removed. Remaining data size: {len(train)} rows.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def calculate_bearing(lon1, lat1, lon2, lat2):
    """
    Calculate the bearing angle between two points on the earth.

    Parameters:
    - lon1 (float): longitude of the 1st point.
    - lat1 (float): latitude of the 1st point.
    - lon2 (float): longitude of the 2nd point.
    - lat2 (float): latitude of the 2nd point.

    Returns:
    - bearing_deg (np.degrees): Bearing angle in degrees.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Calculate the differences in longitude
    d_lon = lon2 - lon1

    # Calculate the bearing using the formula
    x = np.sin(d_lon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)

    # Calculate the initial bearing angle in radians
    bearing_rad = np.arctan2(x, y)

    # Convert the bearing angle from radians to degrees
    bearing_deg = np.degrees(bearing_rad)

    # Normalize the bearing to 0-360 degrees
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def add_bearing_feature(train, test):
    """
    Adds a new 'bearing_angle' column to both train and test DataFrames by calculating
    the bearing angle between the pickup and dropoff coordinates.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.

    Returns:
    - train: The training DataFrame with the new 'bearing_angle' column.
    - test: The test DataFrame with the new 'bearing_angle' column.
    """
    log_message("Calculating bearing angle for train and test datasets.")
    try:
        # Calculate bearing angle for train
        train['bearing_angle'] = calculate_bearing(train['pickup_longitude'].values,
                                                   train['pickup_latitude'].values,
                                                   train['dropoff_longitude'].values,
                                                   train['dropoff_latitude'].values)

        # Calculate bearing angle for test
        test['bearing_angle'] = calculate_bearing(test['pickup_longitude'].values,
                                                  test['pickup_latitude'].values,
                                                  test['dropoff_longitude'].values,
                                                  test['dropoff_latitude'].values)

        log_message('Bearing angle feature added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_average_speed_and_remove_outliers(train, lower_percentile=0.01, upper_percentile=0.99):
    """
    Adds 'avg_speed_h' and 'avg_speed_m' columns to the train DataFrame by calculating
    the average speed based on Haversine and Manhattan distances in kilometers per hour.
    Then removes outliers based on the 1st and 99th percentiles of 'avg_speed_h'.

    Parameters:
    - train: The training DataFrame.
    - lower_percentile: The lower percentile for the cutoff (default: 0.01).
    - upper_percentile: The upper percentile for the cutoff (default: 0.99).

    Returns:
    - train: The filtered DataFrame with 'avg_speed_h' and 'avg_speed_m' columns added and outliers removed.
    """
    log_message('Calculating average speed and removing outliers.')
    try:
        # Step 1: Calculate average speed in km/h
        train['avg_speed_h'] = train['distance_km'] / (train['trip_duration'] / 3600)
        train['avg_speed_m'] = train['manhattan_distance_km'] / (train['trip_duration'] / 3600)

        log_message('Average speed columns added: "avg_speed_h" and "avg_speed_m".')

        # Step 2: Calculate the lower and upper bounds for 'avg_speed_h'
        lower_bound = train['avg_speed_h'].quantile(lower_percentile)
        upper_bound = train['avg_speed_h'].quantile(upper_percentile)

        log_message(f'Lower bound for speed outliers: {lower_bound} km/h, Upper bound: {upper_bound} km/h.')

        # Step 3: Remove outliers in 'avg_speed_h' and 'avg_speed_m'
        train = train[(train['avg_speed_h'] >= lower_bound) & (train['avg_speed_h'] <= upper_bound)]
        train = train[(train['avg_speed_m'] >= lower_bound) & (train['avg_speed_m'] <= upper_bound)]

        log_message(f'Outliers removed. Remaining data size: {len(train)} rows.')

        return train

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_cluster_features(train, test, n_clusters=100, sample_size=500000, batch_size=10000):
    """
    Performs MiniBatch K-Means clustering on pickup and dropoff coordinates and
    adds 'pickup_cluster' and 'dropoff_cluster' columns to both train and test DataFrames.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - n_clusters: Number of clusters for K-Means (default: 100).
    - sample_size: Number of random points to sample for clustering (default: 500000).
    - batch_size: Batch size for MiniBatch K-Means (default: 10000).

    Returns:
    - train: The training DataFrame with 'pickup_cluster' and 'dropoff_cluster' columns added.
    - test: The test DataFrame with 'pickup_cluster' and 'dropoff_cluster' columns added.
    """
    log_message('Preparing coordinates for clustering.')

    try:
        # Stack coordinates for pickup and dropoff locations
        coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                            train[['dropoff_latitude', 'dropoff_longitude']].values,
                            test[['pickup_latitude', 'pickup_longitude']].values,
                            test[['dropoff_latitude', 'dropoff_longitude']].values))

        # Randomly sample indices
        sample_ind = np.random.permutation(len(coords))[:sample_size]
        sampled_coords = coords[sample_ind]

        # Perform MiniBatch K-Means clustering
        log_message(f'Performing MiniBatch K-Means clustering with {n_clusters} clusters.')
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, n_init='auto').fit(sampled_coords)

        # Add cluster labels to train and test DataFrames
        log_message('Adding cluster labels to train and test datasets.')
        train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']].values)
        train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']].values)
        test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']].values)
        test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']].values)

        log_message('Cluster features added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_pca_features_and_manhattan_distance(train, test, n_components=2):
    """
    Performs PCA on pickup and dropoff coordinates and adds PCA-transformed
    coordinates as features to both train and test DataFrames. Also calculates
    the Manhattan distance between pickup and dropoff points in PCA space.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - n_components: Number of PCA components (default: 2).

    Returns:
    - train: The training DataFrame with PCA features and PCA-based Manhattan distance.
    - test: The test DataFrame with PCA features and PCA-based Manhattan distance.
    """
    log_message('Preparing coordinates for PCA transformation.')

    try:
        # Stack coordinates for pickup and dropoff locations
        coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                            train[['dropoff_latitude', 'dropoff_longitude']].values,
                            test[['pickup_latitude', 'pickup_longitude']].values,
                            test[['dropoff_latitude', 'dropoff_longitude']].values))

        # Fit PCA on the combined coordinates
        log_message(f'Performing PCA with {n_components} components.')
        pca = PCA(n_components=n_components).fit(coords)

        # Transform and add PCA features to train DataFrame
        train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']].values)[:, 0]
        train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']].values)[:, 1]
        train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']].values)[:, 0]
        train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']].values)[:, 1]

        # Transform and add PCA features to test DataFrame
        test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']].values)[:, 0]
        test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']].values)[:, 1]
        test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']].values)[:, 0]
        test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']].values)[:, 1]

        # Calculate Manhattan distance in PCA space
        train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(
            train['dropoff_pca0'] - train['pickup_pca0'])
        test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(
            test['dropoff_pca0'] - test['pickup_pca0'])

        log_message('PCA features and PCA-based Manhattan distance added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_center_coordinates_and_bins(train, test, bin_precision=2):
    """
    Calculates center coordinates between pickup and dropoff points and creates
    latitude and longitude bins for pickup and center locations in both train and test DataFrames.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - bin_precision: Number of decimal places for latitude and longitude binning (default: 2).

    Returns:
    - train: The training DataFrame with center coordinates and bins added.
    - test: The test DataFrame with center coordinates and bins added.
    """
    log_message('Calculating center coordinates between pickup and dropoff points.')

    try:
        # Calculate center coordinates for train
        train['center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
        train['center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

        # Calculate center coordinates for test
        test['center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
        test['center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

        log_message('Creating latitude and longitude bins for pickup and center coordinates.')

        # Create latitude and longitude bins for train
        train['pickup_lat_bin'] = np.round(train['pickup_latitude'], bin_precision)
        train['pickup_lon_bin'] = np.round(train['pickup_longitude'], bin_precision)
        train['center_lat_bin'] = np.round(train['center_latitude'], bin_precision)
        train['center_lon_bin'] = np.round(train['center_longitude'], bin_precision)

        # Create latitude and longitude bins for test
        test['pickup_lat_bin'] = np.round(test['pickup_latitude'], bin_precision)
        test['pickup_lon_bin'] = np.round(test['pickup_longitude'], bin_precision)
        test['center_lat_bin'] = np.round(test['center_latitude'], bin_precision)
        test['center_lon_bin'] = np.round(test['center_longitude'], bin_precision)

        log_message('Center coordinates and bins added successfully.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_group_aggregations(train, test, group_columns):
    """
    Adds group-based average speed and log trip duration features to both train and test DataFrames.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - group_columns: List of columns to group by.

    Returns:
    - train, test: Updated DataFrames with group-based aggregations.
    """
    try:
        for group_col in group_columns:
            log_message(f'Calculating mean values for {group_col}.')

            # Group by column and calculate mean for specified features
            group = train.groupby(group_col)[['avg_speed_h', 'avg_speed_m', 'log_trip_duration']].mean()
            group.columns = [f'{col}_group_{group_col}' for col in group.columns]

            # Merge the grouped results into the train and test datasets
            train = pd.merge(train, group, how='left', left_on=group_col, right_index=True)
            test = pd.merge(test, group, how='left', left_on=group_col, right_index=True)

            log_message(f'Added group-based aggregation features for {group_col}.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_multiple_column_aggregations(train, test, multi_group_columns):
    """
    Adds multi-column-based average speed and trip count features to both train and test DataFrames.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - multi_group_columns: List of column combinations to group by.

    Returns:
    - train, test: Updated DataFrames with multi-column aggregations.
    """
    try:
        for group_cols in multi_group_columns:
            log_message(f'Calculating mean average speed and trip counts for columns: {group_cols}.')

            # Calculate mean speed and count for each grouping of columns
            coord_speed = train.groupby(group_cols)['avg_speed_h'].mean().reset_index()
            coord_count = train.groupby(group_cols)['id'].count().reset_index()

            # Merge speed and count tables, then filter for groups with more than 100 trips
            coord_stats = pd.merge(coord_speed, coord_count, on=group_cols)
            coord_stats = coord_stats[coord_stats['id'] > 100]
            coord_stats.columns = group_cols + [f'avg_speed_h_{"_".join(group_cols)}', f'cnt_{"_".join(group_cols)}']

            del coord_speed, coord_count
            gc.collect()

            # Merge aggregated statistics into train and test datasets
            train = pd.merge(train, coord_stats, how='left', on=group_cols)
            test = pd.merge(test, coord_stats, how='left', on=group_cols)

            del coord_stats
            gc.collect()

            log_message(f'Added multi-column aggregation features for columns: {group_cols}.')

        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_rolling_trip_count(train, test, group_freq='60min'):
    """
    Adds a rolling trip count feature based on a specified time interval to both train and test DataFrames.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - group_freq: The frequency for the rolling operation (default: '60min').

    Returns:
    - train, test: Updated DataFrames with rolling trip counts.
    """
    try:
        log_message(f'Calculating rolling trip count with frequency: {group_freq}.')

        # Concatenate train and test, keeping relevant columns
        df_combined = pd.concat((train, test))[['id', 'pickup_datetime', 'pickup_cluster', 'dropoff_cluster']]

        # Group by time interval frequency
        train['pickup_datetime_group'] = train['pickup_datetime'].dt.round(group_freq)
        test['pickup_datetime_group'] = test['pickup_datetime'].dt.round(group_freq)

        # Set datetime as index and sort by datetime
        df_counts = df_combined.set_index('pickup_datetime')[['id']].sort_index()

        # Apply rolling count for specified time frequency
        df_counts['count_60min'] = df_counts.isnull().rolling(group_freq).count()['id']

        # Merge rolling trip counts into train and test
        train = train.merge(df_counts, on='id', how='left')
        test = test.merge(df_counts, on='id', how='left')

        log_message('Added rolling trip count feature.')
        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)


def add_dropoff_cluster_counts(train, test, group_freq='60min', rolling_window='240min', shift_minutes=120):
    """
    Adds a rolling average of trip counts for each dropoff cluster by time interval.

    Parameters:
    - train: The training DataFrame.
    - test: The test DataFrame.
    - group_freq: The frequency for grouping by time interval (default: '60min').
    - rolling_window: Window size for the rolling average (default: '240min').
    - shift_minutes: Time to shift the interval by (default: 120).

    Returns:
    - train, test: Updated DataFrames with dropoff cluster counts.
    """
    log_message(
        f'Calculating dropoff cluster counts with a {rolling_window} rolling average'
        f' and a {shift_minutes}-minute shift.')

    try:
        # Concatenate train and test for combined processing
        df_combined = pd.concat((train, test))[['id', 'pickup_datetime', 'dropoff_cluster']]

        # Count dropoffs in clusters by time interval
        dropoff_counts = (
            df_combined.groupby([pd.Grouper(key='pickup_datetime', freq=group_freq), 'dropoff_cluster'])
            .agg({'id': 'count'})
            .reset_index())

        # Calculate rolling average of trip counts per cluster
        dropoff_counts['dropoff_cluster_count'] = (
            dropoff_counts.set_index('pickup_datetime')
            .groupby('dropoff_cluster')['id']
            .rolling(rolling_window).mean()
            .reset_index(drop=True))

        # Shift by specified time interval
        dropoff_counts['pickup_datetime_group'] = dropoff_counts['pickup_datetime'] - pd.Timedelta(minutes=shift_minutes)

        # Merge the new dropoff cluster count features into train and test datasets
        train = train.merge(
            dropoff_counts[['pickup_datetime_group', 'dropoff_cluster', 'dropoff_cluster_count']],
            on=['pickup_datetime_group', 'dropoff_cluster'], how='left')
        train['dropoff_cluster_count'] = train['dropoff_cluster_count'].fillna(0)

        test = test.merge(
            dropoff_counts[['pickup_datetime_group', 'dropoff_cluster', 'dropoff_cluster_count']],
            on=['pickup_datetime_group', 'dropoff_cluster'], how='left')
        test['dropoff_cluster_count'] = test['dropoff_cluster_count'].fillna(0)

        log_message('Added dropoff cluster count features.')
        return train, test

    except Exception as e:
        log_message(f'An unknown error has occurred: {e}', 'ERROR')
        raise CustomException(e, sys)
