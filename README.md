# NYC Taxi Trip Duration Prediction
This project is a machine learning pipeline for predicting the duration 
of taxi trips in New York City based on various data features such as 
pickup and drop off coordinates, pickup time, and weather conditions.

## Dataset
The dataset used is sourced from the 
[NYC Taxi Trip Duration Kaggle Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration),
which provides details of trips made by taxis in New York City.
The dataset includes the following key features:
* **id**: Unique identifier for each trip.
* **vendor_id**: Identifier for the provider (taxi company).
* **pickup_datetime**: Timestamp for when the trip started.
* **dropoff_datetime**: Timestamp for when the trip ended.
* **passenger_count**: Number of passengers in the vehicle.
* **pickup_longitude** / **pickup_latitude**: Coordinates for the pickup location.
* **dropoff_longitude** / **dropoff_latitude**: Coordinates for the dropoff location.
* **store_and_fwd_flag**: Whether the trip data was held in the vehicle’s memory before sending to the server.

The goal of the project is to predict the **trip_duration** for each taxi trip using these
features and additional engineered features.

## Project Structure
```plaintext
nyc_taxi/
├── data/                                  # Dataset files here
├── notebooks/                             # Jupyter notebooks folder
│   ├── notebook.ipynb                     # Notebook with EDA and algorithm          
├── nyc_taxi/                              # Source code folder containing src/ 
│   ├── src/                     
│   │   ├── features/                      # Feature engineering scripts
│   │   │   ├── feature_engineering.py     
│   │   ├── loader/                        # Data loading scripts
│   │   │   ├── data_loader.py     
│   │   ├── preprocessor/                  # Data preprocessing scripts
│   │   │   ├── preprocessor.py     
│   │   ├── model/                         # Model training and prediction functions
│   │   │   ├── train_model.py     
├── README.md                              # Project documentation
├── requirements.txt                       # List of dependencies
├── setup.py                               # Package setup script
└── .gitignore                             # Ignore files for git
```

## Installation
To install the package and its dependencies, you can use `setup.py`. This will make the project installable as a package,
enabling you to import its modules directly into any Python code.

1. Clone the Repository:
```bash
git clone https://github.com/heddence/nyc_taxi_predictions.git
cd nyc_taxi_predictions
```
2. Install the Package:
```bash
pip install .
```
This command will install the package along with all dependencies specified in `requirements.txt`.

## Usage
Run the Entire Pipeline: The `main.py` script will load data, preprocess it, engineer features, train the model,
and make predictions.
```bash
python main.py
```

### Example Output
Predictions will be saved to `data/predictions.csv` in the following format:
```plaintext
id,trip_duration
id001,764.3
id002,1352.5
id003,456.1
```

## Algorithm

The algorithm chosen for this project is **XGBoost**, a highly efficient and flexible gradient-boosting framework.
XGBoost is well-suited for this task because of its ability to handle large datasets with complex interactions,
and its performance is generally strong for structured data.

**Model Hyperparameters**: I selected specific parameters to balance accuracy with memory efficiency:

* `max_depth=4`, `min_child_weight=10`: Control tree depth and complexity to manage overfitting.
* `colsample_bytree=0.7`, `subsample=0.5`: Use subsets of data and features to increase model robustness.
* `learning_rate=0.15`: Learning rate for gradient boosting steps.
* `tree_method='hist'`, `grow_policy='depthwise'`: Set XGBoost to use histogram-based, 
depthwise growth to reduce memory load.

## Evaluation
I evaluate the model using **Root Mean Squared Logarithm Error (RMSLE)**, which penalizes larger errors more heavily
and is well-suited for continuous prediction tasks like trip duration.

$\epsilon = \sqrt{\frac{1}{n} \cdot \sum_{i=1}^{n} \left( \log\left( p_i + 1 \right) - \log\left( a_i + 1 \right) \right)^2}$

where

$\epsilon$ is the RMSLE values (score)

$n$ is the total number of observations

$p_i$ is the prediction of trip duration

$a_i$ is the actual of trip duration

### Evaluation Process:
* **Train-Validation Split**: The dataset is split into a training set (80%) and a validation set (20%)
to evaluate model performance before testing.
* **Early Stopping**: During training, the model monitors validation RMSE, stopping if performance doesn’t improve over
a specified number of rounds (50 in this case), preventing overfitting.
* **Final Metrics**: After training, the model is evaluated on the validation set. Validation RMSE provides a reliable 
estimate of model accuracy before applying the model to test data.

## Key Modules

* **Data Loading (`loader/data_loading.py`)**: Loads the dataset, handling compressed files and data integrity checks.
* **Feature Engineering (`features/feature_preprocessing.py`)**: Generates features like Haversine distance, clustering
coordinates, weather conditions, and more.
* **Preprocessing (`preprocessor/preprocessor.py`)**: Handles data cleaning, normalization, and encoding.
* **Model Training and Prediction (`model/train_model.py`)**: Defines XGBoost model parameters, trains the model,
and performs batch predictions.

## Author
Ilia Koiushev