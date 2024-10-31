import pandas as pd
import tarfile
import sys

from ny_taxi.src.utils import CustomException, log_message


def read_csv_from_tar_gz(tar_gz_path: str, csv_filename: str) -> pd.DataFrame:
    """
    Function to read a specific CSV file from a .tar.gz archive.

    Args:
    - tar_gz_path (str): Path to the .tar.gz file.
    - csv_filename (str): Name of the CSV file to read inside the archive.

    Returns:
    - dataframe (DataFrame): pandas DataFrame loaded from the CSV file.
    """
    log_message(f'Attempting to read {csv_filename} from {tar_gz_path}')

    # Open the tar.gz file
    try:
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            # Find the specific CSV file
            csv_file = tar.extractfile(csv_filename)
            if csv_file is None:
                raise FileNotFoundError(f'{csv_filename} not found in the archive.')

            log_message(f'Successfully located {csv_filename} inside {tar_gz_path}')

            # Load the CSV file into a pandas DataFrame
            dataframe = pd.read_csv(csv_file)

            log_message(f'Successfully loaded {csv_filename} into DataFrame')

        return dataframe

    except Exception as e:
        log_message(f'Error occurred while reading {csv_filename}: {str(e)}', 'ERROR')
        raise CustomException(e, sys)


def downcast_numeric(df):
    """
    Downcasts numeric columns in a DataFrame to reduce memory usage.

    Parameters:
    - df: DataFrame with numeric columns.

    Returns:
    - df: Optimized DataFrame with smaller data types.
    """
    for col in df.select_dtypes(include=['float']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
