import os
import sys
import logging
from datetime import datetime


class Config:
    """
    Configuration class for defining project-wide constants.

    Attributes:
    - PROJECT_DIR (str): The root directory of the project.
    - DATA_DIR (str): The directory where data files are stored.
    - LOG_DIR (str): The directory where log files are stored.
    """
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    LOG_DIR = os.path.join(PROJECT_DIR, 'ny_taxi', 'logs')

class CustomException(Exception):
    """
    A custom exception class for handling and logging exceptions.

    Attributes:
    - error_msg (Exception): The error message.
    - error_description (sys): The sys module instance used to extract traceback information.
    """

    def __init__(self, error_msg: Exception, error_description: sys):
        """
        Initializes the CustomException with error details.

        Args:
        - error_msg (Exception): The error message.
        - error_description (sys): The sys module instance used to extract traceback information.
        """
        super().__init__(error_msg)
        try:
            self.error_msg = get_error_details(error_msg, error_description)
        except Exception as e:
            log_message(f'Failed to initialize CustomException: {e}', 'ERROR')
            self.error_msg = 'An error occurred, but the details could not be extracted.'

    def __str__(self) -> str:
        """
        Returns the string representation of the exception.

        Returns:
        - str: The detailed error message.
        """
        return self.error_msg


def setup_logging():
    """
    Sets up the logging configuration for the project. Creates a logs directory if it does not exist.

    Args:
    - None

    Returns:
    - None
    """
    # Ensure the logs directory exists
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # Log file path
    log_file = f'{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log'
    log_file_path = os.path.join(Config.LOG_DIR, log_file)

    # Configure logging
    logging.basicConfig(
        filename=log_file_path,
        format='%(asctime)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s',
        filemode='w',
        level=logging.INFO
    )

    log_message(f'Logging setup complete. Log file: {log_file_path}')


def get_error_details(error_msg: Exception, error_description: sys) -> str:
    """
    Extracts detailed error information including the filename, line number, and error message.

    Args:
    - error_msg (str): The error message.
    - error_description (sys): The sys module, used to extract the traceback information.

    Returns:
    - str: A formatted string containing detailed error information.
    """
    try:
        _, _, exc_tb = error_description.exc_info()
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f'Error occurred in {filename}, line {line_number}, error message: {error_msg}.'
    except Exception as e:
        log_message(f'Failed to extract error details: {e}', 'ERROR')
        return 'Error details could not be extracted.'

def log_message(message, level='INFO'):
    """
    Function to log messages to the file.
    Args:
    - message (str): The message to log.
    - level (str): The log level (INFO, ERROR, DEBUG).
    """
    if level == 'INFO':
        logging.info(message)
    elif level == 'ERROR':
        logging.error(message)
    else:
        logging.debug(message)

# Initialize logging when this module is imported
setup_logging()
