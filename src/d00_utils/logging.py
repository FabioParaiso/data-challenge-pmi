import logging
from datetime import datetime as dt
import sys

# definition of the logging message
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')


def log_wrapper(f):
    """ Decoder to control function output data structure.

        :params:
        f: function being controlled
    """

    def wrapper(dataf, *args, **kwargs):
        # runs the function and controls the  time
        start = dt.now()
        result = f(dataf, *args, **kwargs)
        end = dt.now()

        # logs the data structure
        logging.info(f"{f.__name__} took {end - start} shape={result.shape} null_values={result.isnull().sum().sum()})")

        return result

    return wrapper


def error_catching(error_msg):
    logging.warning(error_msg)
    sys.exit(1)
