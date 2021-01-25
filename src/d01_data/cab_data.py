import os
import pandas as pd
import tarfile
import sys

# setting src package
src_dir = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_dir)

from d00_utils.logging import (error_catching,
                               log_wrapper
                               )


def extract_cab_files(folder, file_name):
    """ Extracts the cab info from the indicated tar file. """

    try:
        file_path = os.path.join(folder, file_name)

        file = tarfile.open(file_path)
        file.extractall(path=folder)
        file.close()
    except Exception as error:
        error_catching(error)


@log_wrapper
def aggregate_cab_data(cab_data_folder, file_extension):
    """ Reads all the cab data files and creates a single data frame with cab data. """

    cab_files_paths_list = _get_cab_files_paths_list(cab_data_folder, file_extension)
    cab_data_list = [_get_cab_data(cab_file_path) for cab_file_path in cab_files_paths_list]
    cab_data = pd.concat(cab_data_list)

    return cab_data


def _get_cab_files_paths_list(cab_data_folder, file_extension):
    """ Creates a list with all the cab files paths. """

    cab_files_paths_list = []

    # populates the list with the cabs files paths
    for file in os.listdir(cab_data_folder):
        if file.endswith(file_extension):
            cab_file_path = os.path.join(cab_data_folder, file)
            cab_files_paths_list.append(cab_file_path)

    return cab_files_paths_list


def _get_cab_data(cab_file_path):
    """ Reads a single cab data file. """

    columns_names = ['latitude', 'longitude', 'occupancy', 'time']
    cab_data = pd.read_csv(cab_file_path, sep=' ', header=None, names=columns_names)

    # adds cab id
    cab_id = os.path.splitext(os.path.split(cab_file_path)[1])[0]
    cab_data = cab_data.assign(cab_id=cab_id)

    return cab_data

