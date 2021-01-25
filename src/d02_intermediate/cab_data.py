import os
import pandas as pd
import sys
from geopy.distance import distance
import numpy as np

# setting src package
src_dir = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_dir)

from d00_utils.logging import (error_catching,
                               log_wrapper
                               )


def prepare_data(cab_data):
    """ Prepares the cab data for the calculation and modelling step. """
    prepared_cab_data = (
        cab_data
        .pipe(format_data)
        .pipe(add_features)
    )

    return prepared_cab_data


def format_data(cab_data):
    """ Performs various formats of the cab data. """

    time_column = 'time'
    sort_columns = ['cab_id', time_column]

    try:
        formatted_cab_data = (
            cab_data
            .pipe(_convert_unix_epoch_to_date, time_column)
            .pipe(_sort_data, sort_columns)
        )
    except Exception as error:
        error_catching(error)

    return formatted_cab_data


@log_wrapper
def _convert_unix_epoch_to_date(cab_data, time_column):
    """ Converts the unix epoch date to a datetime format. """

    cab_data_w_date = cab_data.copy()
    cab_data_w_date.loc[:, time_column] = pd.to_datetime(cab_data.loc[:, time_column], unit='s')

    return cab_data_w_date


@log_wrapper
def _sort_data(cab_data, sort_columns):
    return cab_data.sort_values(by=sort_columns)


def add_features(cab_data):
    """ Adds features to the cab data. """

    try:
        featured_cab_data = (
            cab_data
            .pipe(_add_distances)
            .pipe(_add_hours)
            .pipe(_add_velocities)
            .pipe(_label_velocity_outliers, velocity_threshold=85)
        )
    except Exception as error:
        error_catching(error)

    return featured_cab_data


@log_wrapper
def _add_distances(cab_data):
    """ Adds point to point distance to the cab data frame. """
    # TODO: convert calculation to class (hour and distance) and refactor code for a single function

    # initial variables definition
    previous_location = None
    previous_cab = None
    distances = []

    # evaluation cycle
    for row in cab_data.itertuples(index=False):

        # update current location and cab
        current_location = (row.latitude, row.longitude)
        current_cab = row.cab_id

        # distance calculation
        if (not previous_location or
                previous_cab != current_cab):
            distances.append(0)
        else:
            distance = _calculate_distance_in_miles(previous_location, current_location)
            distances.append(distance)

        # update previous location and cab variables
        previous_location = (row.latitude, row.longitude)
        previous_cab = row.cab_id

    # adds distance column to the cab data
    cab_data_w_distances = cab_data.copy()
    cab_data_w_distances.loc[:, 'distance'] = distances

    return cab_data_w_distances


def _calculate_distance_in_miles(initial_location, final_location):
    """ Returns the distance in miles between two geolocation tuples (lat, lon). """
    return distance(initial_location, final_location).miles


@log_wrapper
def _add_hours(cab_data):
    """ Adds the hour between two points in time to the cab data frame. """
    # TODO: convert calculation to class (hour and distance) and refactor code for a single function

    # initial variables definition
    previous_time = None
    previous_cab = None
    hours = []

    # evaluation cycle
    for row in cab_data.itertuples(index=False):

        # update current time and cab
        current_time = row.time
        current_cab = row.cab_id

        # hour calculation
        if (not previous_time or
                previous_cab != current_cab):
            hours.append(0)
        else:
            hour = _calculate_hours(previous_time, current_time)
            hours.append(hour)

        # update previous time and cab
        previous_time = row.time
        previous_cab = row.cab_id

    # adds the hour column to the cab data
    cab_data_w_hours = cab_data.copy()
    cab_data_w_hours.loc[:, 'hour'] = hours

    return cab_data_w_hours


def _calculate_hours(initial_time, end_time):
    """ Returns the number of hours between two points in time. """
    return (end_time - initial_time) / np.timedelta64(1, 'h')


@log_wrapper
def _add_velocities(cab_data):
    """ Adds the current velocity to the cab data. """

    # adds velocity column to cab data
    cab_data_w_velocity = cab_data.copy()
    cab_data_w_velocity.loc[:, 'velocity'] = cab_data.loc[:, 'distance'] / cab_data.loc[:, 'hour']

    # replaces nan values
    cab_data_w_velocity.fillna(0, inplace=True)

    return cab_data_w_velocity


@log_wrapper
def _label_velocity_outliers(cab_data, velocity_threshold):
    """
        Evaluates the cab velocities and removes all the entries that are above the velocity threshold.
        When entries are removed, an updated calculation of the velocity is made using the previous locatiom
        and time. this velocity is the evaluated and teh cycle repeats.
    """

    # initial variables definition
    previous_location = None
    previous_time = None
    previous_keep = 1
    keep = []

    # evaluation cycle
    for row in cab_data.itertuples(index=False):

        # update current location and time
        current_location = (row.latitude, row.longitude)
        current_time = row.time

        # evaluates it the last value was keep
        # if 1 does nothing, else recalculates the velocity
        if previous_keep == 1:
            velocity = row.velocity
        else:
            point_distance = _calculate_distance_in_miles(previous_location, current_location)
            hour = _calculate_hours(previous_time, current_time)
            velocity = point_distance / hour

        # evaluates if the velocity is below the threshold
        # if True updates all the variables, and if false only updates the markers
        if velocity <= velocity_threshold:
            keep.append(1)
            previous_keep = 1

            previous_location = (row.latitude, row.longitude)
            previous_time = row.time
        else:
            keep.append(-1)
            previous_keep = -1

    # adds the velocity outlier column to the cab data
    cab_data_w_out = cab_data.copy()
    cab_data_w_out.loc[:, 'outlier'] = keep

    return cab_data_w_out


def label_acceleration_outliers():
    pass
