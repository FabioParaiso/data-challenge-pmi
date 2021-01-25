import os
import sys
import h3
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
import math

# setting src package
src_dir = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_dir)

from d00_utils.logging import (error_catching,
                               log_wrapper
                               )


def get_passenger_pickup_locations(cab_data):
    """ Gets the locations were a client was picked up. """

    passenger_pickup_locations = (
        cab_data
        .assign(previous_occupancy=cab_data.groupby('cab_id')['occupancy'].shift(1))
        .query('previous_occupancy == 0 and occupancy == 1')
    )

    return passenger_pickup_locations


def add_h3_hexes(cab_data, aperture_size):
    """ Adds the h3 hex location designation to the cab data frame. """

    cab_data_w_h3_hex = (
        cab_data
        .assign(h3_hex=cab_data.apply(lambda x: h3.geo_to_h3(x.latitude, x.longitude, aperture_size), 1))
    )

    return cab_data_w_h3_hex


def hex_selection(df, min_pickups):
    """ Selects the hexes that have a minimum number of pickups. """

    selected_hexes = (
        df
        .groupby('h3_hex')
        .agg(n=('h3_hex', np.size))
        .query(f'n > {min_pickups}')
        .sort_values(by='n', ascending=False)
        .reset_index()
    )

    return selected_hexes



