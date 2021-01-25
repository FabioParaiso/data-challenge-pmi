import os
import sys
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


def get_roaming_distance_mean_per_weekday(cab_data):
    """ Calculates the mean and 95% confidence interval of the roaming distance per weekday. """

    try:
        roaming_distance_mean_per_weekday = (
            cab_data
            .pipe(_select_columns, ['occupancy', 'time', 'cab_id', 'distance'])
            .pipe(_select_vacant)
            .pipe(_add_day)
            .pipe(_add_weekday)
            .pipe(_get_total_roaming_distance_per_day_and_cab)
            # .pipe(_remove_outliers, contamination=0.01)
            .pipe(_get_roaming_distance_per_weekday_metrics)
        )
    except Exception as error:
        error_catching(error)

    return roaming_distance_mean_per_weekday


@log_wrapper
def _select_columns(cab_data, columns_names):
    return cab_data[columns_names]


@log_wrapper
def _select_vacant(cab_data):
    """ Selects the cabs that are vacant. """

    vacant_cab_data = (
        cab_data
        .query('occupancy == 0')
        .drop(columns='occupancy')
    )

    return vacant_cab_data


@log_wrapper
def _add_day(cab_data):
    """ Adds the day to the cab data frame. """
    return cab_data.assign(day=lambda x: x.time.dt.strftime('%m%d'))


@log_wrapper
def _add_weekday(cab_data):
    """ Adds the weekday to the cab data frame. """
    return cab_data.assign(weekday=lambda x: x.time.dt.weekday)


@log_wrapper
def _get_total_roaming_distance_per_day_and_cab(cab_data):
    """" Calculates the the total distance a cab roams each day. """

    total_roaming_distance_per_day_and_cab = (
        cab_data
        .groupby(['cab_id', 'day', 'weekday'])
        .agg(day_roam_distance=('distance', np.sum))
        .reset_index()
    )

    return total_roaming_distance_per_day_and_cab


@log_wrapper
def _remove_outliers(df, contamination):
    """ Removes the cabs outliers that have high our low roaming day distances"""

    day_roam_distance = df[['day_roam_distance']]

    clf = IsolationForest(n_estimators=100, contamination=contamination, random_state=0, n_jobs=-1)
    outliers = clf.fit_predict(day_roam_distance)

    inline_data = df.loc[outliers == 1]

    return inline_data


@log_wrapper
def _get_roaming_distance_per_weekday_metrics(df):
    """ Calculates the roaming distance metrics per weekday. """

    roaming_distance_per_weekday_metrics = (
        df
        .groupby('weekday')
        .agg(mean_distance=('day_roam_distance', 'mean'),
             std_distance=('day_roam_distance', 'std'))
        .assign(upper_95_distance=lambda x: x['mean_distance'] + 2 * x['std_distance'],
                lower_95_distance=lambda x: x['mean_distance'] - 2 * x['std_distance'])
        .reset_index()
    )
    return roaming_distance_per_weekday_metrics


def get_year_days_df(year):
    """ Creates a data frame with all the days, weekdays and months of a specific year. """

    year_days_df = (
        _year_days(year)
        .pipe(_add_weekday)
        .pipe(add_month)
    )

    return year_days_df


@log_wrapper
def _year_days(year):
    """ Creates a data frame with all the days of a specific year. """
    return pd.DataFrame({'time': pd.date_range(f'{year}-01-01', f'{year}-12-31')})


@log_wrapper
def add_month(cab_data):
    """ Adds the month to the cab data frame. """
    return cab_data.assign(month=lambda x: x.time.dt.month)


def get_internal_combustion_cabs_per_month(n_cabs, n_months, replacing_rate):
    """ Creates a list with the number of internal combustion cars per month. """

    n_cabs_per_month = []

    for month in range(0, n_months):
        n_cabs_per_month.append(n_cabs)
        n_cabs -= math.ceil(replacing_rate * n_cabs)

    return n_cabs_per_month


def get_annual_co2_emissions(roaming_distance_week_metrics, year_day_list, combustion_cabs_per_month):
    """ Calculates the annual c02 roaming emission for a specific year. """
    annual_co2_emissions = (
        year_day_list
        .pipe(_merge_roaming_distances, roaming_distance_week_metrics)
        .pipe(_calculate_month_roam_distances)
        .pipe(_add_internal_combustion_cabs_per_month, combustion_cabs_per_month)
        .pipe(_calculate_total_fleet_roaming_distances_per_month)
        .pipe(_calculate_yearly_roaming_distance)
        .pipe(_convert_to_co2_emissions, emission_grams_per_mile=404)
        .pipe(_convert_to_ton)
    )

    return annual_co2_emissions


@log_wrapper
def _merge_roaming_distances(df, roaming_distances):
    return df.merge(roaming_distances, left_on='weekday', right_on='weekday', how='left')


@log_wrapper
def _calculate_month_roam_distances(df):
    """ Calculates the roam distance per month. """

    month_df = (
        df
        .groupby('month')
        .sum()
    )

    return month_df


@log_wrapper
def _add_internal_combustion_cabs_per_month(df, combustion_cabs_per_month):
    return df.assign(n_cabs=combustion_cabs_per_month)


@log_wrapper
def _calculate_total_fleet_roaming_distances_per_month(df):
    return df[['mean_distance', 'upper_95_distance', 'lower_95_distance']].mul(df['n_cabs'], axis=0)


def _calculate_yearly_roaming_distance(df):
    return df.sum()


def _convert_to_co2_emissions(df, emission_grams_per_mile):
    df = df * emission_grams_per_mile
    df = df.rename({'mean_distance': 'co2_ton',
                    'upper_95_distance': 'upper_95_co2_ton',
                    'lower_95_distance': 'lower_95_co2_ton'})
    return df


def _convert_to_ton(df):
    return df/1_000_000

