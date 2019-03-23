import json
import pandas as pd
import pathlib
import numpy as np

import apollo.storage as storage
from apollo.datasets.solar import ATHENS_LATLON


def _datestring_to_posix(date_string):
    timestring = pd.to_datetime(date_string, utc=True).timestamp()
    return int(timestring * 1000)  # convert to milliseconds


def _format_date(date_string):
    dt = pd.to_datetime(date_string, utc=True)
    return dt.strftime('%Y-%m-%dT%X')


def write_json(forecast, reftime, source, name, description,
               location=ATHENS_LATLON, out_path=None):
    """ Writes a JSON file including both metadata and forecasted irradiance

    Args:
        forecast (pandas.Dataframe):
            A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
            The index gives the timestamp of the forecast hours, and each
            column corresponds to a target variable being forecast.

        reftime (str or pandas.Timestamp):
            The reference time for the forecast.

        source: (str):
            A descriptive name for the source of the forecast.
            This is often a model name.

        name (str):
            A descriptive name for the forecast being written

        description (str):
            A long (2-3 sentence) description of the forecast.

        location (Tuple[float, float]):
            A tuple (latitude, longitude) specifying the geographic location of
            the forecast.

        out_path (str):
            A destination path for the serialized predictions.
            If None, defaults to the path specified by the APOLLO_DATA
            environment variable.

    Returns:
        str: the path of the json file written to disk

    """
    if out_path is not None:
        output_path = pathlib.Path(out_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # get output paths from storage module
        output_path = storage.get('output/json')

    output_filename = output_path / f'{name}.json'

    # build output file
    columns = [{
        'label': 'TIMESTAMP',
        'units': 'ms',
        'longname': 'The timestamp of the forecasted value, '
                    'expressed as milliseconds since 1970-01-01.',
        'type': 'datetime'
    }]
    # add a column for each forecasted variable
    for column in list(forecast.columns.values):
        columns.append({
            'label': column,
            'units': 'W/m2',
            'longname': f'The predicted irradiance for array {column}.',
            'type': 'number'
        })

    raw_data = []
    for timestamp, series in forecast.iterrows():
        # add tuple (idx, val_1, val_2, . . ., val_n)
        formatted_timestamp = _datestring_to_posix(str(timestamp))
        raw_data.append((formatted_timestamp, *series.values))

    # contents of the output file
    output_dict = {
        'source': source,
        'name': name,
        'description': description,
        'targets': ','.join(forecast.columns),
        'location': location,
        'created': _datestring_to_posix('now'),
        'reftime': _datestring_to_posix(pd.Timestamp(reftime)),
        'start': _datestring_to_posix(forecast.first_valid_index()),
        'stop': _datestring_to_posix(forecast.last_valid_index()),
        'columns': columns,
        'rows': raw_data
    }

    # write the file
    with open(output_filename, 'w') as out_file:
        json.dump(output_dict, out_file, separators=(',', ':'))

    return output_filename


def write_csv(forecast, name, out_path=None):
    """ Writes a CSV file with predicted irradiance values

    Args:
        forecast (pandas.Dataframe):
                A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
                The index gives the timestamp of the forecast hours, and each
                column corresponds to a target variable being forecast.

        name (str):
            A descriptive name for the forecast being written

        out_path (str):
            A destination path for the serialized predictions.
            If None, defaults to the path specified by the APOLLO_DATA
            environment variable.

    Returns:
        str: path to the csv file written to disk

    """
    # add column for unix timestamps (instead of the forecast's DateTimeIndex)
    unix_timestamps = forecast.index.astype(np.int64) // 10 ** 6
    forecast['timestamp'] = unix_timestamps
    # reorder so that timestamp is always the first output column
    reordered_cols = ['timestamp'] + \
                     [col for col in forecast.columns if not col == 'timestamp']
    forecast = forecast.reindex(columns=reordered_cols)
    if out_path is not None:
        output_path = pathlib.Path(out_path).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # get output path from the storage module
        output_path = storage.get('output/csv')

    output_filename = output_path / f'{name}.csv'
    forecast.to_csv(output_filename, index=False)

    return output_filename
