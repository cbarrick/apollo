import abc
import json
import pandas as pd
import pathlib
import numpy as np

import apollo.storage as storage


def _datestring_to_posix(date_string):
    timestring = pd.to_datetime(date_string, utc=True).timestamp()
    return int(timestring * 1000)  # convert to milliseconds


def _format_date(date_string):
    dt = pd.to_datetime(date_string, utc=True)
    return dt.strftime('%Y-%m-%dT%X')


class ForecastWriter(abc.ABC):
    ''' Abstract base class for objects capable of serializing Apollo forecasts to a file
    '''
    @abc.abstractmethod
    def write(self, forecast, name, out_path=None):
        ''' Write a forecast

        Args:
            forecast (pandas.Dataframe):
                A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
                The index gives the timestamp of the forecast hours, and each
                column corresponds to a target variable being forecast.

            name (str):
                A descriptive name for the forecast being written

            out_path (str):
                A destination path for the serialized predictions.
                If None, defaults to the path specified by the APOLLO_DATA environment variable.

        Returns:
            tuple of str:
                The full paths of files that were written to disk

        '''
        pass


class JsonWriter(ForecastWriter):
    def __init__(self, source):
        ''' ForecastWriter which writes a JSON file including both metadata and forecasted values

        The output file has the following format:

        {
           "source":"rf_test",
           "sourcelabel":"rf test",
           "site":"UGABPAO1IRR",
           "created":1550736650906,
           "start":1510142400000,
           "stop":1510228800000,
           "columns":[
              {
                 "label":"TIMESTAMP",
                 "units":"",
                 "longname":"",
                 "type":"datetime"
              },
              {
                 "label":"UGA-C-POA-1-IRR",
                 "units":"w/m2",
                 "longname":"",
                 "type":"number"
              }
           ],
           "rows":[
              [
                 "2017-11-08 12:00:00",
                 6.065183055555549
              ],
              ...
              [
                 "2017-12-31 18:00:00",
                 0.41544999999999954
              ]
           ]
        }

        There may an arbitrary number of columns

        Args:
            source (str):
                A descriptive name for the source of the forecast.  This is often a model name or description.
        '''

        self.source = source

    def write(self, forecast, name, out_path=None):
        if out_path is not None:
            output_path = pathlib.Path(out_path).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            # get output paths from storage module
            output_path = storage.get(pathlib.Path('predictions/json'))

        output_filename = output_path / f'{name}.json'

        # build output file
        columns = [{
            'label': 'TIMESTAMP',
            'units': '',
            'longname': '',
            'type': 'datetime'
        }]
        # add a column for each forecasted variable
        for column in list(forecast.columns.values):
            columns.append({
                'label': column,
                'units': 'w/m2',
                'longname': '',
                'type': 'number'
            })

        raw_data = []
        for timestamp, series in forecast.iterrows():
            # add tuple (idx, val_1, val_2, . . ., val_n)
            formatted_timestamp = _datestring_to_posix(str(timestamp))
            raw_data.append((formatted_timestamp, *series.values))

        # contents of the output file
        output_dict = {
            'source': self.source,
            'sourcelabel': self.source.replace('_', ' '),
            'site': ','.join(forecast.columns),
            'created': _datestring_to_posix('now'),
            'start': _datestring_to_posix(forecast.first_valid_index()),
            'stop': _datestring_to_posix(forecast.last_valid_index()),
            'columns': columns,
            'rows': raw_data
        }

        # write the file
        with open(output_filename, 'w') as out_file:
            json.dump(output_dict, out_file, separators=(',', ':'))

        return output_filename,


class CommaSeparatedWriter(ForecastWriter):
    ''' ForecastWriter which dumps forecast data to a CSV file '''
    def write(self, forecast, name, out_path=None):
        # add column for unix timestamps (instead of the forecast's DateTimeIndex)
        unix_timestamps = forecast.index.astype(np.int64) // 10 ** 6
        forecast['timestamp'] = unix_timestamps
        # reorder so that timestamp is always the first output column
        reordered_cols = ['timestamp'] + [col for col in forecast.columns if not col == 'timestamp']
        forecast = forecast.reindex(columns=reordered_cols)
        if out_path is not None:
            output_path = pathlib.Path(out_path).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            # get output path from the storage module
            output_path = storage.get(pathlib.Path('predictions/csv'))

        output_filename = output_path / f'{name}.csv'
        forecast.to_csv(output_filename, index=False)

        return output_filename,
