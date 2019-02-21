import abc
import json
import pandas as pd
import pathlib

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


class SummaryResourceWriter(ForecastWriter):
    def __init__(self, source):
        ''' ForecastWriter which writes a metadata (summary) file and a data (resource) file

        Args:
            source (str):
                A descriptive name for the source of the forecast.  This is often a model name or description.
        '''

        self.source = source

    def write(self, forecast, name, out_path=None):
        if out_path is not None:
            summary_path = pathlib.Path(out_path)
            resource_path = pathlib.Path(out_path)
            summary_path, resource_path = summary_path.resolve(), resource_path.resolve()
            summary_path.mkdir(parents=True, exist_ok=True)
            resource_path.mkdir(parents=True, exist_ok=True)
        else:
            # get output paths from storage module
            summary_path = storage.get(pathlib.Path('predictions/summaries'))
            resource_path = storage.get(pathlib.Path('predictions/resources'))

        summary_filename = summary_path / f'{name}.summary'
        resource_filename = resource_path / f'{name}.resource'

        # contents of the summary file
        summary_dict = {
            'source': self.source,
            'sourcelabel': self.source.replace('_', ' '),
            'site': ','.join(forecast.columns),
            'created': _datestring_to_posix('now'),
            'start': _datestring_to_posix(forecast.first_valid_index()),
            'stop': _datestring_to_posix(forecast.last_valid_index()),
            'resource': str(resource_filename)
        }

        # build resource file
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
            raw_data.append((str(timestamp), *series.values))

        # contents of the resource file
        data_dict = {
            'site': ','.join(forecast.columns),
            'columns': columns,
            'rows': raw_data
        }

        # write the summary file
        with open(summary_filename, 'w') as summary_file:
            json.dump(summary_dict, summary_file, separators=(',', ':'))

        # write the resource file
        with open(resource_filename, 'w') as resource_file:
            json.dump(data_dict, resource_file, separators=(',', ':'))

        return summary_filename, resource_filename


class CommaSeparatedWriter(ForecastWriter):
    ''' ForecastWriter which dumps forecast data to a CSV file '''
    def write(self, forecast, name, out_path=None):
        if out_path is not None:
            output_path = pathlib.Path(out_path).resolve()
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            # get output path from the storage module
            output_path = storage.get(pathlib.Path('predictions/csv'))

        output_filename = output_path / f'{name}.csv'
        forecast.to_csv(output_filename, index_label='reftime')

        return output_filename,
