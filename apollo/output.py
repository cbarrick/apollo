import abc
import pandas as pd


class ForecastWriter(abc.ABC):
    ''' Abstract base class for objects capable of serializing Apollo forecasts to a file
    '''
    @abc.abstractmethod
    def write(self, forecast):
        ''' Write a forecast

        Args:
            forecast (pandas.Dataframe):
                A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
                The index gives the timestamp of the forecast hours, and each
                column corresponds to a target variable being forecast.

        Returns:
            list of str:
                The full paths of files that were written to disk

        '''
        pass
