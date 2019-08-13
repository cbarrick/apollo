import logging

import numpy as np
import pandas as pd

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import apollo
from apollo import nam
from apollo.models.base import IrradianceModel


logger = logging.getLogger(__name__)


class NamModel(IrradianceModel):
    '''A concrete irradiance model using NAM forecasts.
    '''

    def __init__(
        self, *,
        features=nam.PLANAR_FEATURES,
        center=nam.ATHENS_LATLON,
        shape=12000,
        **kwargs,
    ):
        '''Initialize a model.

        Keyword Arguments:
            features (list of str):
                The NAM features to use for prediction.
            center (pair of float):
                The center of the geographic area, as a latitude-longited pair.
            shape (float or pair of float):
                The height and width of the geographic area, measured in meters.
                If a scalar, both height and width are the same size.
            **kwargs:
                Forwarded to :class:`IrradianceModel` and :class:`Model`.
        '''
        super().__init__(**kwargs, center=center)
        self.features = list(features)
        self.center = center
        self.shape = shape

    def load_data(self, index, dedupe_strategy='best'):
        '''Load input data for the given times.

        Arguments:
            index (pandas.DatetimeIndex):
                The times to forecast.
            dedupe_strategy (str or int):
                The strategy for selecting between duplicate forecasts.
                **TODO:** Better documentation.

        Returns:
            pandas.DataFrame:
                A data fram indexed by the forecast time.
        '''
        index = apollo.DatetimeIndex(index)
        index = index.floor('1h').unique()

        # Load the xarray data.
        logger.debug('load: loading netcdf')
        reftimes = nam.times_to_reftimes(index)
        data = nam.open(reftimes, on_miss='skip')
        data = data[self.features]
        data = data.astype('float32')

        # Select geographic area.
        logger.debug('load: slicing geographic area')
        data = nam.slice_geo(data, center=self.center, shape=self.shape)

        # Create a data frame.
        # This will have a multi-index of `(reftime, forecast, x, y, *z)`,
        # where `*z` is all of the different z-axes in the dataset.
        logger.debug('load: converting to dataframe')
        data = data.to_dataframe().drop(['lat', 'lon'], axis=1)

        # Replace `reftime` and `forecast` levels with `time`.
        logger.debug('load: reindex by time')
        old_index = data.index
        data = data.set_index('time', append=True)
        data = data.reorder_levels(['time', *old_index.names])
        data = data.reset_index('reftime', drop=True)
        data = data.reset_index('forecast', drop=False)

        # Filter to only the times requested.
        data = data.loc[index]

        # Handle duplicates.
        logger.debug(f'load: selecting forecast hour (dedupe_strategy={dedupe_strategy})')
        if dedupe_strategy == 'best':
            data = data.groupby(data.index) \
                .apply(lambda g: g[g.forecast == g.forecast.min()]) \
                .droplevel(0)
        elif isinstance(dedupe_strategy, int) and dedupe_strategy < 6:
            delta = pd.Timedelta('6h')
            lo = delta * dedupe_strategy
            hi = lo + delta
            data = data.groupby(data.index) \
                .apply(lambda g: g[(lo <= g.forecast) & (g.forecast < hi)]) \
                .droplevel(0)
        else:
            raise ValueError(f'invalid dedupe_strategy {repr(dedupe_strategy)}')

        # We no longer need the forecast hour.
        data = data.drop('forecast', axis=1)

        # Unstack until we're indexed by time alone, then flatten the columns.
        logger.debug('load: unstacking geographic indices')
        while 2 < len(data.index.levels):
            data = data.unstack()
        data = data.unstack()
        data.columns = data.columns.to_flat_index()

        # The index has lost its timezone information. Fix it.
        data.index = apollo.DatetimeIndex(data.index)

        # We're done.
        return data

    def score(self, targets, **kwargs):
        '''Score this model against some target values.

        Arguments:
            targets (pandas.DataFrame):
                The targets to compare against.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.
                The ``dedupe_strategy`` argument is ignored.

        Returns:
            pandas.DataFrame:
                A table of metrics.
        '''
        # Compute scores for every forecast period.
        scores = pd.DataFrame()
        for quantile in range(6):
            kwargs['dedupe_strategy'] = quantile
            quantile_scores = super().score(targets, **kwargs)
            lo = quantile * 6
            hi = lo + 5
            quantile_scores.index += f'_{lo:02}h-{hi:02}h'
            scores = scores.append(quantile_scores)

        # Ensure the index column has a name.
        scores.index.name = 'metric'
        return scores.sort_index()
