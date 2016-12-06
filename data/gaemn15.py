import logging
import os
import re
import sys

from itertools import chain
from weakref import WeakValueDictionary
from zipfile import ZipFile

import numpy as np

logger = logging.getLogger(__name__)
_cache = WeakValueDictionary()


class DataSet:
    def __init__(self,
            path='./gaemn15.zip',
            city='griffin',
            years=range(2003,2015),
            x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
            y_features=('solar radiation (+4)',),
            window=4,
            deltas=False):

        self.path = os.path.realpath(path)
        self.city = str(city).upper()
        self.years = tuple(years)
        self.x_features = tuple(FeatureSpec(s) for s in x_features)
        self.y_features = tuple(FeatureSpec(s) for s in y_features)
        self.features = self.x_features + self.y_features
        self.window = int(window)
        self.deltas = bool(deltas)

        # Load data from cache if possible
        key = (self.path, self.city, self.years, self.features, self.window)
        try:
            data = _cache[key]
        except KeyError as e:
            data = self._load_data(*key)
            _cache[key] = data

        # Separate data from targets
        split = len(self.x_features)
        self.data = data[..., :split]
        self.target = data[..., split:]

        # Apply deltas
        if self.deltas:
            d = self._get_deltas(self.data)
            self.data = self.data[:-1]
            self.target = self.target[:-1]
            self.data = np.concatenate((self.data, d), axis=1)

        # Apply windowing
        if self.window > 0:
            self.data = self._apply_window(self.data,   self.window)
            self.target = self._apply_window(self.target, self.window)

    def dump(self, file=sys.stdout, delim=',', header=True, max_rows=-1):
        # TODO: The header code needs to be rewritten to account for deltas
        if header:
            if self.deltas:
                logger.warn('cannot dump headers for delta columns')
            elif self.window:
                x_titles = ('{} {}'.format(f,i)
                    for i in range(self.window)
                    for f in self.x_features)
                y_titles = ('{} {}'.format(f,i)
                    for i in range(self.window)
                    for f in self.y_features)
                titles = chain(x_titles, y_titles)
                print(delim.join(titles), file=file)
            else:
                print(delim.join(str(f) for f in self.features), file=file)

        for row in self.data:
            print(delim.join(row.astype(str)), file=file)
            max_rows -= 1
            if max_rows == 0:
                break

    def _get_deltas(self, data):
        return np.diff(data, axis=0)

    def _apply_window(self, data, w):
        shards = []
        n = len(data)
        for i in range(w):
            d = data[i:n-((n-i)%w)]
            d = d.reshape((d.shape[0]//w, d.shape[1]*w))
            shards.append(d)
        return np.concatenate(shards)

    def _load_data(self, path, city, years, features, window):
        logger.info('loading data: city=%s, years=%s, features=%s, window=%s',
                    city, years, features, window)

        # Data is split across space-separated files with names like `GRIFFIN.F03`.
        with ZipFile(path) as archive:
            cols = tuple(f.index for f in features)
            fnames = ('FifteenMinuteData/{}.F{:02d}'.format(city, y-2000) for y in years)
            tables = (np.loadtxt(archive.open(f), usecols=cols) for f in fnames)
            data = np.concatenate(tuple(tables))

        # Roll the shifted features.
        for i, f in enumerate(features):
            if f.shift != 0:
                data[..., i] = np.roll(data[..., i], -f.shift)

        # Discard rows that become invalid due to wrap around.
        min_shift = min(f.shift for f in features)
        max_shift = max(f.shift for f in features)
        if min_shift < 0: data = data[-min_shift:]
        if 0 < max_shift: data = data[:-max_shift]

        # Discard partial window at the end of the data.
        extra = len(data) % window
        if extra > 0:
            data = data[:-extra]
        return data


class FeatureSpec:
    # Mapping from column index to feature name.
    names = [
        'id',
        'year',
        'day',
        'time',
        'timestamp',
        'air temp',
        'humidity',
        'dewpoint',
        'vapor pressure',
        'vapor pressure deficit',
        'barometric pressure',
        'wind speed',
        'wind direction',
        'wind direction stddev',
        'max wind speed',
        'time of max wind speed',
        'soil temp 2cm',
        'soil temp 5cm',
        'soil temp 10cm',
        'soil temp 20cm',
        'soil temp a',
        'soil temp b',
        'soil moisture',
        'pan',
        'evap',
        'water temp',
        'solar radiation',
        'total solar radiation',
        'par',
        'total par',
        'net radiation',
        'total net radiation',
        'rainfall',
        'rainfall 2',
        'max rainfall',
        'time of max rainfall',
        'max rainfall 2',
        'time of max rainfall 2',
        'leaf wetness',
        'wetness frequency',
        'battery voltage',
        'fuel temp',
        'fuel moisture'
    ]

    spec_fmt = re.compile('([^\(]+)(\(([+-][0-9]+)\))?')

    def __init__(self, spec):
        m = FeatureSpec.spec_fmt.match(spec)
        assert m is not None, 'invalid feature spec'
        name = m.group(1)
        shift = m.group(3)
        self.name = name.strip()
        self.index = FeatureSpec.names.index(self.name)
        self.shift = int(shift or 0)

    def __str__(self):
        s = self.name
        if self.shift:
            s += ' ({:+})'.format(self.shift)
        return s

    def __repr__(self):
        return "FeatureSpec('{}')".format(str(self))
