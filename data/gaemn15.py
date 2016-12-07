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
            window=4):

        self.path = os.path.realpath(path)
        self.city = str(city).upper()
        self.years = tuple(years)
        self.x_features = tuple(FeatureSpec(s) for s in x_features)
        self.y_features = tuple(FeatureSpec(s) for s in y_features)
        self.features = self.x_features + self.y_features
        self.window = int(window)

        # Load data from cache if possible
        key = (self.path, self.city, self.years, self.features, self.window)
        try:
            data = _cache[key]
        except KeyError as e:
            data = self._load_data(*key)
            _cache[key] = data
        self._data = data

    @property
    def data(self):
        split = len(self.x_features)
        d = self._data[..., :split]
        if self.window:
            return d.reshape(-1, d.shape[1] * d.shape[2])
        else:
            return d

    @property
    def target(self):
        split = len(self.x_features)
        d = self._data[..., split:]
        if self.window:
            return d[:, -1, :].ravel()
        else:
            return d.ravel()

    def dump(self, file=sys.stdout, delim=',', header=True, max_rows=-1):
        if header:
            headers = (str(f) for f in self.features)
            if delim == ' ':
                headers = (s.replace(' ', '_') for s in headers)
            else:
                headers = (s.replace(delim, ' ') for s in headers)
            print(delim.join(headers), file=file)

        for row in self._data:
            print(delim.join(row.astype(str)), file=file)
            max_rows -= 1
            if max_rows == 0:
                break

    def _load_data(self, path, city, years, features, window):
        logger.info('loading data: city=%s, years=%s, features=%s, window=%s',
                    city, years, features, window)

        # Data is split across space-separated files with names like `GRIFFIN.F03`.
        with ZipFile(path) as archive:
            cols = tuple(f.index for f in features)
            fnames = ('FifteenMinuteData/{}.F{:02d}'.format(city, y-2000) for y in years)
            tables = (np.loadtxt(archive.open(f), usecols=cols) for f in fnames)
            data = np.concatenate(tuple(tables))

        # Some columns may be pseudo features, transforms of the raw features.
        # Here we apply the various transforms.
        for i, f in enumerate(self.features):
            if f.noise:
                z = float(f.noise)
                std = np.std(data[..., i])
                noise = np.random.randn(len(data)) * std * z
                data[..., i] += noise

            if f.shift:
                data[..., i] = np.roll(data[..., i], -f.shift)

            if f.delta:
                d = np.diff(data[..., i])
                data[1:, i] = d
                data[0,  i] = 0

        # Discard rows that become invalid due to wrap around.
        min_shift = min(f.shift or 0 for f in self.features)
        max_shift = max(f.shift or 0 for f in self.features)
        if min_shift < 0: data = data[-min_shift:]
        if 0 < max_shift: data = data[:-max_shift]

        # Apply windowing
        if window > 0:
            shards = []
            for i in range(window):
                d = data[i:]
                s = d.shape
                extra = s[0] % window
                if extra > 0: d = d[:-extra]
                d = d.reshape((s[0]//window, window, s[1]))
                shards.append(d)
            data = np.concatenate(shards)

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

    spec_fmt = re.compile('([^\(]+)(\((.+)\))?')
    key_val_fmt = re.compile('(.+)=(.+)')

    def __init__(self, spec):
        spec = FeatureSpec.spec_fmt.match(spec)
        assert spec is not None, 'invalid feature spec'
        self.name = spec.group(1).strip()
        self.index = FeatureSpec.names.index(self.name)
        try:
            self.modifiers = spec.group(3).split(',')
        except:
            self.modifiers = []
        for m in self.modifiers:
            kv = FeatureSpec.key_val_fmt.match(m)
            if kv:
                key = kv.group(1).strip().replace(' ', '_')
                val = kv.group(2).strip().replace(' ', '_')
                setattr(self, key, val)
            elif m[0] in '+-':
                self.shift = int(m)
            else:
                setattr(self, m, True)

    def __getattr__(self, key):
        return None

    def __str__(self):
        s = self.name
        if len(self.modifiers) > 0:
            s += ' ('
            s += ','.join(s.strip().replace(' ', '_') for s in self.modifiers)
            s += ')'
        return s

    def __repr__(self):
        return "FeatureSpec('{}')".format(str(self))
