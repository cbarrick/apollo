import copy
import logging
import os
import re
import sys

from itertools import chain
from weakref import WeakValueDictionary
from zipfile import ZipFile

import numpy as np
import sklearn

logger = logging.getLogger(__name__)


class DataSet:

    def __init__(self,
            path='./gaemn15.zip',
            city='griffin',
            years=range(2003,2015),
            x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
            y_features=('solar radiation (+4)',),
            window=4, # TODO: rename to `lag`
            lead=0,
            scale=None,
            threshold=0.0):

        self.path = path
        self.city = str(city).upper()
        self.years = tuple(years)
        self.x_features = tuple(FeatureSpec(s) for s in x_features)
        self.y_features = tuple(FeatureSpec(s) for s in y_features)
        self.features = self.x_features + self.y_features
        self.lag = max(1, int(window))
        self.scale = scale
        self.threshold = threshold

        self._split = len(self.x_features)
        self._shards = self.load_data()

    def load_data(self):
        logger.info('loading data: %s', self)

        # Load raw data from the zip archive. Within the zip, data is split
        # across space-separated files with names like `GRIFFIN.F03`.
        cols = tuple(f.index for f in self.features)
        with ZipFile(self.path) as archive:
            fnames = ('FifteenMinuteData/{}.F{:02d}'.format(self.city, y-2000) for y in self.years)
            tables = (np.loadtxt(archive.open(f), usecols=cols) for f in fnames)
            data = np.concatenate(tuple(tables))

        # Some columns may be pseudo features, transforms of the raw features.
        # Here we apply the various transforms.
        for i, f in enumerate(self.features):
            if f.int:
                x, n = np.modf(data[..., i])
                data[..., i] = n

            if f.frac:
                x, n = np.modf(data[..., i])
                data[..., i] = x

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

        # Apply scaling
        if callable(self.scale):
            data[..., :self._split] = self.scale(data[..., :self._split])

        # Apply windowing
        shards = []
        for i in range(self.lag):
            d = data[i:]
            s = d.shape
            extra = s[0] % self.lag
            if extra > 0: d = d[:-extra]
            d = d.reshape((s[0]//self.lag, self.lag, s[1]))
            shards.append(d)

        # Apply filtering
        if self.threshold > 0:
            discarded = 0
            for i,sh in enumerate(shards):
                fsh = np.ndarray(sh.shape)
                j = 0
                for instance in sh:
                    m = np.mean(instance[..., self._split:])
                    if self.threshold <= m:
                        fsh[j] = instance
                        j += 1
                    else:
                        discarded += 1
                shards[i] = fsh[:j]
            logger.debug('discarded %d', discarded)

        return tuple(shards)

    def batch(self, batch_size=32):
        xn = len(self.x_features) * self.lag
        for sh in self._shards:
            for i in range(0,len(sh),batch_size):
                batch = sh[i:i+batch_size]
                x = batch[..., :self._split].reshape((-1,xn))
                y = batch[..., self.lag-1, self._split:]
                yield x, y

    @property
    def data(self):
        xn = len(self.x_features) * self.lag
        return self._join_shards()[..., :self._split].reshape((-1,xn))

    @property
    def target(self):
        return self._join_shards()[..., self.lag-1, self._split:]

    def _join_shards(self):
        if len(self._shards) > 1:
            d = np.concatenate(self._shards)
            self._shards = [d]
        return self._shards[0]

    def split(self, p):
        n = len(self._shards[0])
        s = int(n * p)
        a = copy.copy(self)
        b = copy.copy(self)
        shards = self._shards
        a._shards = tuple(sh[:s] for sh in shards)
        b._shards = tuple(sh[s:] for sh in shards)
        return a, b

    def dump(self, file=sys.stdout, delim=',', header=True, max_rows=-1):
        # TODO WINDOW SUPPORT
        if header:
            headers = (str(f) for f in self.features)
            if delim == ' ':
                headers = (s.replace(' ', '_') for s in headers)
            else:
                headers = (s.replace(delim, ' ') for s in headers)
            print(delim.join(headers), file=file)

        for sh in self._shards:
            for row in sh:
                print(delim.join(row.astype(str)), file=file)
                max_rows -= 1
                if max_rows == 0:
                    return

    def __repr__(self):
        return "DataSet(path={}, city={}, years={}, x_features={}, y_features={}, lag={})"\
            .format(
                self.path.__repr__(),
                self.city.__repr__(),
                self.years.__repr__(),
                tuple(f.__str__() for f in self.x_features),
                tuple(f.__str__() for f in self.y_features),
                self.lag.__repr__(),
            )


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
        spec = str(spec)
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
