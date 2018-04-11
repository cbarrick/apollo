import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from apollo.datasets import nam, ga_power, simple_loader

desired_attributes = []

data, targets = simple_loader.load(target_hour=24, start='2017-06-08', stop='2017-06-09', cache_dir='../data')

print(data)
print(targets)
