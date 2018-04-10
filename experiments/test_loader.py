import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from apollo.datasets import nam, ga_power, simple_loader

data, targets = simple_loader.load(target_hour=24)

print(targets)
