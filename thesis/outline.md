## Introduction
...


## Background
- Numerical weather prediction (NWP)
	- NWP vs Cloud imagery models
	- Bias correction
		- Quantile mapping
		- Multivariate
		- Sam #2: learning to correct

- Solar irradiance prediction
	- ...

- Tuning NWP model for a fixed location
	- Sam #1

- Models
	- Random forest
	- Gradient boosted trees
	- Deep learning

- Metrics
	- MSE, MAE, MAPE, SMAPE
	- MAE has nice practical interpretability
	- SMAPE has nice theoretical interpretability, e.g. bounded


## The dataset
- North American Mesoscale model (NAM)
	- Dozens of features
	- Contiguous US
	- Highly dimensional
- File formats
	- GRIB
	- netCDF
- Sources
	- NOAA
	- NCAR


## Experiments
- Baselines
	- Persistence
		- Ground truth vs 0h forecast

- Linear vs RF vs GBT
	- Cross validated (2017)
	- 24h lag, 24h target
	- 1x1 area, 0h forecast
	- Analysis: error vs runtime

- How do aspects of NWP affect accuracy
	- Constant: Prediction = 24h
	- Cross validated (2017):
		- DSWRF alone vs all data variables
		- lag vs forecast
		- forecast vs area
			- Best lag from above
			- lag: 1 cell vs matching area
	- Analysis: error
	- Known weaknesses:
		- Not using time series cross validation

- Bias correction

- Test set (2018)
	- Constant: best model/lag/forecast/area/bias correction
	- For all target hours
	- Retrain model for each (range of?) hour(s)
	- Analysis: error


## Future Work
- Deep learning
