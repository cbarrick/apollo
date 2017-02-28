# Solar Radiation Prediction

This repository contains code and notes for a project to predict solar radiation and energy production in Georgia.

## Dependencies

The experiments are written in Python and depend on:
- [The Scipy stack](http://scipy.org/) (`numpy`, `scipy`, `matplotlib`)
- [Scikit-learn](http://scikit-learn.org/stable/)
- [The Matplotlib Basemap Toolkit](http://matplotlib.org/basemap/)
- [pygrib](https://github.com/jswhit/pygrib)
- [XGBoost](https://github.com/dmlc/xgboost)

Basemap, pygrib, and XGBoost are linked in this repository as git submodules.

## Contents
- `data` - data access classes to read from our various data sets. No actual data is hosted in this repository.
- `experiments` - code and notes for the various experiments. Each experiment contains a README that explains the experiment and presents the results.
- `models` - code for custom models used in some experiments (e.g. neural nets).
