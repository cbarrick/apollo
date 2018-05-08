# Apollo

Utilities and experiments to analyze solar radiation forecasts.

## Setup

### Installing project dependencies

1. Install [Conda](https://conda.io/docs/user-guide/install/index.html)

2. Create a conda environment from the included `environment.yml` file

    `$ conda env create -f environment.yml`

3. Activate the environment

    `$ source activate apollo`
    
4. Run experiments
    
    `$ (apollo) python -m apollo`
    
### Running experiments

Apollo exposes a command line interface for running machine learning models against locally cached data.
The general form of the command is:

    `$ python -m apollo <experiment args> <action> <action args>`

The `dataset_args` define the model that you want to run and the dataset that you want to target.
The `action` is a positional argument that specifies what you would like to do with the model.
Action should be one of:
 - `train`: to train the model on the selected dataset and serialized it to a file
 - `evaluate`: to evaluate a model on the selected dataset using n-fold cross validation
 - `predict`: to use a trained model to make predictions on the selected dataset
 
The `action_args` are specific to each action, and can modify the behavior of the selected action.
For example, the `--num_folds` argument in the `evaluate` action determines the number of folds to use
when estimating cross-validated accuracy.

Run `python -m apollo -h` to view a list of `experiment_args` and the available actions.

Run `python -m apollo <action> -h` to view help for a specific action.

### Additional Notes

Data must be cached (downloaded) locally before it can be targeted by Apollo.  
The `--cache_dir` argument specifies the location of the downloaded dataa.
It should contain two subdirectories, `NAM-NMM` and `GA-POWER`.

The `NAM-NMM` subdirectory should contain weather forecast data downloaded and processed using the 
`apollo.datasets.nam` package.  See [`bin/download.py`](bin/download.py) for a script that can 
download and process NAM forecasts.

The `GA-POWER` subdirectory should contain a summary log file with the target readings from solar arrays.
Note that this directory structure is currently unstable due to impending changes in the `apollo.datasets.ga_power`
loader.

## Contributing

### Contribution Guidelines

If you would like to contribute, please send us a pull request!  
We are always happy to look at improvements and new experiments.

Code should comply with PEP8 standards as closely as possible.  
We use [Google-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 
to document the Python modules in this project.

### Writing new experiments

An experiment is a Python object that inherits from [Experiment](experiments/Experiment.py).
Custom experiments should overwrite the `save`, `load`, `train`, `evaluate`, and `predict` functions.

An experiment's `train` function should train a machine learning model on the dataset specified 
by the set of `experiment_args` and should serialize the model to a file.
The `train` function must return the path of the serialized model.

An experiment's `evaluate` function should evaluate the same machine learning model on the specified dataset
using n-fold cross-validation.  
The `evaluate` function must return a single value which estimates the model's performance using the Mean Absolute Error (MAE) criterion.

An experiment's `predict` function is used to generate predictions for a dataset.
This function should use the serialized model from the `train` function to make predictions and write the 
predictions to a file.
Currently, the format of the predictions file is unstable.  It will be documented once stabilized.
The `predict` function must return the path where the predictions were saved.


## Contributors
- Chris Barrick
- Zach Jones

## Acknowledgements
- Dr. Frederick Maier
- Dr. Khaled Rasheed
