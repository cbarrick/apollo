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

### Output Files
The `predict` action will write two files - a summary file and a prediction file.  

The summary file will be written to in the directory specified by the `summary_dir` action argument.  The summary file provides meta-data on the predictions made by a model.  It is a json file formatted as follows :
```javascript
{
   "source":"rf",                           // model used to generate the predictions
   "sourcelabel":"rf",                      // human-readable label
   "site":"UGA-C-POA-1-IRR",                // name of the target variable
   "created":1536119284,                    // timestamp (epoch time) when the prediction file was created
   "start":1514764800,                      // timestamp (epoch time) of the first prediction
   "stop":1517356800,                       // timestamp (epoch time) of the last prediction
   "resource":"/PATH/TO/PREDICTION.json"    // the location of the prediction file
}
```

The prediction file contains the raw prediction data.  It is also a json file with the following format:
```javascript
{
   "start":1514764800,                  // timestamp (epoch time) of the first prediction
   "stop":1517356800,                   // timestamp (epoch time) of the last prediction
   "site":"UGA-C-POA-1-IRR",            // name of the target variable
   "columns":[                          // column metadata
      {
         "label":"TIMESTAMP",           // name of the column
         "units":"",                    // units for the data in the column
         "longname":"",
         "type":"datetime"              // type of the data in the column.  One of {datetime, number, string}
      },
      {
         "label":"UGA-C-POA-1-IRR",
         "units":"w/m2",
         "longname":"",
         "type":"number"
      }
   ],
   "rows":[
      [
         1514764800,
         15.912018618425853
      ],
      [
         1514786400,
         26.86869409006629
      ],
      ...
   ]
}
```


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

### Writing new models

A Model is a Python object that inherits from [Model](apollo/models/base.py).
Custom models should overwrite the `save`, `load`, `train`, `evaluate`, and `predict` functions.  Models that 
conform to the API can be used with the [command-line interface](apollo/__main__.py).  The CLI does not automatically
discover models, so an instance of the custom model needs to be added to the `MODELS` dictionary in the main file.


## Contributors
- Chris Barrick
- Zach Jones

## Acknowledgements
- Dr. Frederick Maier
- Dr. Khaled Rasheed
