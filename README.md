# Apollo

Machine learning for solar irradiance forecasting.

## Installation

Read the official 
[Getting Started](https://cbarrick.github.io/apollo/user-guides/getting-started.html)
guide.

## Usage

Apollo can be used in the following capacities:
    
- A software library
- A standalone tool for training and using machine learning models of solar irradiance
- An application to explore and visualize locally cached data sources and irradiance predictions

The [Apollo docs](https://cbarrick.github.io/apollo/index.html)
include guides for each of these use cases.

## Contributing

### Contribution Guidelines

If you would like to contribute, please send us a pull request!  
We are always happy to look at improvements and new experiments.

Code should comply with PEP8 standards as closely as possible.  
We use [Google-style docstrings](http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) 
to document the Python modules in this project.

### Implementing custom models

Apollo models are trainable, serializable predictors of solar irradiance.
All models should inherit from the base `Model` class in [`apollo.models.base`](apollo/models/base.py).
Custom models should overwrite all abstract methods on the base class.
Models that conform to the API can be used with the [command-line interface](apollo/__main__.py).  


## Contributors
- [Chris Barrick](https://github.com/cbarrick)
- [Zach Jones](https://github.com/zachdj)
- [Frederick Maier](https://github.com/fwmaier)
- [Aashish Yadavally](https://github.com/aashishyadavally)

## Acknowledgements
- Dr. Frederick Maier
- Dr. Khaled Rasheed
