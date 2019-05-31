Apollo
==================================================

Apollo is a system for solar irradiance prediction, developed at the University of Georgia. Apollo provides a machine learning framework for fine-tuning numerical weather forecasts to observed solar irradiance on-site, and a web application for exploring predictions and observations.

Apollo is deployed as a containerized service and provides three interfaces:

- A web application for data exploration.
- A command-line interface for administering the system. This includes commands to ingest numerical weather forecasts, form predictions, and train new models.
- A Python API including tools for data analysis.


Documentation
-------------------------

[Prebuilt documentation is hosted online][docs].

The documentation is built from source with [Sphinx][sphinx]. You can build and browse the documentation locally with the following:

```console
$ make -C docs html
$ open ./docs/_build/html/index.html
```

[docs]: https://cbarrick.github.io/apollo
[sphinx]: http://www.sphinx-doc.org/en/master/


Contributing
-------------------------

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


Contributors
-------------------------

- [Chris Barrick](https://github.com/cbarrick)
- [Zach Jones](https://github.com/zachdj)
- [Frederick Maier](https://github.com/fwmaier)
- [Aashish Yadavally](https://github.com/aashishyadavally)


Acknowledgements
-------------------------

- Dr. Frederick Maier
- Dr. Khaled Rasheed


License
-------------------------

Copyright (c) 2019 Southern Company and University of Georgia.

Apollo is released under the terms of the MIT license. See `LICENSE.md` for details.
