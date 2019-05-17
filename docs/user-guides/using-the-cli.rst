Using the CLI
=============

.. contents::
    :local:

Apollo Command-Line-Interface
-----------------------------

Apollo includes a command-line interface which exposes a high-level API
to train, manage, and use models that predict solar irradiance.
Most of the commands require NAM data to be cached locally in the directory
specified by the ``APOLLO_DATA`` environment variable.
This guide also assumes that you have setup a conda environment with the
appropriate dependencies.  If you need help placing data in the right location,
or if you need using Conda, see the :doc:`getting-started` guide.

The CLI includes four scripts, one for each of the following tasks:
    - :ref:`Predicting Irradiance <predicting>` using previously trained models
    - :ref:`Training new models <training>` using historical NAM data
    - :ref:`Evaluating models <evaluating>` to approximate their accuracy.
    - :ref:`Deleting models <deleting>` that are no longer useful

Apollo ships with a set of models that have been pre-trained on the NAM data
from 2017-2018.  Models were trained to predict irradiance readings logged by
Kipp & Zonan model SMP-11 pyranometers from each of the solar arrays at the
University of Georgia Solar Farm in Athens, GA:

    - ``UGAAPOA1IRR``: irradiance measured by the SMP-11 pyranometer on Array A,
      the dual-axis tracker.
    - ``UGABPOA1IRR``: irradiance measured by the SMP-11 pyranometer on Array B,
      the fixed array with 200-degree azimuth.
    - ``UGACPOA1IRR``: irradiance measured by the SMP-11 pyranometer on Array C,
      a fixed array with 270 azimuth.
    - ``UGADPOA1IRR``: irradiance measured by the SMP-11 pyranometer on Array D,
      a fixed array with 270 azimuth.
    - ``UGAEPOA1IRR``: irradiance measured by the SMP-11 pyranometer on Array E,
      the single-axis tracker.

Each model targets future hours 1-24.
The following types of models ship with Apollo:

    - 24-hour Persistence Model
    - Linear Regression
    - Support-Vector Regression
    - k-Nearest Neighbors
    - Model Tree (i.e. a Decision Tree with a continuous target)
    - Multilayer Perceptron
    - Random Forest
    - Gradient-Boosted Trees

A description of each of these models and the hyperparameters used to customize
their behavior can be found in the :doc:`../api/apollo.models` API docs.

The following sections demonstrate how the CLI scripts can be used to
generate irradiance predictions, train new models, evaluate accuracy, and delete
saved models.
A brief description of each script is provided below along with examples
demonstrating its use.


.. _predicting:

Predicting Irradiance with Trained Models
-----------------------------------------

Models that have already been trained and saved can be used to generate
predictions of solar irradiance.  The values predicted by a model correspond
to the *variable* and the *future hours* targeted by the model.  These variables
are set at the time when a model is trained.

The predict script will use a saved model to generate predictions relative to
some *reference time*.  If the reference time is not specified, predictions will
be generated for the reference time midnight on January 1st, 2018.
By default, predictions will be written to the ``output`` subdirectory in the
path specified by ``$APOLLO_DATA``.

By default, predictions are output as JSON files containing the raw predictions
as well as meta-data about the timeframe of the prediction, the model used to
generate the prediction, and the variables being predicted.
A full description of the JSON output format can be found TODO.
Alternatively, predictions can be output as CSV files with no metadata except
column names.

| After activating the conda environment, the predict script can be used by
  invoking the following command:
| ``python -m apollo predict <model-name> <arguments>``

The predict script should be given the name of the saved model that will be used
to generate the prediction.  To view the other optional arguments, run
``python -m apollo predict -h``.  This will print the following
usage instructions::

    usage: python -m apollo predict [-h] [-j | -c] [-t TIMESTAMP] MODEL

    Generate a prediction from a trained Apollo model.

    positional arguments:
      MODEL                 the name of the model

    optional arguments:
      -h, --help            show this help message and exit
      -j, --json            write predictions as JSON (default)
      -c, --csv             write predictions as CSV
      -t TIMESTAMP, --reftime TIMESTAMP
                            make a prediction for the given reftime

Examples
^^^^^^^^

| **Predicting irradiance using a Random Forest**
| ``python -m apollo.bin.predict random_forest_a``

This will use the *Random Forest* model trained against the
pyranometer on Array A to generate a prediction for the reference time
January 1st, 2018, 12:00AM
(this is one of the pre-trained models that ships with Apollo).

| **Similar commands are used to generate predictions using other models**
| ``python -m apollo predict random_forest_b``
| ``python -m apollo predict random_forest_e``
| ``python -m apollo predict neural_net_c``
| ``python -m apollo predict your_custom_model_name``

| **Predicting irradiance for a specific reference time**
| ``python -m apollo predict random_forest_e --reftime 2019-06-22 18:00:00``

This will use the Random Forest model trained against the pyranometer on Array E
to predict irradiance relative to the reference time June 22nd, 2019 at 6PM UTC.
Note that the NAM data for the desired reftime must be cached locally in the
directory specified by ``$APOLLO_DATA``.

| **Predicting irradiance for the latest available reference time**
| ``python -m apollo predict random_forest_e --latest``

Note that that option is incompatible with the 'reftime' argument.

| **Writing predictions to a custom directory**
| ``python -m apollo predict random_forest_e --reftime 2018-12-31 --out_path path/to/target/directory``

| **Writing predictions in CSV format**
| ``python -m apollo predict random_forest_e --csv``

| **More Examples**
| ``python -m apollo predict gbt_a --reftime 2019-04-16 --out_path path/to/target/directory --csv``
| ``python -m apollo predict dtree_b --latest --out_path path/to/target/directory --csv``
| ``python -m apollo predict linear_regression_d --reftime 2017-01-01``
| ``python -m apollo predict svr_e --latest --csv``


.. _training:

Training New Models
-------------------

| After activating the conda environment, new models can be trained using locally cached NAM data using the following command:
| ``python -m apollo train <model-type> <arguments>``

The training script should be given the type model to be trained.
Apollo also allows for extensive customization of the dataset used to train a
model as well as the hyperparameters than control a model's behavior.
These options are passed to the script as *keyword arguments* using the syntax
``--set keyword=value``.
The dataset can be customized using the keyword arguments described in
:doc:`../api/stubs/apollo.datasets.solar.SolarDataset`.
The hyperparameters for each model are documented in the
:doc:`../api/apollo.models` API docs.

By default, models will be trained on the data spanning January 1st, 2017 to
December 31, 2018 using the default arguments of ``SolarDataset``.

Models trained with the CLI will always be saved in the ``$APOLLO_DATA`` directory.

To view a full description of the arguments, run
``python -m apollo train -h``.
This will print the following usage instructions::

    usage: python -m apollo train [-h] [--set KEY=VALUE] [-r START STOP] MODEL

    Train a new Apollo model.

    positional arguments:
      MODEL                 the class of the model to train

    optional arguments:
      -h, --help            show this help message and exit
      --set KEY=VALUE       set a hyper-parameter for the model, may be specified
                            multiple times
      -r START STOP, --range START STOP
                            train on all forecast on this range, inclusive

Examples
^^^^^^^^

| **Training a new Random Forest model**
| ``python -m apollo train RandomForest``

This command will train and save a new Random Forest model on the data from
January 1st, 2017 to December 31st, 2018.
The model will be saved with a unique name that is automatically generated.

| The command is similar for different types of models:
| ``python -m apollo train KNearest``
| ``python -m apollo train GradientBoostedTrees``
| ``python -m apollo train MultilayerPerceptron``

| **Training a new model with a custom name**
| ``python -m apollo train RandomForest --set name=my-custom-tree``

The 'name' keyword argument can be passed to save the model with a custom name.
The name can be referenced when using the :ref:`predict script <predicting>`.

| **Training on a custom historical period**
| ``python -m apollo train KNearest --start 2017-06-01 --stop 2018-03-15``

The 'start' and 'stop' arguments are used to select a subset of the historical
NAM data used to train a model.  This example trained a KNN model using historical
data between June 1st, 2017 and March 15, 2018.

| **Customizing model behavior with kwargs**
| ``python -m apollo train KNearest --set target=UGADPOA1IRR --set n_neighbors=15``

A set of keyword arguments can be passed to customize the data used to train the
model and the model's hyperparameters.
This example trains a KNN model that targets the readings from the SMP-11 \
pyranometer on Array D.  It also sets a hyperparameter of the KNN model,
``n_neighbors`` to 15.

The keyword arguments can be any keyword from the
:doc:`../api/stubs/apollo.datasets.solar.SolarDataset` constructor, or any
appropriate model hyperparameter documented in :doc:`../api/apollo.models`.

| **More Examples**
| ``python -m apollo train SVR``
| ``python -m apollo train LinearRegression --set target=UGAEPOA3IRR``
| ``python -m apollo train LinearRegression --start 2017-01-01 --stop 2017-12-31 --set target=UGACPOA2IRR``
| ``python -m apollo train SVR --set forecast=24 --set temporal_features=False``
| ``python -m apollo train SVR --set kernel=sigmoid --set epsilon=5``
| ``python -m apollo train MultilayerPerceptron --set activation=logistic --set solver=sgd``
| ``python -m apollo train DecisionTree --start 2017-01-01 --stop 2019-06-31 --set target=UGADPOA1IRR --set max_depth=30``


.. _evaluating:

Evaluating Model Accuracy
-------------------------

Apollo includes a utility that can be used to estimate the accuracy of a saved
model using the metrics *mean absolute error*, *mean squared error*,
*root mean squared error*, and *coefficient of determination*.

Models are evaluating on a *validation dataset*.  The model will be re-trained
on a portion of the validation dataset and evaluated on the remaining portion.
For all models other than Persistence models, the NAM data for the validation
dataset needs to be cached locally in the ``$APOLLO_DATA`` directory.
By default, models will be evaluated on the data between
January 1st, 2017 and December 31st, 2017.

Two methods are available to evaluate models, timeseries cross-validation and
train-test splitting.
The train-test splitting method is very simple.  The dataset used for validation
is split into two pieces, one for training and one for testing.  The model is
trained on the training set, then its performance is evaluated on the test set.
This method is relatively fast, very simple, and can provide a good estimation
of accuracy given a sufficient quantity of data.

The other method, n-fold Timeseries cross-validation, is the typical method used
to evaluate machine learning models that deal with ordered data.
The dataset used for validation is split into *n* folds.  For each iteration from
i=1 to *n*, the model is trained on the first i folds, then evaluated on fold i+1.
This method often provides a robust estimation of accuracy, but, compared to the
train-test split method, it takes much longer.

Many Apollo models target numerous future hours.  For these models, there are
two options for the reporting of evaluation results.  Results can be computed
for each specific target hour, or results can be combined into a single number
expressing average performance across all target hours.

| After activating the conda environment, trained models can be evaluated using the following command:
| ``python -m apollo evaluate <model-name> <method> <arguments>``

To view a full description of the arguments, run
``python -m apollo evaluate -h``.
This will print the following usage instructions::

    usage: python -m apollo evaluate [-h] [-a] [-c] [-r START STOP] (-k K | -p RATIO)
                                MODEL

    Evaluate a trained Apollo model.

    positional arguments:
      MODEL                 the name of the saved model to be evaluated

    optional arguments:
      -h, --help            show this help message and exit
      -a, --average         evaluate the mean error of forecasts for all hours
      -c, --csv             output the results as a csv
      -r START STOP, --range START STOP
                            evaluate using forecast on this range, inclusive
      -k K, --cross-val K   evaluate using K-fold timeseries cross-validation
      -p RATIO, --split RATIO
                            evaluate using a test-train split with this ratio

The following examples demonstrate how the CLI can be used to evaluate models.

Examples
^^^^^^^^

| **Evaluating a model**
| ``python -m apollo evaluate random_forest_a cross_val``

This command will evaluate the model named 'random_forest_a'
(this is one of the pre-trained models that ships with Apollo)
using the timeseries cross-validation method.

| **Train-test split validation**
| ``python -m apollo evaluate random_forest_a split``

This is identical to the first example, except the train-test split validation
method is used to evaluate the model's performance.

| **Customizing the validation dataset**
| ``python -m apollo evaluate random_forest_a cross_val --first 2018-01-01 --last 2019-04-15``

This command will evaluate the model named 'random_forest_a' using a validation
dataset spanning January 1st, 2018 to April 15th, 2019.

| **Customizing the number of cross-validation folds**
| ``python -m apollo evaluate random_forest_a cross_val --k 10``

| **Customizing the train-test split**
| ``python -m apollo evaluate random_forest_a split --split_size 0.2``

This command will cause 20% of the validation dataset to be used for testing and
the other 80% to be used for trianing.

| **Combining results from multiple target hours**
| ``python -m apollo evaluate random_forest_a cross_val --average``

The '--average' flag will combine results from all target hours into a single
number expressing the average performance across all target hours.

| **More Examples**
| ``python -m apollo evaluate svr_a cross_val``
| ``python -m apollo evaluate my_custom_model split --split_size 0.3``
| ``python -m apollo evaluate linear_regression_c cross_val --k 10 --first 2017-01-01 --last 2018-12-31``
| ``python -m apollo evaluate dtree_d split --first 2017-06-01 --average``


.. _deleting:

Deleting Models
---------------

When using the :ref:`training script <training>`, models are automatically saved
to ``$APOLLO_DATA``.  The CLI provides a utility to delete old models that are
no longer useful.

| To delete a model, run the following command:
| ``python -m apollo delete <model-name>``

The model with a matching name will be permanently deleted.
To view a list of trained models by name, run the command with the ``-h`` flag:
``python -m apollo delete -h``.

.. danger::
    Be careful when using the CLI to delete models.  The selected model will be
    permanently deleted and will be unrecoverable.  Some models take several
    hours to train and may be difficult to replace if deleted.

Examples
^^^^^^^^

| **Deleting a model by name**
| ``python -m apollo delete my-custom-model``

Assuming you have previously trained a model with the name 'my-custom-model',
this command will delete the model.

