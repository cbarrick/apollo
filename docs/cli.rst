Command Line Reference
===========================================================================

.. highlight:: none

Apollo provides a command line toolbox for managing forecast data, developing forecast models, and administering the server. All Apollo commands sport a ``--help`` option with detailed descriptions. This documentation provides an overview of the commands.

.. contents::
    :local:
    :depth: 1


apollo
---------------------------------------------------------------------------

Summary
^^^^^^^

The Apollo CLI toolbox.

Usage
^^^^^

::

    apollo [-h] [--quiet | --debug | --log LEVEL] COMMAND ...

Description
^^^^^^^^^^^

The ``apollo`` command provides a *toolbox* style CLI, like ``git``. The root ``apollo`` command takes a single required argument, ``COMMAND``, giving the subcommand to execute. Optional arguments that come before the subcommand are applicable to all subcommands, while arguments that come after are specific to the subcommand.


apollo ls
---------------------------------------------------------------------------

Summary
^^^^^^^

List items within the Apollo database.

Usage
^^^^^

::

    apollo ls [-h] [COMPONENT]

Description
^^^^^^^^^^^

The ``apollo ls`` command is for listing different items stored in the Apollo database. You can optionally specify a component to list only those items.

Components include:

- ``models``: The trained models.
- ``templates``: Templates for training new models.
- ``nam``: Available NAM forecasts.

Examples
^^^^^^^^

List everything in the database::

    $ apollo ls
    models/linear-nam-uga
    models/xgboost-nam-uga
    ...
    templates/linear-nam
    templates/xgboost-nam
    ...
    nam/2017-01-01T00Z
    nam/2017-01-01T06Z
    nam/2017-01-01T12Z
    nam/2017-01-01T18Z
    nam/2017-01-02T00Z
    nam/2017-01-02T06Z
    nam/2017-01-02T12Z
    nam/2017-01-02T18Z
    ...

List only NAM forecasts::

    $ apollo ls nam
    2017-01-01T00Z
    2017-01-01T06Z
    2017-01-01T12Z
    2017-01-01T18Z
    2017-01-02T00Z
    2017-01-02T06Z
    2017-01-02T12Z
    2017-01-02T18Z
    ...


apollo predict
---------------------------------------------------------------------------

Summary
^^^^^^^

Execute an Apollo model

.. todo::
    Document


apollo train
---------------------------------------------------------------------------

Summary
^^^^^^^

Train a new model

.. todo::
    Document


apollo score
---------------------------------------------------------------------------

Summary
^^^^^^^

Compute metrics for model output

.. todo::
    Document


apollo nam download
---------------------------------------------------------------------------

Summary
^^^^^^^

Download and process a NAM forecast

.. todo::
    Document
