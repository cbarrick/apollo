Getting Started
==================================================

.. contents::
    :local:


Installing conda
--------------------------------------------------

Conda is a cross-platform, language-agnostic environment and package manager. This guide will use a minimal installer of conda, called Miniconda, to setup a development environment for Apollo.

Local vs global installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before installing conda, we must decide if we are installing to our home directory or to a global system directory. To setup a development environment, it is best to install to your home directory. For a deployment environment, it is best to install globally.

We will use a shell variable to control our installation prefix. For local installations, ``~/conda`` is a typical choice::

    $ CONDA_PREFIX="$HOME/conda"

For global installations, consider ``/opt/conda``::

    $ CONDA_PREFIX="/opt/conda"

Note that installing into ``/opt`` usually requires superuser privileges. The rest of this guide assumes the ``CONDA_PREFIX`` variable is set.

Download and install
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our next step is to download the Miniconda3 installer. These can be found online at https://conda.io/miniconda.html. For a detailed list of all Miniconda versions and their MD5 checksums, see https://repo.continuum.io/miniconda/. For this guide, we will install the latest Miniconda3 on a 64-bit Linux platform.

First, download the latest Miniconda installer::

    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh

Before installing, you may check the MD5 checksum against the value published at https://repo.continuum.io/miniconda/. If it matches, we can be sure that the download is not corrupt or tampered with::

    $ md5sum Miniconda3-latest-Linux-x86_64.sh
    e1045ee415162f944b6aebfe560b8fee  Miniconda3-latest-Linux-x86_64.sh

If the MD5 checksum matches, we can continue to install::

    $ chmod +x ./Miniconda3-latest-Linux-x86_64.sh
    $ ./Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_PREFIX"

The ``-b`` argument tells the installer to use "batch mode". This implicitly agrees to the terms of service and installs with sane defaults. The ``-p "$CONDA_PREFIX"`` argument gives the installation directory.

You may view additional options with ``-h``::

    $ ./Miniconda3-latest-Linux-x86_64.sh -h

    usage: ./Miniconda3-latest-Linux-x86_64.sh [options]

    Installs Miniconda3 4.5.11

    -b           run install in batch mode (without manual intervention),
                 it is expected the license terms are agreed upon
    -f           no error if install prefix already exists
    -h           print this help message and exit
    -p PREFIX    install prefix, defaults to /home/chris/miniconda3, must not contain spaces.
    -s           skip running pre/post-link/install scripts
    -u           update an existing installation
    -t           run package tests after installation (may install conda-build)


Setting ``$PATH`` and activating the base environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Properly adding conda to your ``$PATH`` is more subtle than it seems. Fortunately, conda provides a script to handle the subtleties::

    $ source "$CONDA_PREFIX/etc/profile.d/conda.sh"

Now we can activate the base environment::

    $ conda activate

To perform this automatically at login, you may add this line to your ``~/.bashrc``::

    $ echo "source '$CONDA_PREFIX/etc/profile.d/conda.sh' && conda activate" >> ~/.bashrc

.. important::
    Adding this line to your ``.bashrc`` is not sufficient to make conda available in all scripts. In particular, cron jobs which require a particular conda environment should manually source the ``conda.sh`` script and activate the appropriate environment.


Working with Apollo
--------------------------------------------------

Downloading Apollo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apollo development is tracked in a git repository. It can be cloned with the following::

    $ git clone https://github.com/cbarrick/apollo

The rest of this guide assumes that your working directory is the root of the Apollo repository::

    $ cd ./apollo

Installing and maintaining the Apollo environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The development environment for Apollo is maintained in the ``environment.yml`` file. We must initialize the environment in conda::

    $ conda env create -f ./environment.yml

This will install an environment named ``apollo`` in the current conda prefix. The environment must be activated before it is used::

    $ conda activate apollo

To update the Apollo environment, use the following::

    $ conda env update -f ./environment.yml

.. warning::
    Do not use the simple ``conda update`` to update this environment. The simple update command uses the global conda settings, which do not include the required channels by default. The full command given above tells conda to use the settings given in the ``environment.yml`` file.
