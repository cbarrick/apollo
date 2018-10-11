#!/bin/bash

# Set the path to the data store
export APOLLO_DATA='/mnt/data6tb/chris/data'

# Change to the working directory of the project.
# This makes sure the proper python modules are in scope.
cd /mnt/data6tb/chris/

# Activate the Anaconda environment.
source '/opt/conda/etc/profile.d/conda.sh'
conda activate ./prod_env

# Make sure the most recent 4 forecasts are in the data store.
python -m apollo.bin.download -n 4 2>&1 | tee -a ./download.log
