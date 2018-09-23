#!/bin/bash

# Change to the working directory of the project.
# This makes sure the proper python modules are in scope.
cd /mnt/data6tb/chris/

# Activate the Anaconda environment.
source '/opt/conda/etc/profile.d/conda.sh'
conda activate ./prod_env

# Make sure the most recent 4 forecasts are in the cache.
python -m bin.download -n 4 2>&1 | tee -a ./download.log
