#!/bin/bash

source '/opt/conda/etc/profile.d/conda.sh'
conda activate

export APOLLO_DATA='/apollo-data'

apollo download | tee -a ./download.log
