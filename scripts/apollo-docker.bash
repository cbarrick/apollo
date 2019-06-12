#!/bin/bash

# This is an example script for launching Apollo inside a Docker container.
# This script is used verbatim by the default Docker image.

# Set defensive shell options.
#
# Cause the script to immediately fail if any command fails. Also disable
# filename expansion ("globbing" / `*`) because we shouldn't depend on the
# filesystem. Note that we can't use `set -u` because the conda source script
# deliberately uses unbound variables.
# See <https://sipb.mit.edu/doc/safe-shell/>
set -ef

# Activate the conda environment.
#
# The default Docker container uses conda and installs Apollo into the base
# environment. We must activate that environment to use Apollo.
source '/opt/conda/etc/profile.d/conda.sh'
conda activate

# Set the path of the Apollo data directory.
#
# Apollo determines its working directory from the `APOLLO_DATA` environment
# variable. If unset, it defaults to `/var/lib/apollo`. The default Docker
# image uses `/apollo-data`.
export APOLLO_DATA='/apollo-data'

# Start cron.
#
# The default Docker containers uses a minimal PID 1 and launches no backround
# daemons. We use cron to schedule Apollo forecasts and thus need to launch the
# cron daemon manually.
cron

# Launch the web UI over HTTP on port 80.
#
# If exposing the service to the internet, this container should be behind some
# reverse proxy that handles authentication and TLS termination.
apollo data-explorer --port=80
