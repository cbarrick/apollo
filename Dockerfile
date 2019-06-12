# This image is based on debian:latest with miniconda3 installed.
# The base conda environment is activated for login shells.
FROM continuumio/miniconda3:latest
WORKDIR /usr/local/src/apollo

# Update Debian and install cron.
RUN apt-get -y update \
 && apt-get -y upgrade \
 && apt-get -y install cron \
 && apt-get clean

# Update conda to avoid the outdated warning.
RUN conda update -n base -c defaults conda

# Install our dependencies in conda's base environment.
# Do this early to save gigabytes of downloads with cached builds.
COPY ./environment.yml ./environment.yml
RUN conda env update -n base \
 && conda clean --all

# Copy the source code into the image.
# See the `.dockerignore` file for specifics.
COPY . .

# Install the python package.
# Perform an "editable" install to use these sources directly.
RUN pip install -e .

# Install the cron job.
RUN install --mode=755 ./scripts/apollo-cron /etc/cron.hourly

# Use `/apollo-data` for the data store.
# This is expected to be mounted as an external volume.
# See <https://docs.docker.com/storage/volumes/>.
ENV APOLLO_DATA="/apollo-data"
VOLUME /apollo-data

# Set the startup command of the container.
# This script launches cron jobs and the data explorer UI.
CMD ["./scripts/apollo-docker.bash"]

# Expose port 80.
#
# The data explorer runs over HTTP on port 80. When running the container,
# use the `-P` or `-p` arguments to map this port to the host machine.
# See <https://docs.docker.com/engine/reference/run/#expose-incoming-ports>.
#
# If exposing the service to the internet, this container should be behind some
# reverse proxy that handles authentication and TLS termination.
EXPOSE 80/tcp
