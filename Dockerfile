# This image is based on debian:latest with miniconda3 installed.
# The base conda environment is activated for login shells.
FROM continuumio/miniconda3:latest

# Update debian and install cron.
RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install cron -y

# The miniconda3 image doesn't necessarily use the latest conda release.
# We update it manually now to avoid the standard outdated warning later.
RUN conda update -n base -c defaults conda --quiet

# The Filesystem Hierarchy Standard says we should build our app here.
# https://wiki.debian.org/FilesystemHierarchyStandard
WORKDIR /usr/local/src/apollo

# Install our environment on top of conda's base environment.
# Do this before copying the rest of the source code to allow docker to cache
# the intermediate image. This saves gigabytes of package downloads when
# rebuilding the image after changing the source code.
COPY environment.yml /usr/local/src/apollo
RUN conda env update -f environment.yml -n base --quiet

# Copy the source code to the image. The actual files copied into the image
# are whitelisted by the `.dockerignore` file.
COPY . /usr/local/src/apollo

# Install the apollo package.
RUN pip install .

# Install the cron job.
RUN install ./scripts/cron.bash /etc/cron.hourly/apollo

# Configure apollo to use `/apollo-data` for the data store.
# This is expected to be mounted as an external volume.
ENV APOLLO_DATA="/apollo-data"
VOLUME /apollo-data
WORKDIR /apollo-data

# The job of this container is to download and process forecasts.
# That is encapsulated by the cron job.
CMD ["cron", "-f"]
