# The docker image uses `python:slim-buster` as a base image. 
# To build a docker image containing `pybann`, you have launch the following 
# command from the `pybann` folder:
#
# >>> docker build --no-cache=true -t <name_of_your_image>:<tag> .
#
# Then, to run a container:
#
# docker run -it --name <name_of_your_container> <name_of_your_image>:<tag>
#

# Use Python:slim-buster as base-image
FROM python:slim-buster

# Update packages
RUN apt update -y


# Copy pybann 
COPY . /app/
WORKDIR /app/

# Install required python package using pip3
RUN pip3 install -r requirements.txt

# Run pybann
RUN poetry build 
RUN pip install dist/pybann-0.1.0.tar.gz

# Run tests
RUN python3 -m unittest tests.tests_activation
RUN python3 -m unittest tests.tests_layers
RUN python3 -m unittest tests.tests_model
RUN python3 -m unittest tests.tests_gradientdescent