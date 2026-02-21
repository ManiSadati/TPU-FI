# TPU-FI
Fault Injection in TFlite for TPU models.

## Setup & Build Instructions

These instructions require **Docker** to be installed on your system.


### 1. Create the docker container

Create a file named `Dockerfile` in the root of the repository with the following content:

```dockerfile
FROM ubuntu:22.04

# Install core utilities and Python
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential curl wget git unzip \
    ca-certificates software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Symlink for python/pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace
```


#### Build the Docker Image:

```bash
$ docker build -t tf_min_dev .
```


#### Run the Container

First time (building the container):

```bash
$ docker run --name tf_tfbuild -it -v $HOME/tfdata:/workspace tf_min_dev
```

After the first time:

```bash
docker start -ai tf_tfbuild
```



### 2. Inside the Docker Container


Clone the Repository and reun the setup script

```bash
git clone --recurse-submodules https://github.com/ManiSadati/TPU-FI.git
cd TPU-FI
cd third_party/tensorflow/ && git pull origin master && cd -
bash install.sh
```


### 3. Install Modified TensorFlow

Navigate to the TensorFlow submodule and run configuration:

```bash
cd /workspace/TPU-FI/third_party/tensorflow/
./configure
```

For all configuration prompts, press **Enter** to accept the default values.


Build TensorFlow Wheel and install the newly built wheel:

```bash
bazel build -j $(nproc) //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
pip uninstall tensorflow-cpu -y
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp310-cp310-linux_x86_64.whl
```



### 4. Setup Complete

You are now ready to run the fault injection framework!

