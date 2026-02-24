# DSN 2026 Artifact Evaluation Guide for TPU-FI

This README file describes the structure of this project, our fault injection tool (TPU-FI), benchmarks, instructions to build the tool, and, finally, instructions to reproduce the experiments reported in our paper.


## Artifacts Description
### LLTFI - Our FI tool

TPU-FI (Tensor Processing Unit Fault Injector) is a Software-implemented fault injection tool that supports fault injection of Machine learning (ML) models in tensorflow lite (TFlite). TPU-FI works by modifying the kernel codes of the tensroflow kernels and then the injection campaign in a python based framework.

In this work, we inject bit-flips into the layers of the ML model, including Vision transformer models (ViT) and segmentation models.

### Benchmarks Used in this Work

As described in Section IV. of the paper. The links in the following table refer to the location of the benchmarks and the benchmark-specific scripts within this project. Note that these are the 

benchmakr name link to model and inputs



Benchmark Name   input cpu-model tpu-model



### Project Structure
benchmark: the inputs and models for each ML model used in this experiment
tensorflow: the modfied tensorflow with kernels modified.
tpu-rad: the setup for networking and handling of TPU based models used for radiation experiments.
├── README.md              # README file for DSN'26 Artifact Evaluation.

├── install.sh              # script to install TPU-FI on docker

├── run.sh              # script to run a large FI campaign using TPU-FI

run_fi_segmentation.py  FI runner for segmenation models
run_fi_vit.py           FI runner for Vit models
fi_config.py            configs for IO passing between python interface and underlying cpp TFlite
fi_runner.py            template functions to run FI campaign used in run_fi_segmentation.py and run_fi_vit.py
head_fc_mapping_3fc_exec.json   mapping from tflite model graph ops to Fi campaign usable layer indexing for ViT16.
attention_fc_graph_map.py code used to generate head_fc_mapping_3fc_exec.json
attention plot.py   code for attetnion block anlaysis of ViT models.
├── utils.py              # helper functions
common_tpu commmon functions used for inferecne of tflite models





## Setup & Build Instructions


### Environment Setup
The following are the recommended system configurations required to reproduce our  experiments.\\

**Docker**  These instructions require **Docker** to be installed on your system.
**system Bit:**   64-bit \
**RAM:**   at least 8 GB (Building LLVM requires a decent amount of RAM) \
**SSD/HDD:**   at least 15 GB (Required to download our benchmarks and run experiments) \

**Note**:-  To obtain the results presented in the paper, the campaign included approximately 8 million FI runs and completed in under three days on a
workstation equipped with 64 CPU cores and (for different runs to run in parallel but we are not going through it in this document)


### 1. Create the docker container

On your workspace, create a file named `Dockerfile` with the following content:

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

You are now ready to run the fault injection framework!


## Minimum working 

