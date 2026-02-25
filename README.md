# DSN 2026 Artifact Evaluation Guide: TPU-FI

This repository contains the artifact for the TPU-FI framework proposed in the paper "Thinking Inside the Box: Injecting Realistic Radiation Faults in ML Accelerators".

TPU-FI is a software fault-injection framework for TFLite models, implemented by instrumenting TensorFlow/TFLite kernels and driving campaigns from Python.

This guide explains:
- what is included in the artifact,
- how to build/run it,
- how to reproduce the main workflow on a small scale,
- where outputs are written.

## 1. Artifact Scope

TPU-FI supports bit-flip fault injection campaigns on:
- Vision Transformer (ViT) classifiers,
- segmentation models (U-Net and DeepLab).

Fault models used by the scripts:
- `single`
- `small-box`
- `medium-box`
- `cpu`


## 2. Repository Layout

Key files/folders:
- `benchmarks/`: prepackaged `.tflite` models and `.npy` inputs.
- `third_party/tensorflow/`: TensorFlow source subtree used for kernel instrumentation/build.
- `run_fi_vit.py`: FI driver for ViT models.
- `run_fi_segmentation.py`: FI driver for segmentation models.
- `fi_runner.py`: shared FI loop.
- `fi_config.py`: FI control-plane utilities and fault sampling logic.
- `getFIT.py`: computes FIT/statistical summaries from CSV files of SDC results.
- `process_attention_results.py`: summarizes attention-level diff outputs.
- `execute_fi.sh`: example long-running campaign script.
- `execute_fi_small.sh`: shorter example of `execute_fi.sh` (takes around 1 or 2 hours).

Outputs:
- `results/*.csv`: campaign raw and processed summaries.
- `diff_results/*.npy`: tensor-level diffs for logged attention observation points.

## 3. System Requirements

Recommended minimum:
- Linux x86_64
- Docker
- 8+ GB RAM
- 16+ GB free disk

## 4. Setup and Build

### 4.1 Create a Docker image

Go the directory containing the [Dockerfile](Dockerfile), or simply create the `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    build-essential curl wget git unzip \
    ca-certificates software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace
```

Build and run:

```bash
docker build -t tf_min_dev .
docker run --name tf_tfbuild -it -v $HOME/tfdata:/workspace tf_min_dev
```

If the container is already built (future runs) use the following command instead:

```bash
docker start -ai tf_tfbuild
```

### 4.2 Clone and install prerequisites

Inside the container:

```bash
cd /workspace
git clone --recurse-submodules https://github.com/ManiSadati/TPU-FI.git
cd TPU-FI
bash install.sh
```

### 4.3 Configure and build TensorFlow

```bash
cd /workspace/TPU-FI/third_party/tensorflow
./configure
```

Accept defaults unless your environment requires changes.

Build wheel and install (this will take multiple hours):

```bash
bazel build -j $(nproc) //tensorflow/tools/pip_package:wheel --repo_env=WHEEL_NAME=tensorflow_cpu
pip uninstall tensorflow-cpu -y
pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow_cpu-2.19.0-cp310-cp310-linux_x86_64.whl
```

## 5. Minimum Working Example (about 10 minutes)

### 5.1 ViT quick campaign

Run the command below from the repository root:

```bash
python run_fi_vit.py \
  --iterations=5 \
  --imageindex=0 \
  --start_layer=0 \
  --end_layer=23 \
  --model_p=16 \
  --check_confidence \
  --check_attention
```

What this does:
- Runs a small FI campaign on ViT-16 (`--model_p=16`).
- Uses only one image (`--imageindex=0`).
- Runs 5 FI iterations per setting (`--iterations=5`), for each layer and fault model.
- Prints golden confidence of output prediction (`--check_confidence`).
- Logs tensor outputs for attention analysis and writes diffs into `diff_results/` (`--check_attention`), which is used for the attention-focused analysis in the paper.

Important layer-range note:
- In the current code, `end_layer` is exclusive (`range(start_layer, end_layer)`).
- So `--start_layer=0 --end_layer=23` runs layers `0..22` (23 layers total).

You can inspect all available options with:

```bash
python run_fi_vit.py -h
```


### 5.2 Segmentation quick campaign (optional)

Example command:

```bash
python run_fi_segmentation.py \
  --iterations=5 \
  --imageindex=0 \
  --start_layer=0 \
  --end_layer=6 \
  --architecture=unet \
  --model_type=small
```

Notes:
- Supported architectures: `unet`, `deeplab`.
- Supported model sizes: `small`, `large`.
- This is a minimum working run; for stable paper-scale numbers, use more images/layers/iterations.


### 5.3 Result processing and interpretation

The ViT/segmentation FI scripts produce CSV files in `results/` containing per-layer and per-fault-model statistics (including SDC/error counts) for fault models `single`, `small-box`, `medium-box`, and `cpu`.

Run:

```bash
python getFIT.py
```

This generates:
- `Full_*.csv`: expanded per-layer files including FIT estimation columns.
- `ByLayerType_*.csv`: aggregation by layer type.
- `ByFaultType_*.csv`: aggregation by fault type.

To analyze attention-level effects from `diff_results/`, run:

```bash
python process_attention_results.py
```

This prints summary statistics of observed differences at the configured attention observation points (for the ViT attention analysis path used in the paper).


you can also inspect runner arguments:

```bash
python run_fi_vit.py -h
python run_fi_segmentation.py -h
```

Common controls:
- `--iterations`: injections per `(layer, fault_type)` configuration.
- `--imageindex`: run one sample only (faster for artifact checks).
- `--start_layer`, `--end_layer`: FI layer interval (`end_layer` exclusive).

## 6. Larger Campaigns

`execute_fi.sh` is intended as a long-running multi-model campaign template takes multiple days to complete.
`execute_fi_small.sh` is a small and less accurate subset of `execute_fi.sh` that takes about 2 hours.
