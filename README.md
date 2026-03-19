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
- `third_party/`: a generated folder to hold the TensorFlow source subtree used for kernel instrumentation/build.
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
- Linux x86_64 (Ubuntu 20.04+)
- Docker
- 8+ GB RAM
- 16+ GB free disk

## 4. Setup and Build

### 4.1 Create a Docker image
*Expected Setup Time: 5 min*

Ensure that you are in the repository root (`/home/TPU-FI`).
Build an image from the [Dockerfile](Dockerfile) by running:

```bash
docker build -t tf_min_dev .
docker run --name tf_tfbuild -it -v $HOME/TPU-FI tf_min_dev
```

If the container is already built (future runs) use the following command instead:

```bash
docker start -ai tf_tfbuild
```

### 4.2 Clone, install prerequisites and configure build options for customized TensorFlow
*Expected Setup Time: 10 min*

Inside the container:

```bash
bash install.sh
```

Accept defaults by repeatedly pressing enter unless your environment requires changes.

### 4.3 Build customized TensorFlow
*Expected Setup Time: 30 min to 3 hours (Depending on Host System)*

Build wheel and install.

```bash
bash install_tensorflow.sh
```

## 5. Minimum Working Example (under 30 minutes)

### 5.1 ViT quick campaign
*Expected Activity Time: 5 min*

Ensure that you are in the repository root (`/home/TPU-FI`).
Run the command below.

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
**For a minimal demo run or a quick artifact evaluation, you may skip this subactivity and proceed directly to result processing (Section 5.3) below.**
*Expected Activity Time: 15 min*


Ensure that you are in the repository root (`/home/TPU-FI`).

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
*Expected Activity Time: 5 minutes*

The ViT/segmentation FI scripts produce CSV files in `results/` containing per-layer and per-fault-model statistics (including SDC/error counts) for fault models `single`, `small-box`, `medium-box`, and `cpu`.

Ensure that you are in the repository root (`/home/TPU-FI`).

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


### 5.4 Making sense of the Results

This section gives one concrete example of how to read the generated files. If you ran the ViT command from Section 5.1 exactly as written, the main raw output will be `results/FI-vit-16-results(img0).csv`.

This CSV contains one row per `(layer, fault type)` pair. Each layer appears four times, once for each fault model: `single`, `small-box`, `medium-box`, and `cpu`. The main columns are:
- `layer`: layer index where the fault was injected.
- `name`: TensorFlow/TFLite kernel or layer type for that row, such as `FullyConnected` or `BroadcastMul6DSlow`(MatMul).
- `type`: fault model used for that row.
- `total runs`: number of injections executed for that `(layer, fault type)` pair.
- `errors`: number of runs whose output tensor differs from the golden run (SDCs).
- `sdc_count`: number of critical SDC events. For ViT16, this means the predicted output class changed. For segmentation, this corresponds to more than 1% of output classifications changing.
- `sdc_rate`: `sdc_count / total runs`.
- `d(out_c)`, `layer area`, `num_ops`: layer descriptors used later for normalization and FIT estimation.


After running `python getFIT.py` (in Section 5.3), the file `results/Full_FI-vit-16-results(img0).csv` adds derived columns such as `portion_of_tpu`, `fault_type_fit_rate`, `layer_vs_fault_fit_rate`, `fit_times_avf`, and `fit_times_avf_critical`. These are the per-row FIT-based estimates used to move from raw SDC counts to reliability-oriented summaries. In practice:
- `fit_times_avf` is the estimated FIT contribution of that row.
- `fit_times_avf_critical` is the critical-only FIT contribution of that row.
- Higher values indicate rows that contribute more strongly to overall vulnerability.

`getFIT.py` also produces two summary files that are easier to compare against the paper:
- `results/ByFaultType_Full_FI-vit-16-results(img0).csv`: aggregates by fault model. This is the easiest file to inspect when asking which of `single`, `small-box`, `medium-box`, or `cpu` produces the largest average SDC or FIT contribution.
- `results/ByLayerType_Full_FI-vit-16-results(img0).csv`: aggregates by layer type. This is the easiest file to inspect when comparing classes of layers such as `FullyConnected` vs. `BroadcastMul6DSlow`.



For accurate paper-scale numbers, use the larger campaigns in Section 6 instead of the minimum working example.

## 6. Larger Campaigns (Optional)
**Not recommended for quick demos.**
*Expected Activity Time: Multiple Days*

`execute_fi.sh` is intended as a long-running multi-model campaign template takes multiple days to complete.
`execute_fi_small.sh` is a small and less accurate subset of `execute_fi.sh` that takes about 2 hours.
