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

This build includes the TFLite fault-injection implementation in `tensorflow/lite/kernels/internal/reference/integer_ops/fault_injection.h`, along with modifications to several TFLite kernels, mainly under `tensorflow/lite/kernels/internal/reference/integer_ops/`.

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
*Expected Activity Time: 5 minutes*

This section gives one concrete example of how to read the generated files and how they connect to the analysis in the paper. The important point is that Section 5 uses the same workflow as the paper, but on a much smaller campaign. The files and metrics are therefore interpreted in the same way, even though the resulting numbers are only for demonstration.

<br>

If you ran the ViT command from Section 5.1 exactly as written, the main raw output will be `results/FI-vit-16-results(img0).csv`. If you change the run arguments, the output filename may change accordingly. This CSV contains one row per `(layer, fault type)` pair. Each layer appears four times, once for each fault model: `single`, `small-box`, `medium-box`, and `cpu`. The main columns are:
- `layer`: layer index where the fault was injected.
- `name`: TensorFlow/TFLite kernel or layer type for that row, such as `FullyConnected` or `BroadcastMul6DSlow` (MatMul).
- `type`: fault model used for that row.
- `total runs`: number of injections executed for that `(layer, fault type)` pair.
- `sdc_count`: number of runs whose output differs from the golden run.
- `critical_sdc_count`: number of critical SDC events. For ViT16, this means the predicted output class changed. For segmentation, this corresponds to more than 1% of output classifications changing.
- `critical_sdc_rate`: `critical_sdc_count / total runs`.
- `d(out_c)`, `layer area`, `num_ops`: layer descriptors used later for normalization and FIT estimation.

For example, the first 7 lines look like:

| layer | name | type | total runs | sdc_count | critical_sdc_count | critical_sdc_rate | d(out_c) | layer area | num_ops |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | FullyConnected | single | 5 | 5 | 2 | 0.4 | 1 | 4096 | 3145728 |
| 0 | FullyConnected | small-box | 5 | 5 | 1 | 0.2 | 1 | 4096 | 3145728 |
| 0 | FullyConnected | medium-box | 5 | 5 | 3 | 0.6 | 1 | 4096 | 3145728 |
| 0 | FullyConnected | cpu | 5 | 5 | 1 | 0.2 | 1 | 4096 | 3145728 |
| 1 | BroadcastMul6DSlow | single | 5 | 4 | 0 | 0.0 | 1 | 4096 | 4096 |
| 1 | BroadcastMul6DSlow | small-box | 5 | 4 | 1 | 0.2 | 1 | 4096 | 4096 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

Note that your results may differ because the fault injections are random and the demo uses a relatively small number of injections, so the measured rates are not expected to be fully stable.

<br>

After running `getFIT.py` (Section 5.3), the file `results/Full_FI-vit-16-results(img0).csv` adds derived columns such as `sdc_rate`, `num_ops_limited`, `portion_of_tpu`, `fault_type_fit_rate`, `layer_vs_fault_fit_rate`, `fit_times_avf`, and `fit_times_avf_critical`. These convert the raw FI outcomes into FIT-oriented estimates. In practice:
- `fit_times_avf` is the estimated FIT contribution of that row.
- `fit_times_avf_critical` is the critical-only FIT contribution of that row.
- `Full_*.csv` is the most direct file for layer-by-layer analysis similar to Figure 10.

For the same ViT example, the first 7 lines are:

| layer | name | type | total runs | sdc_count | critical_sdc_count | critical_sdc_rate | d(out_c) | layer area | num_ops | sdc_rate | num_ops_limited | portion_of_tpu | fault_type_fit_rate | layer_vs_fault_fit_rate | fit_times_avf | fit_times_avf_critical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | FullyConnected | single | 5 | 5 | 2 | 0.4 | 1 | 4096 | 3145728 | 1.0 | 65536 | 0.0052083 | 13.41935484 | 0.069892473125 | 0.069892473125 | 0.0279569892499 |
| 0 | FullyConnected | small-box | 5 | 5 | 1 | 0.2 | 1 | 4096 | 3145728 | 1.0 | 65536 | 0.0052083 | 3.634408602 | 0.01892921146875 | 0.01892921146875 | 0.003785842 |
| 0 | FullyConnected | medium-box | 5 | 5 | 3 | 0.6 | 1 | 4096 | 3145728 | 1.0 | 65536 | 0.0052083 | 8.946236559 | 0.046594982078125 | 0.046594982078125 | 0.0279569892468 |
| 0 | FullyConnected | cpu | 5 | 5 | 1 | 0.2 | 1 | 4096 | 3145728 | 1.0 | 65536 | 0.0052083 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1 | BroadcastMul6DSlow | single | 5 | 4 | 0 | 0.0 | 1 | 4096 | 4096 | 0.8 | 4096 | 0.00032552 | 13.41935484 | 0.0043682795703125 | 0.00349462365624 | 0.0 |
| 1 | BroadcastMul6DSlow | small-box | 5 | 4 | 1 | 0.2 | 1 | 4096 | 4096 | 0.8 | 4096 | 0.00032552 | 3.634408602 | 0.001183075716796875 | 0.0009464605734375 | 0.0002366151433 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

<br>

`getFIT.py` also produces two summary files that are easier to compare against the paper:
- `results/ByFaultType_Full_FI-vit-16-results(img0).csv`: aggregates by fault model. This is the easiest file to inspect when asking which of `single`, `small-box`, `medium-box`, or `cpu` produces the largest average SDC or FIT contribution. This is the most natural demo file for reproducing the same type of fault-type comparison used in Figures 6, 7, 8, and 11.
- `results/ByLayerType_Full_FI-vit-16-results(img0).csv`: aggregates by layer type. This is the easiest file to inspect when comparing different layer types such as `FullyConnected` vs. `BroadcastMul6DSlow`, following the same type of analysis used in Figure 9.

<br>

Finally, by including `--check_confidence` and `--check_attention` flags in `run_fi_vit.py` run (Section 5.1), the framework also prints the golden confidence information (for Figure 11) and saves tensor-difference logs in `diff_results/`. Running `process_attention_results.py` summarizes these differences into `attention_results_exec_top8/top_heads_summary.txt`, following the same processing path used for the attention-head analysis associated with Figure 12.

<br>

Note that this section is intended for demo purposes only, so the numbers from this small run should not be treated as accurate paper-scale results. If the same scripts are run with more iterations, more images, and full layer coverage as described in Section 6, then the same workflow can be used to obtain the paper-scale results.

## 6. Larger Campaigns (Optional)
**Not recommended for quick demos.**
*Expected Activity Time: Multiple Days*

`execute_fi.sh` is intended as a long-running multi-model campaign template takes multiple days to complete.
`execute_fi_small.sh` is a small and less accurate subset of `execute_fi.sh` that takes about 2 hours.
