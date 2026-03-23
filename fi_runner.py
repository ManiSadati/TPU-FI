# fi_runner.py
"""
Shared FI loop for TPU-FI.

This module centralizes the *experiment engine*:
- per-layer profiling (fi_init_profile + run golden)
- per-fault-type injections (fi_init_inject + run faulty)
- per-iteration bookkeeping (SDC, Critical SDC)
- optional tensor-dump post-processing (fi_post_process)

Model-specific behavior is injected via callbacks (task interface):
- how to prepare input
- how to run inference
- how to create golden representation
- how to compute (critical) SDC

"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Any

from fi_config import (
    init_fi,
    reset_fi_folder,
    fi_init_profile,
    get_dims,
    fi_init_inject,
    fi_post_process,
)


# -----------------------------
# Types / Interfaces
# -----------------------------

# What we store per image after profiling.
# golden_dims is what get_dims() returns (a list of strings from fi/dimension.txt).
GoldenEntry = Tuple[Any, List[str], int]  # (golden_obj, golden_dims, img_index)

# Callback types
PrepareInputFn = Callable[[int], Any]
InferFn = Callable[[Any], Any]
MakeGoldenFn = Callable[[Any], Any]
CompareFn = Callable[[Any, Any], Tuple[bool, bool]]  # (is_error, is_sdc)
LoggedLayersFn = Callable[[int], List[int]]  # per fi_layer


@dataclass
class FIRunConfig:
    csv_filename: str
    start_layer: int
    end_layer: int
    max_iterations: int
    fault_types: Sequence[str] = ("single", "small-box", "medium-box", "cpu")

    # If provided, only run these image indices (e.g. [7])
    img_indices: Optional[Sequence[int]] = None


@dataclass
class FITask:
    """
    A model/task adapter.
    Implement these as plain callables; no inheritance required.
    """
    name: str

    prepare_input: PrepareInputFn
    infer: InferFn
    make_golden: MakeGoldenFn
    compare: CompareFn

    # Optional: choose which internal layers to tensor-dump for a given fi_layer.
    # Return [] to disable logging.
    get_logged_layers: LoggedLayersFn = lambda _fi_layer: []


# -----------------------------
# Runner
# -----------------------------

def run_fault_injection(task: FITask, cfg: FIRunConfig) -> None:
    """
    Run fault injection from cfg.start_layer .. cfg.end_layer-1 and write CSV.

    CSV columns match your existing scripts:
    ["layer", "name", "type", "total runs", "sdc_count", "critical_sdc_count", "critical_sdc_rate",
     "d(out_c)", "layer area", "num_ops"]
    """
    init_fi()

    with open(cfg.csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "name", "type", "total runs", "sdc_count",
            "critical_sdc_count", "critical_sdc_rate", "d(out_c)", "layer area", "num_ops"
        ])

        for fi_layer in range(cfg.start_layer, cfg.end_layer):
            reset_fi_folder()
            logged_layers = task.get_logged_layers(fi_layer) or []

            # Decide which images to process for this layer.
            if cfg.img_indices is None:
                raise ValueError(
                    "cfg.img_indices is None. Pass explicit indices (e.g. range(n_images)) "
                    "from the model script so fi_runner stays model-agnostic."
                )

            # -----------------------------
            # Profiling phase (goldens)
            # -----------------------------
            print("----- PROFILING -----")
            golden_list: List[GoldenEntry] = []
            for img_index in cfg.img_indices:
                model_input = task.prepare_input(img_index)

                # This resets files + sets mode.txt to profiling and also sets layer_num.txt = 0.
                logged_layers = task.get_logged_layers(fi_layer)
                fi_init_profile(fi_layer, img_index, logged_layers)

                raw_output = task.infer(model_input)
                golden_obj = task.make_golden(raw_output)
                golden_dims = get_dims()  # from fi/dimension.txt written by kernel


                golden_list.append((golden_obj, golden_dims, img_index))
                del raw_output

                        

            # -----------------------------
            # Injection phase
            # -----------------------------
            print("----- INJECTION -----")
            for fi_type in cfg.fault_types:
                print(f"[{task.name}] fi_layer={fi_layer} fi_type={fi_type}")

                layer_name = ""
                total_runs = 0
                sdc_count = 0
                crit_sdc_count = 0
                c = -1
                layer_area = -1
                num_ops = -1

                for it in range(cfg.max_iterations):
                    print(f"  iter {it}/{cfg.max_iterations}")

                    for golden_obj, golden_dims, img_index in golden_list:
                        model_input = task.prepare_input(img_index)

                        # Writes layer_num.txt = 0 and writes locations + mode injection line
                        layer_name, status, c, layer_area, num_ops = fi_init_inject(
                            fi_layer, img_index, fi_type, it, golden_dims
                        )
                        if status == -1:
                            # e.g., invalid box sizes; skip this run
                            continue

                        raw_output = task.infer(model_input)
                        out_obj = task.make_golden(raw_output)  # same transform as golden
                        del raw_output

                        is_error, is_sdc = task.compare(out_obj, golden_obj)

                        total_runs += 1
                        if is_error:
                            sdc_count += 1
                        if is_sdc:
                            crit_sdc_count += 1

                if total_runs > 0:
                    crit_sdc_rate = crit_sdc_count / total_runs
                    writer.writerow([
                        fi_layer, layer_name, fi_type, total_runs, sdc_count,
                        crit_sdc_count, crit_sdc_rate, c, layer_area, num_ops
                    ])

                # Post-process tensor dumps (diff_results/*.npy)
                for _golden_obj, _golden_dims, img_index in golden_list:
                    fi_post_process(logged_layers, fi_layer, img_index, [fi_type], cfg.max_iterations)