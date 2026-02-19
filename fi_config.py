import random
import math
import numpy as np
import os, shutil


# -----------------------------------------------------------------------------
# TPU-FI control-plane utilities (Python side)
#
# This module manages the file-based handshake between:
#   (1) Python driver (chooses fault type / location / iteration / target layer)
#   (2) Instrumented TFLite kernels (C++ side) that:
#       - increment a global op counter (./fi/layer_num.txt)
#       - read mode + target layer from ./fi/mode.txt
#       - read injection locations from ./fi/locations.txt
#       - optionally write profiling dims to ./fi/dimension.txt
#       - optionally dump layer outputs to ./fi/output_*.txt
#
# Key directory contracts:
#   ./fi/
#     layer_num.txt          Global per-inference op counter (starts at 0)
#     mode.txt               "profiling ..." or "injection ..." + metadata
#     locations.txt          One line per fault site: "c x y bit"
#     dimension.txt          Profiling output: "<name> <layer> <c> <x> <y> <numOps>"
#     layer_output_list.txt  List of layer indices to dump during inference
#
#   ./diff_results/
#     diff_<...>.npy         Post-processed tensor diffs vs golden outputs
# -----------------------------------------------------------------------------


def init_fi():
    """Initialize top-level output directories for post-processing.

    This is typically called once before a full fault-injection campaign.
    """
    shutil.rmtree("./diff_results", ignore_errors=True)
    os.makedirs("./diff_results", exist_ok=True)


def reset_fi_folder():
    shutil.rmtree("./fi", ignore_errors=True)
    os.makedirs("./fi", exist_ok=True)
    
def reset_files():
    """Create/clear all control files used by the C++ fault injection runtime.

    Note: We always create empty files even if not used in a run so that C++
    code can safely open() them.
    """
    open("./fi/layer_num.txt", "w").close()
    open("./fi/mode.txt", "w").close()
    open("./fi/locations.txt", "w").close()
    open("./fi/dimension.txt", "w").close()
    open("./fi/layer_output_list.txt", "w").close()

def fi_init_profile(layer, img_index, layer_output_list):
    """Prepare the next inference run for PROFILING mode.

    Profiling mode is used to record:
      - output tensor dimensions of the selected target op ("layer")
      - a rough op-count (numOps) computed inside kernels (e.g., MAC count)

    Args:
      layer:       Target layer index (matches the TFLiteC++ global op counter).
      img_index:   Which input sample is being run (propagated to filenames).
      layer_output_list: list of layer indices whose outputs should be dumped.
    """

    reset_files()

    # mode.txt format:
    #   profiling <fault_layer> <img_index> <fault_type> <iteration>
    # During profiling, fault_type is "None" and iteration is -1 by convention.
    with open("./fi/mode.txt", "w") as file:
        file.write(f"profiling {layer} {img_index} None -1\n")


    # layer_output_list.txt contains integers (layer indices) to dump.
    # C++ checks current_layer_num is in this list to decide whether to log output.
    with open("./fi/layer_output_list.txt", "w") as file:
        for layer_output in layer_output_list:
            file.write(f"{layer_output}\n")

    # reset the global op counter for this inference.
    # Each instrumented kernel invocation reads current value and increments it.
    with open("./fi/layer_num.txt", "w") as file:
        file.write("0\n")
        

def get_dims():
    """Read the last profiling result from ./fi/dimension.txt."""
    with open("./fi/dimension.txt", "r") as file:
        dimensions = file.read().split()
    return dimensions


def fi_init_inject(layer, img_index, type, it, dimensions):
    """Prepare the next inference run for INJECTION mode.

    This implements the fault model at the tensor level:
      - choose a fault shape/region (single / small-box / medium-box / cpu)
      - choose a channel and top-left corner (l_c, l_x, l_y)
      - sample a boolean mask within the box with probability 'prob'
      - for each True entry, write the injection location (c, x, y, bit) into ./fi/locations.txt

    Args:
      layer: target layer index to inject into.
      img_index: input sample index.
      type: fault model type string.
      it: iteration id (used in repeated trials).
      dimensions: list read from get_dims().

    Returns:
      (layer_name, status, c_size, layer_area, num_ops)
      status==0 means success; status==-1 means "skip" (e.g., invalid box).
    """

    # Define initial values for variables
    box_x = box_y = l_x = r_x = l_y = r_y = -1
    x_size = y_size = c_size = 0
    num_ops = 0
    prob = 0.0

    # Reset the global op counter for this inference run.
    with open("./fi/layer_num.txt", "w") as file:
        file.write("0\n")

    # Parse profiling output (dimensions list format):
    layer_name = dimensions[0]
    c_size = int(dimensions[2])
    x_size = int(dimensions[3])
    y_size = int(dimensions[4])
    num_ops = int(dimensions[5])
    print(f"Layer: {layer_name}, Channels: {c_size}, X Size: {x_size}, Y Size: {y_size}, Num Ops: {num_ops}")

    # -------------------------------------------------------------------------
    # Fault model parameterization:
    #
    # - "single"    : box area = 1, inject prob = 1.0
    # - "small-box" : area ∈ [41, 113], inject prob = 0.07
    # - "medium-box": area ∈ [949, 1351], inject prob = 0.035
    # - "cpu"       : (singl uniform bit flips), box area = 1, inject prob = 1.0
    #
    # Box sampling: choose a target "area", then sample (box_x, box_y) such that
    # box_x * box_y ≈ area, with constraints by x_size/y_size.
    # -------------------------------------------------------------------------
    if type == "single":
        box_x = box_y = 1
        prob = 1.0

    elif type == "small-box":
        area = random.randint(41, max(41,min(x_size * y_size,113)))
        if area > x_size * y_size:
            area = max(1,(x_size * y_size) // 2)
        box_y = random.randint(max(1,math.ceil(area/x_size)), min(y_size, area))
        box_x = max(1, area // box_y)
        if box_x > x_size:
            return layer_name, -1, c_size,  x_size * y_size, num_ops
        prob = 0.07

    elif type == "medium-box":
        area = random.randint(949, max(949,min(x_size * y_size,1351)))
        if area > x_size * y_size:
            area = x_size * y_size
            # print ("area: ",area)
            box_x = x_size
            box_y = y_size
            #return layer_name, -1, c_size,  x_size * y_size, num_ops
        else:
            box_y = random.randint(max(1,math.ceil(area/x_size)), min(y_size, area))
            box_x = max(1, area // box_y)
            # print ("box_x , x_size, box_y, y_size ",box_x, x_size, box_y, y_size)
            if box_x > x_size:
                return layer_name, -1, c_size,  x_size * y_size, num_ops
        prob = 0.035

    elif type == "cpu":
        box_x = box_y = 1
        prob = 1.0

    # Choose a random placement of the box within the x/y plane and a random channel.
    l_x = random.randint(0, x_size - box_x)
    l_y = random.randint(0, y_size - box_y)
    l_c = random.randint(0, c_size - 1)

    # Sample injection mask: each box cell becomes faulty with probability 'prob'.
    matrix = np.random.rand(box_x, box_y) < prob

    # Extract locations where the condition is True
    locs = np.argwhere(matrix)

    # locations.txt format:
    #   c x y bit
    #
    # Bit distribution:
    #   - 59%: bit=0
    #   - 41%: uniform bit in [1..7]
    #   - for cpu fault type: uniform bit in [0..7]
    with open("./fi/locations.txt", "w") as file:
        for loc in locs:
            fi_bit = 0 if random.random() <= 0.59 else random.randint(1, 7)
            if type == "cpu":
                fi_bit = random.randint(0, 7)
            file.write(f"{l_c} {loc[0] + l_x} {loc[1] + l_y} {fi_bit}\n")

    # mode.txt updated for injection run:
    #   injection <fault_layer> <img_index> <fault_type> <iteration>
    with open("./fi/mode.txt", "w") as file:
        file.write(f"injection {layer} {img_index} {type} {it}\n")
    
    return layer_name, 0, c_size, x_size * y_size, num_ops

def get_tensor_from_file(layer_output, fi_layer, img_index, type, it):
    """Load a dumped tensor output written by instrumented kernels.

    Expected file format:
      line 1: "<c_size> <x_size> <y_size>"
      remaining lines: flattened values written in kernel-specific loop order,
                       typically c-major then x then y (or kernel-defined).
    """
    file_path = f"./fi/output_{layer_output}-{fi_layer}-{img_index}-{type}-{it}.txt"
    if not os.path.exists(file_path):
        print("path not exists",(file_path))
        return None, -1
    with open(file_path, "r") as file:
        # read first line
        c_size, x_size, y_size = map(int, file.readline().split())
        output = file.read().splitlines()
        output = np.array([list(map(float, line.split())) for line in output])
        output = output.reshape((c_size, x_size, y_size))
    return output, 1



def fi_post_process(layer_output_list, fi_layer, img_index, fault_types, max_iterations):
    """Compute and save elementwise diffs between injected outputs and golden outputs.

    For each layer in layer_output_list:
      - read golden output: type="None", it=-1
      - for each fault type and each iteration:
          diff = injected - golden
          save to ./diff_results/diff_<...>.npy
    """
    for layer_output in layer_output_list:
        golden_tensor, status = get_tensor_from_file(layer_output, fi_layer, img_index, "None", -1)
        if status == -1:
            print("Golden tensor not found!")
            exit()
        for type in fault_types:
            for it in range(max_iterations):
                output_tensor, status = get_tensor_from_file(layer_output, fi_layer, img_index, type, it)
                if status == -1:
                    continue
                diff = output_tensor - golden_tensor
                np.save(f"./diff_results/diff_{layer_output}-{fi_layer}-{img_index}-{type}-{it}.npy", diff)
    return