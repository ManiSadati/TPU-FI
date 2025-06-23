import numpy as np
import time
import inspect
import traceback
import sys
from pathlib import Path


class LHLogger:
    def __init__(self):
        pass

    def start_log_file(self, *args, **kwargs):
        pass

    def end_log_file(self, *args, **kwargs):
        pass

    def start_iteration(self, *args, **kwargs):
        pass

    def end_iteration(self, *args, **kwargs):
        pass

    def perf(self, text, *args, **kwargs):
        print(f"PERF: {text}")

    def error(self, *args, **kwargs):
        print(f"ERROR: {args}")

    def info(self, *args, **kwargs):
        print(f"INFO: {args}")

    def debug(self, text, *args, **kwargs):
        print(f"DEBUG: {text}")

    def warning(self, text, *args, **kwargs):
        print(f"WARNING: {text}")


class Timer:
    def __init__(self):
        self.time_measure = 0

    def tic(self):
        self.time_measure = time.perf_counter()

    def toc(self):
        self.time_measure = time.perf_counter() - self.time_measure

    @property
    def diff_time(self):
        return self.time_measure

    @property
    def diff_time_str(self):
        return str(self)

    def __str__(self):
        return f"{self.time_measure:.4f}s"

    def __repr__(self):
        return str(self)

    def to_str(self, some_time):
        return f"{some_time:.4f}s"


def copy_tf_tensor(tensor):
    try:
        return np.copy(tensor.numpy())
    except AttributeError:
        return np.copy(tensor)


def log_and_crash(fatal_string: str) -> None:
    frame = inspect.stack()[1][0]
    info = inspect.getframeinfo(frame)
    fatal_log_string = f"SETUP_ERROR:{fatal_string} FILE:{info.filename}:{info.lineno} F:{info.function}\nTRACEBACK:{traceback.format_exc()}"
    print(fatal_log_string)
    with open(Path("/tmp/") / "TMP_CRASH_FILE", "w") as tmp_fp:
        tmp_fp.write(f"{fatal_log_string}\n")
    sys.exit(1)

def attention_calculation():
    # attention_layers{block_number: {head_number: [0: Q=X*W_q, 1: Q*K_t, 2: Fnal_layer=Softmax(Q*K_t).V, 3: K=X*W_k, 4: V=X*W_v]}}
    attention_layers = {0: {0: [47, 48, 50, 4, 12], 1: [43, 44, 46, 3, 11], 2: [39, 40, 42, 10, 22], 3: [35, 36, 38, 9, 21], 4: [31, 32, 34, 8, 20], 5: [27, 28, 30, 7, 19], 6: [23, 24, 26, 6, 14], 7: [15, 16, 18, 5, 13]}, 
                        1: {0: [108, 109, 111, 65, 73], 1: [104, 105, 107, 64, 72], 2: [100, 101, 103, 71, 83], 3: [96, 97, 99, 70, 82], 4: [92, 93, 95, 69, 81], 5: [88, 89, 91, 68, 80], 6: [84, 85, 87, 67, 75], 7: [76, 77, 79, 66, 74]}, 
                        2: {0: [169, 170, 172, 126, 134], 1: [165, 166, 168, 125, 133], 2: [161, 162, 164, 132, 144], 3: [157, 158, 160, 131, 143], 4: [153, 154, 156, 130, 142], 5: [149, 150, 152, 129, 141], 6: [145, 146, 148, 128, 136], 7: [137, 138, 140, 127, 135]}}

    # map_attention_layer = {layer_num: (block_num, head_num, layer_type)}
    map_attention_layer = {attention_layers[i][j][k]: (i,j,k) for i in range(3) for j in range(8) for k in range(5)} 
    return attention_layers, map_attention_layer


def which_attention_layer2log(layer_num, map_attention_layer, attention_layers):
    if layer_num not in map_attention_layer:
        return []
    block_num, head_num, layer_type = map_attention_layer[layer_num]
    log_layers = []
    if layer_type != 2:
        log_layers.append(layer_num)
    log_layers.append(attention_layers[block_num][head_num][2])
    for i in range(block_num + 1, 3):
        for j in range(8):
            log_layers.append(attention_layers[i][j][2])
    log_layers.append(190) # last layer
    return log_layers
