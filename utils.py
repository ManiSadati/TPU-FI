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
