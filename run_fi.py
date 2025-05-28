import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import time
import csv
import sys
import traceback
import random
from pathlib import Path
from typing import Tuple, List, Union

import console_logger
from common_tpu import *
from fi_config import *
from utils import LHLogger, Timer, copy_tf_tensor, log_and_crash


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="D(L)eiT TPU radiation setup")
    parser.add_argument("--iterations", "-it", default=int(1), type=int)
    parser.add_argument("--testsamples", "-n", default=128, type=int)
    parser.add_argument("--generate", "-gen", action="store_true")
    parser.add_argument("--enableconsolelog", "-log", action="store_true")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--tokens", "-t")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--golden", "-g", required=True)
    parser.add_argument("--reload", "-r", action="store_true")
    parser.add_argument("--vit", "-v", "--notokens", "-nt", dest="vit", action="store_true")
    parser.add_argument("--log_interval", default=10, type=int)
    args = parser.parse_args()
    if args.generate:
        args.iterations = 1
    return args, [f"{k}={v}" for k, v in vars(args).items()]


def are_equal(lhs: tf.Tensor, rhs: tf.Tensor, threshold: Union[None, float]) -> bool:
    if threshold is not None:
        return np.all(np.abs(lhs.numpy() - rhs.numpy()) <= threshold)
    return np.all(tf.equal(lhs, rhs))



def run_fault_injection(interpreter, images, tokens, n_images, max_iterations, csv_filename):
    fault_types = ["single", "small-box", "medium-box"]
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["layer", "name", "type", "total runs", "error", "missclassification", "sdc rate", "layer area", "num_ops"])
        for fi_layer in range(190):
            for type in fault_types:
                layer_name, total_runs, err, miss_classification = "", 0, 0, 0
                layer_area, num_ops, status = -1, -1, 0
                for _ in range(max_iterations):
                    for img_index in range(n_images):
                        image = images[img_index]
                        fi_init_profile(fi_layer)
                        output = run_inference(interpreter, image, tokens)
                        golden = copy_tf_tensor(output)
                        del output

                        layer_name, status, layer_area, num_ops = fi_init_inject(fi_layer, type)
                        if status == -1:
                            continue

                        output = run_inference(interpreter, image, tokens)
                        equal = are_equal(output, golden, None)
                        total_runs += 1
                        if golden.argmax() != output.argmax():
                            miss_classification += 1
                        if not equal:
                            err += 1
                        del output

                if total_runs > -1:
                    sdc_rate = miss_classification / (total_runs + 1)
                    writer.writerow([fi_layer, layer_name, type, total_runs, err, miss_classification, sdc_rate, layer_area, num_ops])


def main():
    args, formatted_args = parse_args()
    logger = LHLogger()
    terminal_logger = console_logger.ColoredLogger(os.path.basename(__file__).replace(".py", "")) if args.enableconsolelog else None

    print(f"running model {Path(args.model).stem}")

    timer = Timer()
    timer.tic()
    interpreter = load_model(args.model, cpu=True)
    timer.toc()
    logger.perf(f"loaded_object:model load_time:{timer.diff_time_str}")
    if terminal_logger:
        terminal_logger.debug(f"Loaded model {args.model} ({timer.diff_time_str})")

    timer.tic()
    images = load_input_data(args.input)
    golden = load_golden(args.golden) if not args.generate else []
    tokens = load_tokens(args.tokens) if not args.vit else None
    timer.toc()
    logger.perf(f"loaded_object:data load_time:{timer.diff_time_str}")
    if terminal_logger:
        terminal_logger.debug(f"Loaded input and golden ({timer.diff_time_str})")

    if args.testsamples > len(images):
        logger.info(f"WARNING n_images:{args.testsamples} input_images:{len(images)}")
        if terminal_logger:
            terminal_logger.warning(f"Requested {args.testsamples} images but only {len(images)} available.")
    n_images = min(args.testsamples, len(images), len(golden) if not args.generate else float('inf'))

    run_fault_injection(
        interpreter=interpreter,
        images=images,
        tokens=tokens,
        n_images=n_images,
        max_iterations=args.iterations,
        csv_filename="fault_injection_results.csv"
    )

    print("Results saved in fault_injection_results.csv")
    logger.end_log_file()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_and_crash(fatal_string=f"EXCEPTION:{e}")
