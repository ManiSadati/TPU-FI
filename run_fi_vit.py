import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from typing import Union

from common_tpu import load_model, load_input_data, run_inference
from utils import copy_tf_tensor, log_and_crash, attention_calculation, which_attention_layer2log

from attention_fc_graph_map import build_attention_fc_map_from_tflite, get_logged_layers_attention, AttentionMapping


from fi_runner import FIRunConfig, FITask, run_fault_injection


def parse_args():
    parser = argparse.ArgumentParser(description="TPU-FI setup (ViT)")
    parser.add_argument("--model_p", "-mp", default="8", choices=["8", "16"],
                        help="Model precision (8 or 16). Defaults to 8-bit model.")
    parser.add_argument("--iterations", "-it", default=1000, type=int)
    parser.add_argument("--testsamples", "-n", default=32, type=int)
    parser.add_argument("--check_attention", "-attention", action="store_true")
    parser.add_argument("--start_layer", default=0, type=int)
    parser.add_argument("--end_layer", default=662, type=int)
    parser.add_argument("--imageindex", "-idx", type=int)

    args = parser.parse_args()

    args.input = f"./benchmarks/vit/inputs/vit_base_{args.model_p}_images.npy"
    args.model = f"./benchmarks/vit/models/vit{args.model_p}_p{args.model_p}.tflite"

    if args.model_p == "8":
        args.end_layer = min(191, args.end_layer)

    return args



def are_equal(lhs: tf.Tensor, rhs: tf.Tensor, threshold: Union[None, float]) -> bool:
    if threshold is not None:
        return np.all(np.abs(lhs.numpy() - rhs.numpy()) <= threshold)
    return np.all(tf.equal(lhs, rhs))



def main():
    args = parse_args()

    print(f"running model {Path(args.model).stem}")

    interpreter = load_model(args.model, cpu=True)

    images = load_input_data(args.input)

    n_images = min(args.testsamples, len(images))
    img_indices = [args.imageindex] if args.imageindex is not None else list(range(n_images))


    attn_map = AttentionMapping("head_fc_mapping_3fc_exec.json")

    def get_logged_layers(fi_layer: int):
        if args.check_attention:
            ll = attn_map.get_logged_layers_attention(fi_layer)
            if(ll == None):
                ll = []
            return ll
        return []

    def prepare_input(img_index: int):
        return images[img_index]

    def infer(image):
        return run_inference(interpreter, image, None)

    def make_golden(output):
        return copy_tf_tensor(output)

    def compare(out_obj, golden_obj):
        # error = tensor not equal, sdc = argmax differs
        equal = are_equal(out_obj, golden_obj, None)
        is_error = (not equal)
        if(is_error == False):  # early exit so that we don't calculate sdc
            return is_error, False 
        is_sdc = (golden_obj.argmax() != out_obj.argmax())
        return is_error, is_sdc

    task = FITask(
        name="vit",
        prepare_input=prepare_input,
        infer=infer,
        make_golden=make_golden,
        compare=compare,
        get_logged_layers=get_logged_layers,
    )

    os.makedirs("./results", exist_ok=True)
    cfg = FIRunConfig(
        csv_filename=f"./results/FI-vit-{args.model_p}-results{ "(img"+str(args.imageindex)+")" if args.imageindex is not None else ""}.csv",
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        max_iterations=args.iterations,
        img_indices=img_indices,
    )

    run_fault_injection(task, cfg)
    print(f"Results saved in {cfg.csv_filename}")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_and_crash(fatal_string=f"EXCEPTION:{e}")