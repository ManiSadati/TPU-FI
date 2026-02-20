import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf

import console_logger
from common_tpu import load_model, load_input_data, run_inference, load_tokens
from utils import LHLogger, Timer, copy_tf_tensor, log_and_crash, attention_calculation, which_attention_layer2log

from fi_runner import FIRunConfig, FITask, run_fault_injection




def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="D(L)eiT TPU radiation setup")
    parser.add_argument("--model_p", "-mp", default="8",
                        help="Model precision (8 or 16). Defaults to 8-bit model.")
    parser.add_argument("--iterations", "-it", default=1000, type=int)
    parser.add_argument("--testsamples", "-n", default=32, type=int)
    parser.add_argument("--generate", "-gen", action="store_true")
    parser.add_argument("--enableconsolelog", "-log", action="store_true")
    parser.add_argument("--check_attention", "-attention", action="store_true")
    parser.add_argument("--start_layer", "-start_layer", default=0, type=int,
                        help="start_layer")
    parser.add_argument("--end_layer", "-end_layer", default=662, type=int,
                        help="end_layer")
    parser.add_argument("--tokens", "-t")
    parser.add_argument(
        "--golden", "-g",
        default="data/golden/vit_base_8_golden.npy"
    )
    parser.add_argument("--reload", "-r", action="store_true")
    parser.add_argument("--vit", "-v", "--notokens", "-nt", dest="vit", action="store_true", default=True)
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--imageindex", "-idx", type=int, help="Specify a single image index to process")
    args = parser.parse_args()
    if args.generate:
        args.iterations = 1
    args.input = f"./benchmarks/vit/inputs/vit_base_{args.model_p}_images.npy"
    args.model = f"./benchmarks/vit/models/vit{args.model_p}_p{args.model_p}.tflite"
    if args.model_p == "8":
        args.end_layer = min(191, args.end_layer) 
    return args, [f"{k}={v}" for k, v in vars(args).items()]



def are_equal(lhs: tf.Tensor, rhs: tf.Tensor, threshold: Union[None, float]) -> bool:
    if threshold is not None:
        return np.all(np.abs(lhs.numpy() - rhs.numpy()) <= threshold)
    return np.all(tf.equal(lhs, rhs))



def main():
    args, formatted_args = parse_args()
    logger = LHLogger()
    terminal_logger = console_logger.ColoredLogger(os.path.basename(__file__).replace(".py", "")) if args.enableconsolelog else None

    print(f"running model {Path(args.model).stem}")

    interpreter = load_model(args.model, cpu=True)

    images = load_input_data(args.input)

    n_images = min(args.testsamples, len(images))
    img_indices = [args.imageindex] if args.imageindex is not None else list(range(n_images))

    attention_layers, map_attention_layer = attention_calculation()

    def get_logged_layers(fi_layer: int):
        if args.check_attention:
            return which_attention_layer2log(fi_layer, map_attention_layer, attention_layers)
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
        csv_filename="./results/FI-vit-results.csv",
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        max_iterations=args.iterations,
        img_indices=img_indices,
    )

    run_fault_injection(task, cfg)
    print("Results saved in ./results/FI-vit-results.csv")
    logger.end_log_file()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_and_crash(fatal_string=f"EXCEPTION:{e}")