import os
import numpy as np
import tensorflow as tf
import argparse
import csv
from pathlib import Path
from PIL import Image
from typing import Union

from utils import copy_tf_tensor, log_and_crash, Timer
from fi_config import *
from common_tpu import load_model, load_input_data

# === Remap model output to binary (pet vs background)
def remap_model_output(mask_pred):
    # 0 = pet, 1 = background, 2 = edge → pet = 1, background = 0
    return np.where(mask_pred == 1, 0, 1)

# === Utility: Critical SDC Checker ===
def is_critical_sdc(faulty_mask, golden_mask):
    total_pixels = golden_mask.size
    changed_pixels = np.sum(faulty_mask != golden_mask)
    pixel_diff_ratio = changed_pixels / total_pixels
    print(pixel_diff_ratio)
    if pixel_diff_ratio > 0.01:
        return True

    golden_classes = set(np.unique(golden_mask))
    faulty_classes = set(np.unique(faulty_mask))
    return golden_classes != faulty_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Fault Injection on Segmentation Model")
    parser.add_argument("--model_type", "-m", default="1",
                        help="Model type (1 or 2). Defaults to model 1.")
    parser.add_argument("--architecture", "-a", default="unet",
                        help="Model architecture (unet or deeplab). Defaults to unet model.")
    parser.add_argument("--input", "-i", default="./benchmarks/segmentation/unet/inputs",
                        help="Path to input image folder (default: ./benchmarks/segmentation/unet/inputs)")
    parser.add_argument("--iterations", "-it", default=200, type=int,
                        help="Number of FI iterations per image")
    parser.add_argument("--start_layer", "-start_layer", default=0, type=int,
                        help="start_layer")
    parser.add_argument("--end_layer", "-end_layer", default=56, type=int,
                        help="end_layer")
    parser.add_argument("--generate", "-gen", action="store_true",
                        help="Disable golden comparison (e.g., dry-run)")
    parser.add_argument("--imageindex", "-img", type=int, default=None,
                        help="Index of a single image to process (0-based)")
    return parser.parse_args()

def run_inference(interpreter, image_np):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = np.expand_dims(image_np.astype(np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output

def load_images_from_folder(input_path, target_size):
    # Load preprocessed numpy array of images
    images = np.load(input_path)
    names = [f"image_{i}.npy" for i in range(len(images))]
    return images, names

def run_fault_injection(interpreter, images, names, max_iterations, start_layer, end_layer, csv_filename, arch, image_index=None):
    fault_types = ["single", "small-box", "medium-box"]

    init_fi()
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["layer", "name", "type", "total runs", "errors", "sdc_count", "sdc_rate", "d(out_c)", "layer area", "num_ops"])

        for fi_layer in range(start_layer, end_layer):
            reset_fi_folder()
            img_indices = [image_index] if image_index is not None else range(len(images))
            golden_list = []
            logged_layers = [fi_layer] # this needs to change, I only log the layer FI happens in.
            for img_index in img_indices:
                image = images[img_index, 0, :, :, :]
                
                fi_init_profile(fi_layer, img_index, logged_layers)
                golden = run_inference(interpreter, image)
                
                if(arch == "unet"):
                    golden = np.argmax(golden, axis=-1) if golden.ndim == 3 else golden
                    golden_bin = remap_model_output(golden)
                if(arch == "deeplab"):
                    golden_bin = golden
                golden_dims = get_dims()
                golden_list.append((golden_bin, golden_dims, img_index))

            for fi_type in fault_types:
                print("fi layer", fi_layer,"fi type", fi_type)
                layer_name, total_runs, errors, sdc_count = "", 0, 0, 0
                layer_area, num_ops, status = -1, -1, 0

                for it in range(max_iterations):
                    print(it,"/",max_iterations)
                    for golden_bin, golden_dims, img_index in golden_list:
                        image = images[img_index, 0, :, :, :]
                        layer_name, status, c , layer_area, num_ops = fi_init_inject(fi_layer, img_index, fi_type, it, golden_dims)
                        if status == -1:
                            continue

                        output = run_inference(interpreter, image)
                        if(arch == "unet"):
                            output = np.argmax(output, axis=-1) if output.ndim == 3 else output
                            output_bin = remap_model_output(output)
                        if(arch == "deeplab"):
                            output_bin = output
                        total_runs += 1
                        if not np.array_equal(output_bin, golden_bin):
                            errors += 1
                        if is_critical_sdc(output_bin, golden_bin):
                            sdc_count += 1

                if total_runs > 0:
                    sdc_rate = sdc_count / total_runs
                    writer.writerow([fi_layer, layer_name, fi_type, total_runs, errors, sdc_count, sdc_rate, c, layer_area, num_ops])
                for _, golden_dims, img_index in golden_list:
                    fi_post_process(logged_layers, fi_layer, img_index, [fi_type], max_iterations)

def main():
    args = parse_args()
    model_path = ""
    input_path = ""
    if args.architecture == "deeplab":
        input_path = "./benchmarks/segmentation/deeplab/inputs/deeplabv3_mnv2_pascal_quant_cityscape_pascalvoc_inputs.npy"
        args.end_layer = min(args.end_layer, 55)
        if args.model_type == "1":
            model_path = "./benchmarks/segmentation/deeplab/models/deeplabv3_mnv2_pascal_quant.tflite"
        elif args.model_type == "2":
            model_path = "./benchmarks/segmentation/deeplab/models/deeplabv3_mnv2_dm05_pascal_quant.tflite"
    elif args.architecture == "unet":
        if args.model_type == "1":
            input_path = "./benchmarks/segmentation/unet/inputs/oxford_images_128.npy"
            model_path = "./benchmarks/segmentation/unet/models/keras_post_training_unet_mv2_128_quant.tflite"
        elif args.model_type == "2":
            input_path = "./benchmarks/segmentation/unet/inputs/oxford_images_256.npy"
            model_path = "./benchmarks/segmentation/unet/models/keras_post_training_unet_mv2_256_quant.tflite"
    

    print(f"Running FI on: {Path(model_path).name} for input folder: {input_path}")

    timer = Timer()
    timer.tic()
    interpreter = load_model(model_path, cpu=True)
    input_size = interpreter.get_input_details()[0]['shape'][1:3]
    timer.toc()
    print(f"Model loaded in {timer.diff_time_str}")

    timer.tic()
    print(input_path)
    images, names = load_images_from_folder(input_path, target_size=tuple(input_size))
    timer.toc()
    print(f"Loaded {len(images)} images in {timer.diff_time_str}")

    os.makedirs("./results", exist_ok=True)
    run_fault_injection(
        interpreter=interpreter,
        images=images,
        names=names,
        max_iterations=args.iterations,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        csv_filename=f"./results/FI-segmentation-{args.model_type}-results.csv",
        arch=args.architecture,
        image_index=args.imageindex
    )

    print(f"Results saved in ./results/FI-segmentation-{args.model_type}-results.csv")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_and_crash(f"EXCEPTION: {e}")