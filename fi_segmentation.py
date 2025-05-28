import os
import numpy as np
import tensorflow as tf
import argparse
import csv
from pathlib import Path
from PIL import Image
from typing import Union

from utils import copy_tf_tensor, log_and_crash, Timer
from fi_config import fi_init_profile, fi_init_inject
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
    parser.add_argument("--model", "-m", default="256",
                        help="Model size (256 or 128). Defaults to 256 model.")
    parser.add_argument("--input", "-i", default="./segmentation/inputs", help="Path to input image folder (default: ./segmentation/inputs)")
    parser.add_argument("--iterations", "-it", default=1, type=int, help="Number of FI iterations per image")
    parser.add_argument("--generate", "-gen", action="store_true", help="Disable golden comparison (e.g., dry-run)")
    return parser.parse_args()


def run_inference(interpreter, image_np):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = np.expand_dims(image_np.astype(np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    return output


def load_images_from_folder(model_input_size, folder_path, target_size):
    # image_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".jpg"))
    images, names = [], []

    # for fname in image_files:
    #     path = os.path.join(folder_path, fname)
    #     img = Image.open(path).convert('RGB')
    #     img = img.resize(target_size, Image.LANCZOS)
    #     images.append(np.array(img))
    #     names.append(fname)

    images = np.load(f"./segmentation/inputs/oxford_images_{model_input_size}.npy")

    return images, names


def run_fault_injection(interpreter, images, names, max_iterations, csv_filename):
    fault_types = ["single", "small-box", "medium-box"]
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["layer", "name", "type", "total runs", "errors", "sdc_count", "sdc_rate", "layer area", "num_ops"])

        for fi_layer in range(55,56):  # adjust as needed
            for fi_type in fault_types:
                layer_name, total_runs, errors, sdc_count = "", 0, 0, 0
                layer_area, num_ops, status = -1, -1, 0

                for _ in range(max_iterations):
                    for idx in range(len(images)):
                        image = images[idx,0,:,:,:]
                        fi_init_profile(fi_layer)

                        golden = run_inference(interpreter, image)
                        golden = np.argmax(golden, axis=-1) if golden.ndim == 3 else golden
                        golden_bin = remap_model_output(golden)

                        layer_name, status, layer_area, num_ops = fi_init_inject(fi_layer, fi_type)
                        if status == -1:
                            continue

                        output = run_inference(interpreter, image)
                        output = np.argmax(output, axis=-1) if output.ndim == 3 else output
                        output_bin = remap_model_output(output)

                        total_runs += 1
                        if not np.array_equal(output_bin, golden_bin):
                            errors += 1
                        if is_critical_sdc(output_bin, golden_bin):
                            sdc_count += 1
                        exit()

                if total_runs > 0:
                    sdc_rate = sdc_count / total_runs
                    writer.writerow([fi_layer, layer_name, fi_type, total_runs, errors, sdc_count, sdc_rate, layer_area, num_ops])


def main():
    args = parse_args()
    model_input_size = args.model
    model_path = f"./segmentation/models/keras_post_training_unet_mv2_{model_input_size}_quant.tflite"
    input_path = args.input
    print(f"Running FI on: {Path(model_path).name} for input folder: {input_path}")

    timer = Timer()
    timer.tic()
    interpreter = load_model(model_path, cpu=True)
    input_size = interpreter.get_input_details()[0]['shape'][1:3]
    timer.toc()
    print(f"Model loaded in {timer.diff_time_str}")

    timer.tic()
    images, names = load_images_from_folder(model_input_size, input_path, target_size=tuple(input_size))
    timer.toc()
    print(f"Loaded {len(images)} images in {timer.diff_time_str}")

    run_fault_injection(
        interpreter=interpreter,
        images=images,
        names=names,
        max_iterations=args.iterations,
        csv_filename="segmentation_fi_results.csv"
    )

    print("Results saved in segmentation_fi_results.csv")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_and_crash(f"EXCEPTION: {e}")