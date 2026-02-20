# run_fi_segmentation.py (your old fi_segmentation.py but loop removed)
import os
import numpy as np
from pathlib import Path

from fi_runner import FIRunConfig, FITask, run_fault_injection
from fi_config import init_fi
from common_tpu import load_model
from utils import Timer, log_and_crash


def remap_model_output(mask_pred):
    return np.where(mask_pred == 1, 0, 1)

def is_critical_sdc(faulty_mask, golden_mask):
    total_pixels = golden_mask.size
    changed_pixels = np.sum(faulty_mask != golden_mask)
    pixel_diff_ratio = changed_pixels / total_pixels
    if pixel_diff_ratio > 0.01:
        return True
    return set(np.unique(faulty_mask)) != set(np.unique(golden_mask))

def parse_args():
    # keep your existing args
    ...


def run_inference(interpreter, image_np):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = np.expand_dims(image_np.astype(np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

def load_images_from_folder(input_path, target_size):
    images = np.load(input_path)
    names = [f"image_{i}.npy" for i in range(len(images))]
    return images, names


def main():
    args = parse_args()

    # your existing model_path / input_path selection logic
    model_path = ...
    input_path = ...
    arch = args.architecture

    timer = Timer()
    timer.tic()
    interpreter = load_model(model_path, cpu=True)
    input_size = interpreter.get_input_details()[0]['shape'][1:3]
    timer.toc()

    images, names = load_images_from_folder(input_path, target_size=tuple(input_size))

    img_indices = [args.imageindex] if args.imageindex is not None else list(range(len(images)))

    def prepare_input(img_index: int):
        # keep your indexing convention
        return images[img_index, 0, :, :, :]

    def infer(image):
        return run_inference(interpreter, image)

    def make_golden(output):
        if arch == "unet":
            pred = np.argmax(output, axis=-1) if output.ndim == 3 else output
            return remap_model_output(pred)
        if arch == "deeplab":
            return output
        return output

    def compare(out_obj, golden_obj):
        is_error = (not np.array_equal(out_obj, golden_obj))
        is_sdc = is_critical_sdc(out_obj, golden_obj)
        return is_error, is_sdc

    task = FITask(
        name=f"seg-{arch}",
        prepare_input=prepare_input,
        infer=infer,
        make_golden=make_golden,
        compare=compare,
        get_logged_layers=lambda _fi_layer: [],  # for now
    )

    os.makedirs("./results", exist_ok=True)
    cfg = FIRunConfig(
        csv_filename=f"./results/FI-segmentation-{args.model_type}-results.csv",
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
        log_and_crash(f"EXCEPTION: {e}")