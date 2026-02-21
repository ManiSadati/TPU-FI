import argparse
import os
import numpy as np

from fi_runner import FIRunConfig, FITask, run_fault_injection
from common_tpu import load_model
from utils import log_and_crash

def parse_args():
    parser = argparse.ArgumentParser(description="TPU-FI setup (Segmentation models)")
    parser.add_argument("--architecture", "-a", default="unet", choices=["unet", "deeplab"],
                        help="Model architecture (unet or deeplab). Defaults to unet model.")
    parser.add_argument("--model_type", "-m", default="small", choices=["small", "large"],
                        help="Model size (large or small). Defaults to the smaller model.")
    parser.add_argument("--iterations", "-it", default=1000, type=int)
    parser.add_argument("--start_layer", "-start_layer", default=0, type=int,
                        help="start_layer")
    parser.add_argument("--end_layer", "-end_layer", default=662, type=int,
                        help="end_layer")
    parser.add_argument("--imageindex", "-idx", type=int, help="Specify a single image index to process")
    args = parser.parse_args()


    if args.architecture == "deeplab":
        args.end_layer = min(args.end_layer, 55)

    args.model_path = None
    args.input_path = None

    if args.architecture == "deeplab":
        if args.model_type == "large":
            args.model_path = "./benchmarks/segmentation/deeplab/models/deeplabv3_mnv2_pascal_quant.tflite"
        elif args.model_type == "small":
            args.model_path = "./benchmarks/segmentation/deeplab/models/deeplabv3_mnv2_dm05_pascal_quant.tflite"
    elif args.architecture == "unet":
        if args.model_type == "large":
            args.model_path = "./benchmarks/segmentation/unet/models/keras_post_training_unet_mv2_256_quant.tflite"
        elif args.model_type == "small":
            args.model_path = "./benchmarks/segmentation/unet/models/keras_post_training_unet_mv2_128_quant.tflite"
    

    if args.architecture == "deeplab":
        args.input_path = "./benchmarks/segmentation/deeplab/inputs/deeplabv3_mnv2_pascal_quant_cityscape_pascalvoc_2_inputs.npy"
    elif args.architecture == "unet":
        if args.model_type == "small":
            args.input_path = "./benchmarks/segmentation/unet/inputs/oxford_images_128.npy"
        elif args.model_type == "large":
            args.input_path = "./benchmarks/segmentation/unet/inputs/oxford_images_256.npy"
        
        

    return args

    

def remap_model_output(mask_pred):
    return np.where(mask_pred == 1, 0, 1)

def is_critical_sdc(faulty_mask, golden_mask):
    total_pixels = golden_mask.size
    changed_pixels = np.sum(faulty_mask != golden_mask)
    pixel_diff_ratio = changed_pixels / total_pixels
    if pixel_diff_ratio > 0.01:
        return True
    return set(np.unique(faulty_mask)) != set(np.unique(golden_mask))


def run_inference(interpreter, image_np):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = np.expand_dims(image_np.astype(np.uint8), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]



def main():
    args = parse_args()


    images = np.load(args.input_path)
    interpreter = load_model(args.model_path, cpu=True)


    img_indices = [args.imageindex] if args.imageindex is not None else list(range(len(images)))

    def prepare_input(img_index: int):
        # keep your indexing convention
        return images[img_index, 0, :, :, :]

    def infer(image):
        return run_inference(interpreter, image)

    def make_golden(output):
        if args.architecture == "unet":
            pred = np.argmax(output, axis=-1) if output.ndim == 3 else output
            return remap_model_output(pred)
        if args.architecture == "deeplab":
            return output
        return output

    def compare(out_obj, golden_obj):  # early exit so that we don't calculate sdc
        is_error = (not np.array_equal(out_obj, golden_obj))
        if(is_error == False):
            return is_error, False
        is_sdc = is_critical_sdc(out_obj, golden_obj)
        return is_error, is_sdc

    task = FITask(
        name=f"seg-{args.architecture}",
        prepare_input=prepare_input,
        infer=infer,
        make_golden=make_golden,
        compare=compare,
        get_logged_layers=lambda _fi_layer: [],  # for now
    )

    os.makedirs("./results", exist_ok=True)
    cfg = FIRunConfig(
        csv_filename=f"./results/FI-segmentation-{args.architecture}-{args.model_type}-results.csv",
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