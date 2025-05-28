#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np

import argparse
import logging
import os
import time
import parse
from typing import Tuple, List, Union

#import console_logger

# TO-DO: better import
from common_tpu import *

import threading
import time
import os
import sys
import inspect
import traceback

def parse_corrupted_file_name(f):
	# yeah, there was a typo in 'BENCH'...
	# (now fixed)
	file_format = "BENCH_{log_bm}_IMG_ID_{img_id:d}_LOG_FILE_{log_file}_SDC_NUM_{sdc_number:d}.npy"

	vals = parse.parse(file_format, f)

	return vals

def bm_to_micro(bm):
	if bm == 'main16':
		return "vit16_MAIN_BLOCK_TO_LOGITS.tflite"
	if bm == 'mha16':
		return "vit16_MHA_from_image_TO_LOGITS.tflite"
	if bm == 'patch16':
		return "vit16_PATCH_ENCODING_TO_LOGITS.tflite"
	if bm == 'main8':
		return "vit8_MAIN_BLOCK_TO_LOGITS.tflite"
	if bm == 'mha8':
		return "vit8_MHA_from_image_TO_LOGITS.tflite"
	if bm == 'patch8':
		return "vit8_PATCH_ENCODING_TO_LOGITS.tflite"
	else:
		return None

def benchmark_to_golden(benchmark):
	if benchmark == 'patch16':
		return 'vit_base_16_golden_PATCH_ENCODING.npy'

	elif benchmark == 'main16':
		return 'vit_base_16_golden_MAIN_BLOCK.npy'

	elif benchmark == 'mha16':
		return 'vit_base_16_golden_MHA_from_image.npy'

	elif benchmark == 'patch8':
		return 'vit_base_8_golden_PATCH_ENCODING.npy'

	elif benchmark == 'main8':
		return 'vit_base_8_golden_MAIN_BLOCK.npy'

	elif benchmark == 'mha8':
		return 'vit_base_8_golden_MHA_from_image.npy'
	
	else:
		raise ValueError(f"Invalid benchmark: {benchmark}")

def bm_to_golden_logits(benchmark):
	vit16_micros = ['patch16', 'main16', 'mha16']
	vit8_micros = ['patch8', 'main8', 'mha8']

	if benchmark in vit8_micros:
		#return 'vit_base_8_golden.npy'
		return 'vit_base_8_golden_FINAL_BLOCK.npy'


	elif benchmark in vit16_micros:
		#return 'vit_base_16_golden.npy'
		return 'vit_base_16_golden_FINAL_BLOCK.npy'
	
	else:
		raise ValueError(f"Invalid benchmark: {benchmark}")

def benchmark_to_input(benchmark):
	vit16_from_image = ['patch16', 'main16', 'mha16']
	vit8_from_image = ['patch8', 'main8', 'mha8']

	if benchmark in vit8_from_image:
		return 'vit_base_8_images.npy'

	elif benchmark in vit16_from_image:
		return 'vit_base_16_images.npy'
	
	else:
		raise ValueError(f"Invalid benchmark: {benchmark}")

def load_model_from_bm(log_bm, model_path):
	model_file = bm_to_micro(log_bm)

	model_file_str = str(model_path / model_file)
	interpreter = load_model(model_file_str)

	return interpreter

def benchmark_to_logits_golden_file(log_bm):
	golden = benchmark_to_golden(log_bm)
	golden = golden.replace('.npy', '_TO_LOGITS.npy')
	return golden

def main():
	#corrupted_output_dir = Path('/home/carol/chipir_05_corrupted_outputs')
	corrupted_output_dir = Path('/home/carol/radef_06_corrupted_outputs')

	#corrupted_logits_dir = Path('/home/carol/chipir_05_logits_from_corrupted')
	corrupted_logits_dir = Path('/home/carol/radef_06_logits_from_corrupted')

	model_path = Path('/home/carol/tpu-rad/models')
	input_path = Path('/home/carol/tpu-rad/inputs')
	golden_path = Path('/home/carol/tpu-rad/golden')

	machine_dirs = sorted([d for d in os.listdir(corrupted_output_dir) if os.path.isdir(corrupted_output_dir/d)])

	validate_micro_models = False

	microbenchmarks = [
		'patch16',
		'main16',
		'mha16',

		'patch8',
		'main8',
		'mha8',
	]
	
	# DEBUG
	
	checked_bms = []

	computed_golden_per_bm = {}
	input_images_per_bm = {}

	for machine in machine_dirs:
		machine_logits_dir = corrupted_logits_dir / machine
		machine_logits_dir.mkdir(exist_ok=True, parents=True)

		machine_path = Path(corrupted_output_dir) / machine

		corrupted_output_files = sorted([f for f in os.listdir(machine_path) if f.endswith('.npy')])

		for corrupted_output_file in corrupted_output_files:
			file_info = parse_corrupted_file_name(corrupted_output_file)

			log_bm = file_info['log_bm']
			img_id = file_info['img_id']
			log_file = file_info['log_file']
			sdc_number = file_info['sdc_number']

			# ignore full models and FINAL benchmark

			if log_bm not in microbenchmarks:
				continue

			assert log_bm is not None

			# DEBUG
			if validate_micro_models:
				if log_bm in checked_bms:
					continue
				else:
					checked_bms.append(log_bm)

			interpreter = load_model_from_bm(log_bm, model_path)

			# check if we already loaded inputs for this benchmark
			# if not, load from file
			if log_bm not in input_images_per_bm:
				input_file = benchmark_to_input(log_bm)
				input_images = np.load(input_path / input_file)
				input_images_per_bm[log_bm] = input_images

			input_images = input_images_per_bm[log_bm]
			print(f"Input image for {log_bm} is {len(input_images)} images of size {input_images[0].shape}")

			# compute and store golden for these models if not already done
			if log_bm not in computed_golden_per_bm:
				computed_golden = []
				golden_file = benchmark_to_golden(log_bm)
				golden_outputs = np.load(golden_path / golden_file)

				for i in range(min(len(golden_outputs), len(input_images))):
					input_data = input_images[i]
					golden_input = golden_outputs[i]
					output = run_inference(interpreter, input_data, golden_input)
					computed_golden.append(np.copy(output))

					del output

				computed_golden_per_bm[log_bm] = np.asarray(computed_golden)

				logits_golden = benchmark_to_logits_golden_file(log_bm)
				corrupted_logits_golden_dir = corrupted_logits_dir / 'golden'
				corrupted_logits_golden_dir.mkdir(exist_ok=True, parents=True)
				np.save(corrupted_logits_golden_dir / logits_golden, computed_golden_per_bm[log_bm])
				print(f"Saved golden output for benchmark {log_bm}")

			input_data = input_images[img_id]

			corrupted_output = np.load(machine_path / corrupted_output_file)

			print(f"Running inference for {log_bm}, input data is {input_data.shape}, output is {corrupted_output.shape}")

			output = run_inference(interpreter, input_data, corrupted_output)

			output_file = corrupted_output_file.replace('.npy', '_TO_LOGITS.npy')

			np.save(machine_logits_dir / output_file, output)

			print(f"Saved corrupted logits to {output_file}")

			del output



if __name__ == '__main__':
	main()