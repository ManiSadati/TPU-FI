#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np

import os
import sys

from pathlib import Path

def output_tensor(interpreter, i):
	"""Gets a model's ith output tensor.
	Args:
	  interpreter: The ``tf.lite.Interpreter`` holding the model.
	  i (int): The index position of an output tensor.
	Returns:
	  The output tensor at the specified position.
	"""
	return interpreter.tensor(interpreter.get_output_details()[i]['index'])()

def input_details(interpreter, key):
	"""Gets a model's input details by specified key.
	Args:
	  interpreter: The ``tf.lite.Interpreter`` holding the model.
	  key (int): The index position of an input tensor.
	Returns:
	  The input details.
	"""
	return interpreter.get_input_details()[0][key]

def input_tensor(interpreter):
	"""Gets a model's input tensor view as numpy array of shape (height, width, 3).
	Args:
	  interpreter: The ``tf.lite.Interpreter`` holding the model.
	Returns:
	  The input tensor view as :obj:`numpy.array` (height, width, 3).
	"""
	tensor_index = input_details(interpreter, 'index')
	return interpreter.tensor(tensor_index)()#[0]

def set_interpreter_input_double(interpreter, a, b):
	"""Copies data to a model's input tensor.
	Args:
	  interpreter: The ``tf.lite.Interpreter`` to update.
	  data: The input tensor.
	"""
	input_details = interpreter.get_input_details()
	interpreter.set_tensor(input_details[0]['index'], a)
	interpreter.set_tensor(input_details[1]['index'], b)

def set_interpreter_input_single(interpreter, data):
	"""Copies data to a model's input tensor.
	Args:
	  interpreter: The ``tf.lite.Interpreter`` to update.
	  data: The input tensor.
	"""
	input_details = interpreter.get_input_details()
	interpreter.set_tensor(input_details[0]['index'], data)

def create_interpreter(model_file, cpu=True, device=":0"):
	"""
		Pablo's code (with some simplifications).

		Returns the interpreter with the loaded model from the file.

	Args:
		model_file: The (.tflite) file with the model
		cpu: Whether to use CPU or TPU interpreter
		device: Which TPU to use. If CPU flag is true, this is ignored.
	
	Returns:
		interpreter: The interpreter created for CPU or TPU.
	"""
	if cpu:
		interpreter = tf.lite.Interpreter(model_file,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.
    	BUILTIN_REF,)
	else:
		from pycoral.utils.edgetpu import make_interpreter
		interpreter = make_interpreter(model_file, device=device)

	return interpreter

def run_inference(interpreter, input_data, additional_data=None):
	"""
		Simple wrapper function to process some input and return only the output data.

	Args:
		interpreter: The CPU or TPU interpreter with the model loaded.
		input_data: The tensor with the input data to be fed into the model.

	Returns:
		output_data: The tensor with the output data from model inference.
	"""
	if additional_data is None:
		set_interpreter_input_single(interpreter, input_data)
	else:
		set_interpreter_input_double(interpreter, input_data, additional_data)

	interpreter.invoke()

	output_details = interpreter.get_output_details()[0]
	output_data = interpreter.tensor(output_details['index'])()

	return output_data

def load_model(*args, **kwargs):
	"""
		Simple wrapper to load a model from file and allocate tensors to interpreter.

	Args:
		See the args for create_interpreter()

	Returns:
		interpreter: The interpreter created for CPU or TPU (after allocating tensors).
	"""
	interpreter = create_interpreter(*args, **kwargs)
	interpreter.allocate_tensors()

	return interpreter

def load_data(file_path):
	np_data = np.load(file_path, allow_pickle=True)
	data = tf.convert_to_tensor(np_data)

	return data

def save_data(data, file_path):
	try:
		data = data.numpy()
	#if already numpy array
	except AttributeError:
		pass

	dir_path = Path(file_path).parent

	dir_path.mkdir(exist_ok=True, parents=True)

	np.save(file_path, data, allow_pickle=True)

# alias

load_input_data = load_data
load_golden = load_data
load_tokens = load_data
save_golden = save_data