#!/bin/bash


python run_fi_vit.py --model_p=16 --check_confidence
python run_fi_segmentation.py --architecture=unet --model_type=small
python run_fi_segmentation.py --architecture=unet --model_type=large
python run_fi_segmentation.py --architecture=deeeplab --model_type=small
python run_fi_segmentation.py --architecture=deeeplab --model_type=large


for i in $(seq 0 31); do
  python run_fi_vit.py --imageindex $i --model_p=16 --check_confidence --check_attention
done


python process_result.py
python attention_plot.py