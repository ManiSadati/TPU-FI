#!/bin/bash


python run_fi_vit.py --imageindex $i --iterations=10 --model_p=16 --check_confidence --check_attention --end_layer=22
python run_fi_segmentation.py --architecture=unet --model_type=small --iterations=10 --end_layer=3
python run_fi_segmentation.py --architecture=unet --model_type=large --iterations=10 --end_layer=3
python run_fi_segmentation.py --architecture=deeeplab --model_type=small --iterations=10 --end_layer=3
python run_fi_segmentation.py --architecture=deeeplab --model_type=large --iterations=10 --end_layer=3


for i in $(seq 0 31); do
  python run_fi_vit.py --imageindex $i --iterations=10 --model_p=16 --check_confidence --check_attention --end_layer=5
done


python process_result.py
python attention_plot.py