#!/bin/bash


python run_fi_vit.py --imageindex 0 --iterations=10 --model_p=16 --check_confidence --check_attention --end_layer=22
python run_fi_segmentation.py --imageindex 0 --architecture=unet --model_type=small --iterations=10 --end_layer=3
python run_fi_segmentation.py --imageindex 0 --architecture=unet --model_type=large --iterations=10 --end_layer=3
python run_fi_segmentation.py --imageindex 0 --architecture=deeplab --model_type=small --iterations=10 --end_layer=3
python run_fi_segmentation.py --imageindex 0 --architecture=deeplab --model_type=large --iterations=10 --end_layer=3


for i in $(seq 0 3); do
  python run_fi_vit.py --imageindex $i --iterations=5 --model_p=16 --check_confidence --check_attention --end_layer=50
done



python getFIT.py
python process_attention_results.py