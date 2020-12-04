#!/bin/bash

# Potsdam- 3

python train.py --dataset Potsdam3  --output model_001 --batch_size 20  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10

python train.py --dataset Potsdam3  --output model_002 --batch_size 20  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True

python train.py --dataset Potsdam3  --output model_003 --batch_size 4  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 2 --spatial_invariance 10

# Potsdam

python train.py --dataset Potsdam  --output model_004 --batch_size 20  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10

python train.py --dataset Potsdam  --output model_005 --batch_size 10  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True

python train.py --dataset Potsdam  --output model_006 --batch_size 4  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 2 --spatial_invariance 10

# Coco-Stuff 3

python train.py --dataset CocoStuff3  --output model_007 --batch_size 30  --learning_rate 4e-5 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10

# Coco-Stuff

python train.py --dataset CocoStuff15  --output model_008 --batch_size 30  --learning_rate 6e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10