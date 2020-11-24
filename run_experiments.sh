#!/bin/bash

# Potsdam3

echo "Starting: 2-layer model on Potsdam3"
python train.py --dataset Potsdam3 --output model_004 --batch_size 20 --learning_rate 1e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam3 --output model_005 --batch_size 20 --learning_rate 1e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam3 --output model_006 --batch_size 20 --learning_rate 1e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam3 --output model_007 --batch_size 20 --learning_rate 1e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
echo "Starting: 3-layer model on Potsdam3"
python train.py --dataset Potsdam3 --output model_008 --batch_size 10 --learning_rate 1e-5 --epochs 10 --num_res_layers 3 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam3 --output model_009 --batch_size 10 --learning_rate 1e-5 --epochs 10 --num_res_layers 3 || { echo "oopsie daisies"; exit 1; }
echo "Starting: 4-layer model on Potsdam3"
python train.py --dataset Potsdam3 --output model_010 --batch_size 4 --learning_rate 1e-5 --epochs 10 --num_res_layers 4 || { echo "oopsie daisies"; exit 1; }

# Potsdam

echo "Starting: 2-layer model on Potsdam"
python train.py --dataset Potsdam --output model_011 --batch_size 20 --learning_rate 2e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam --output model_012 --batch_size 20 --learning_rate 2e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam --output model_013 --batch_size 20 --learning_rate 2e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam --output model_014 --batch_size 20 --learning_rate 2e-5 --epochs 10 --num_res_layers 2 || { echo "oopsie daisies"; exit 1; }
echo "Starting: 3-layer model on Potsdam"
python train.py --dataset Potsdam --output model_015 --batch_size 10 --learning_rate 2e-5 --epochs 10 --num_res_layers 3 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam --output model_016 --batch_size 10 --learning_rate 2e-5 --epochs 10 --num_res_layers 3 || { echo "oopsie daisies"; exit 1; }
echo "Starting: 4-layer model on Potsdam"
python train.py --dataset Potsdam --output model_017 --batch_size 4 --learning_rate 2e-5 --epochs 10 --num_res_layers 4 || { echo "oopsie daisies"; exit 1; }

# # CocoStuff-15
# python train.py --dataset CocoStuff15 --output model_006 --batch_size 25 --learning_rate 2e-5 --epochs 10

# # CocoStuff-15
# python train.py --dataset CocoStuff15 --output model_006 --batch_size 25 --learning_rate 2e-5 --epochs 10