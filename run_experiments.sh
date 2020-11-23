#!/bin/bash

# Potsdam3
python train.py --dataset Potsdam3 --output model_004 --batch_size 20 --learning_rate 2e-5 --epochs 10

# Potsdam
python train.py --dataset Potsdam --output model_005 --batch_size 20 --learning_rate 2e-5 --epochs 10

# CocoStuff-15
python train.py --dataset CocoStuff15 --output model_006 --batch_size 25 --learning_rate 2e-5 --epochs 10

# CocoStuff-15
python train.py --dataset CocoStuff15 --output model_006 --batch_size 25 --learning_rate 2e-5 --epochs 10