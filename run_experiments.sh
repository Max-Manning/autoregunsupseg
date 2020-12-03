#!/bin/bash

#Potsdam3
# python train.py --dataset Potsdam3 --output model_001 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_002 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_003 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_004 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_005 --batch_size 10 --learning_rate 5e-5 --epochs 3 --num_res_layers 3 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_006 --batch_size 10 --learning_rate 5e-5 --epochs 3 --num_res_layers 3 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_007 --batch_size 5  --learning_rate 5e-5 --epochs 3 --num_res_layers 4 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

# #Potsdam
# python train.py --dataset Potsdam --output model_008 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_009 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_010 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_011 --batch_size 20 --learning_rate 5e-5 --epochs 5 --num_res_layers 2 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam --output model_012 --batch_size 10 --learning_rate 5e-5 --epochs 3 --num_res_layers 3 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_013 --batch_size 10 --learning_rate 5e-5 --epochs 3 --num_res_layers 3 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_014 --batch_size 5  --learning_rate 5e-5 --epochs 3 --num_res_layers 4 --output_stride 2 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

# CocoStuff 3
# python train.py --dataset CocoStuff3 --output model_015 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_016 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_017 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_018 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 1 || { echo "oopsie daisies"; exit 1; }

##### FIXED THE LOSS FUNCTION HERE -- ALL PREV RESULTS QUESTIONABLE ####


# python train.py --dataset CocoStuff3 --output model_019 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_020 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_021 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_022 --batch_size 60 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_023 --batch_size 30 --learning_rate 5e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3 --output model_024 --batch_size 30 --learning_rate 5e-5 --epochs 12 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }


# # # # CocoStuff15
# python train.py --dataset CocoStuff15 --output model_025 --batch_size 60 --learning_rate 1e-4 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15 --output model_026 --batch_size 60 --learning_rate 1e-4 --epochs 8 --num_res_layers 2 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15 --output model_027 --batch_size 60 --learning_rate 1e-4 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15 --output model_028 --batch_size 60 --learning_rate 1e-4 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15 --output model_029 --batch_size 30 --learning_rate 1e-4 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15 --output model_030 --batch_size 30 --learning_rate 1e-4 --epochs 12 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }


# # # back to Potsdam
# # python train.py --dataset Potsdam --output model_031 --batch_size 20 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_032 --batch_size 10 --learning_rate 5e-6 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_033 --batch_size 15 --learning_rate 5e-6 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# # python train.py --dataset Potsdam --output model_034 --batch_size 20 --learning_rate 5e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_035 --batch_size 10 --learning_rate 5e-6 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam --output model_036 --batch_size 10 --learning_rate 5e-6 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# # ## Potsdam-3
# # python train.py --dataset Potsdam3 --output model_037 --batch_size 20 --learning_rate 5e-5 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_038 --batch_size 20 --learning_rate 5e-6 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_039 --batch_size 20 --learning_rate 5e-6 --epochs 8 --num_res_layers 3 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# # python train.py --dataset Potsdam3 --output model_040 --batch_size 20 --learning_rate 5e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_041 --batch_size 10 --learning_rate 5e-6 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam3 --output model_042 --batch_size 10 --learning_rate 5e-6 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_101 --batch_size 18  --learning_rate 5e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_102 --batch_size 18  --learning_rate 4e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15  --output model_103 --batch_size 30  --learning_rate 4e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3  --output model_104 --batch_size 30  --learning_rate 6e-6 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_105 --batch_size 10  --learning_rate 5e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_106 --batch_size 10  --learning_rate 4e-5 --epochs 8 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_107 --batch_size 18  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_108 --batch_size 18  --learning_rate 1e-5 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# # python train.py --dataset CocoStuff15  --output model_109 --batch_size 30  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3  --output model_110 --batch_size 30  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_111 --batch_size 10  --learning_rate 1e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_112 --batch_size 10  --learning_rate 2e-5 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# # # python train.py --dataset Potsdam3 --output model_113 --batch_size 18  --learning_rate 1e-7 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# # python train.py --dataset Potsdam  --output model_114 --batch_size 18  --learning_rate 4e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# # # python train.py --dataset CocoStuff15  --output model_115 --batch_size 30  --learning_rate 4e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3  --output model_116 --batch_size 30  --learning_rate 6e-6 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_117 --batch_size 10  --learning_rate 1e-7 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_118 --batch_size 10  --learning_rate 4e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_120 --batch_size 18  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_121 --batch_size 18  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3  --output model_122 --batch_size 30  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_123 --batch_size 10  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_124 --batch_size 10  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam3 --output model_125 --batch_size 4  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 2 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_126 --batch_size 4  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 2 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset Potsdam  --output model_127 --batch_size 18  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff3  --output model_128 --batch_size 30  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15  --output model_129 --batch_size 30  --learning_rate 1e-6 --epochs 4 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_130 --batch_size 10  --learning_rate 1e-6 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset CocoStuff3  --output model_131 --batch_size 30  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset CocoStuff15  --output model_132 --batch_size 30  --learning_rate 1e-5 --epochs 4 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
# python train.py --dataset Potsdam  --output model_133 --batch_size 10  --learning_rate 1e-5 --epochs 10 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }

# python train.py --dataset CocoStuff3  --output model_134 --batch_size 30  --learning_rate 4e-5 --epochs 12 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset CocoStuff15  --output model_135 --batch_size 30  --learning_rate 6e-6 --epochs 12 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam  --output model_136 --batch_size 20  --learning_rate 4e-5 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 || { echo "oopsie daisies"; exit 1; }
python train.py --dataset Potsdam  --output model_137 --batch_size 10  --learning_rate 4e-5 --epochs 20 --num_res_layers 4 --output_stride 4 --spatial_invariance 10 --attention True || { echo "oopsie daisies"; exit 1; }