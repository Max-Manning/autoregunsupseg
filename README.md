Implementation of "Autoregressive Unsupervised Image Segmentation" by Y. Ouali et. al., ECCV 2020.

https://arxiv.org/abs/2007.08247

#### Requirements
```
numpy
scipy
pytorch 
tqdm
cv2 (only for the Potsdam data preprocessing script)
pillow
```

#### Datasets

The Coco-Stuff dataset can be obtained from [here]( https://github.com/nightrome/cocostuff ). A data request form for downloading the Potsdam dataset can be found [here]( https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/ ). Run `datasets/potsdam_prepare.py` to preprocess the Potsdam images.
 
#### Usage

Once you have downloaded the datasets, update the paths where you have stored them in `train.py` (lines 73, 84, 95, 101).

Next, see `run_experiments.sh` for a list of commands used to train the model.