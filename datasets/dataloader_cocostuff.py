import numpy as np
import os
from PIL import Image
import pickle

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

with open('/home/miamanning/unsupseg/datasets/fine_to_coarse_dict.pickle', 'rb') as f:
    fine_to_coarse_dict = pickle.load(f)

class CocoStuff(Dataset):
    ''' COCO STUFF DATASET CLASS'''
    
    def __init__(self, data_list_path, data_path, label_path, version='CocoStuff3', transforms=None, return_label=True):
        
        self.data_list_path = data_list_path       # path to file list
        self.data_path = data_path                 # path to data directory
        self.label_path = label_path               # path to label directory
        self.transforms = transforms               # list of transforms for the image
        self.version = version                     # version to use (CocoStuff3 or CocoStuff15)
        self.return_label = return_label
        
        self.files = list(np.loadtxt(self.data_list_path, dtype=np.str))
        
    def __getitem__(self, idx):
        
        ## get an image index idx ##
        # fileno = self.files[idx]
        image_path = os.path.join(self.data_path,  idx + ".jpg")
        label_path = os.path.join(self.label_path, idx + ".png")
        
        # load the image
        # have to make sure its in RGB because apparently there are some greyscale images in the dataset??? That's unfortunate
        img = Image.open(image_path).convert("RGB") 
        img = np.array(img.resize((128, 128), resample = Image.BILINEAR))
        
        assert img.shape == (128, 128, 3), f"WRONG IMAGE DIMENSIONS!!!111 {img.shape}"
        
        # apply transforms
        if self.transforms:
            img = self.transforms(img)
            
        if self.return_label:
            
            # format and return the label
            label_fine = Image.open(label_path)
            label_fine = np.array(label_fine.resize((128, 128), resample=Image.NEAREST))

            # convert from fine to coarse labels
            label_coarse = np.zeros(label_fine.shape, dtype=np.uint8)
            for c in range(1,182):
                label_coarse[label_fine == c] = fine_to_coarse_dict['fine_index_to_coarse_index'][c]

            if self.version == 'CocoStuff3':
                # 4 class labels: 3-sky, 2-plants, 1-ground, 0-other.
                label = np.zeros(label_fine.shape, dtype = np.uint8)
                label[label_coarse == 21] = 1 # ground
                label[label_coarse == 22] = 2 # plants
                label[label_coarse == 23] = 3 # sky

            elif self.version == 'CocoStuff15':
                # the 15 coarse 'stuff' classes (1 - 15). Everything else is 0.
                label = label_coarse - 12
                label[label < 0]  = 0
                label[label > 15] = 0
            return img, label
        else:
            # no label needed for unsupervised training
            return img

    def __len__(self):
        # how many files are there???
        return len(self.files)
    
class CocoDataLoader(DataLoader):
    """
    COCO-STUFF DATA LOADER CLASS
    """
    def __init__(self, dataset, batch_size=10, num_workers=1):
        
        self.sampler = SubsetRandomSampler(dataset.files)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': None, # shuffle has to be none for using SubsetRandomSampler
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
        
def get_coco_dataloader(batch_size, version='CocoStuff3', split='train'):
    
    if version == 'CocoStuff3':
        if split == 'train':
            data_list_path = '/home/miamanning/unsupseg/datasets/CocoStuff3_file_list_train.txt'
        else:
            data_list_path = '/home/miamanning/unsupseg/datasets/CocoStuff3_file_list_val.txt'
    elif version == 'CocoStuff15':
        if split == 'train':
            data_list_path = '/home/miamanning/unsupseg/datasets/CocoStuff15_file_list_train.txt'
        else:
            data_list_path = '/home/miamanning/unsupseg/datasets/CocoStuff15_file_list_val.txt'
    else:
        raise ValueError("Unknown dataset version.")
    
    if split == 'train':
        data_path = '/mnt/D2/Data/CocoStuff164k/images/train2017/'
        label_path = '/mnt/D2/Data/CocoStuff164k/annotations/train2017/'
    else:
        data_path = '/mnt/D2/Data/CocoStuff164k/images/val2017/'
        label_path = '/mnt/D2/Data/CocoStuff164k/annotations/val2017/'
        
    cropsize = 128
        
    # transforms for the image
    img_trsfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    if split == 'train':
        cocodataset = CocoStuff(data_list_path, data_path, label_path, version=version,
                            transforms=img_trsfm, return_label=False)
    else:
        cocodataset = CocoStuff(data_list_path, data_path, label_path, version=version,
                            transforms=img_trsfm, return_label=True)
        
    return CocoDataLoader(cocodataset, batch_size)
    
    

    