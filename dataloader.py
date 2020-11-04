import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import scipy.io as sio
import os

class Potsdam(Dataset):
    
    ''' POTSDAM DATASET CLASS'''
    
    def __init__(self, data_path, split='labelled_train', transforms=None, is_test=False):
        self.data_path = data_path    # path to data directory
        self.split = split            # train/ test split. "labelled_train","unlabelled_train", "labelled_test"
        self.transforms = transforms  # list of transforms
        self.is_test = is_test        # train or test
        
        ## GET FILE NUMBERS FOR THE CURRENT SPLIT ##
        file_list = os.path.join(self.data_path, self.split + ".txt")
        file_list = tuple(open(file_list, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list  # list of image ids
        
    def __getitem__(self, idx):
        ## get an image index idx ##
        image_path = os.path.join(self.data_path, "imgs/" + str(idx) + ".mat")
        label_path = os.path.join(self.data_path, "gt/" + str(idx) + ".mat")
        
        # load the image
        img = sio.loadmat(image_path)["img"]
        
        # convolutional layers expect channels first format
        # i.e. tensors will have size (batch_size, num_channels, height, width)
        img = np.transpose(img, (2,0,1))
        
        # apply transforms if there are any
        if self.transforms:
            img = self.transforms(**{"image": np.array(img)})["image"]
        
        # Check if there's a label for this image. If so, load it.
        if os.path.exists(label_path):
            label = sio.loadmat(label_path)["gt"]
            return torch.tensor(img).float(), torch.tensor(label)
        else:
            # return torch.tensor(img), None
            return torch.tensor(img).float()
    
    def __len__(self):
        # how many files are there???
        return len(self.files)
    
class PotsdamDataLoader(DataLoader):
    """
    POTSDAM DATA LOADER CLASS
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

    