import numpy as np
import os
from torch.utils.data import Dataset
import torch
import mmcv

class ImageDataset(Dataset):

    def __init__(self, imgRoot, annfile=None, imgNames=None, classes=None, pipeline=None):
        super(ImageDataset, self).__init__()
        self.imgRoot = imgRoot
        if imgNames:
            if classes:
                self.imgNames = np.array([os.path.join(imgRoot, name) for name in imgNames if any(name.startswith(s) for s in classes)])
            else:
                self.imgNames = np.array([os.path.join(imgRoot, name) for name in imgNames])
        elif annfile:
            assert os.path.isfile(annfile), f'annfile {annfile} is no file.'
            with open(annfile) as f:
                if classes:
                    names = [os.path.join(imgRoot, x.strip().rsplit(' ', 1)[0]) for x in f.readlines() if any(x.startswith(s) for s in classes)]
                else:
                    names = [os.path.join(imgRoot, x.strip().rsplit(' ', 1)[0]) for x in f.readlines()]
            self.imgNames = np.array(names)
        else:
            self.imgNames = np.array([f for f in os.listdir(imgRoot) if os.path.isfile(os.path.join(imgRoot,f))])
        self.pipeline = pipeline

    def __len__(self):
        return self.imgNames.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgArray = mmcv.imread(self.imgNames[idx])

        if self.pipeline:
            imgArray = self.pipeline(imgArray)

        return imgArray