import numpy as np
import os
from torch.utils.data import Dataset
import torch
import mmcv

from .utils.io import get_samples

class ImageDataset(Dataset):

    def __init__(self, imgRoot, annfile=None, imgNames=None, dataClasses=[], pipeline=None):
        super(ImageDataset, self).__init__()
        self.imgRoot = imgRoot
        if imgNames:
            if len(dataClasses)>0:
                self.data = [name for name in imgNames if any(name.startswith(s) for s in dataClasses) and os.path.isfile(os.path.join(imgRoot, name))]
                self.imgPaths = np.array([os.path.join(imgRoot, name) for name in self.data])
            else:
                self.data = [name for name in imgNames if os.path.isfile(os.path.join(imgRoot, name))]
                self.imgPaths = np.array([os.path.join(imgRoot, name) for name in self.data])
        else:
            self.data = [name for name in get_samples(annfile, imgRoot, None, dataClasses)]
            self.imgPaths = np.array([os.path.join(imgRoot, name) for name in self.data])
        self.pipeline = pipeline

    def __len__(self):
        return self.imgPaths.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgArray = mmcv.imread(self.imgPaths[idx])

        if self.pipeline:
            imgArray = self.pipeline(imgArray)

        item = {'img':imgArray, 'name':os.path.basename(self.imgPaths[idx])}

        return item