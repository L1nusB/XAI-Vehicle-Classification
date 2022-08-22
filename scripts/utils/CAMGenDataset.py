import warnings
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import mmcv
import cv2

from .io import get_samples
from .preprocessing import load_classes
from .imageProcessing import convert_numpy_to_PIL
from .io import savePIL

class ImageDataset(Dataset):

    def __init__(self, imgRoot, annfile=None, imgNames=None, dataClasses=[], pipeline=None, get_gt=False):
        super(ImageDataset, self).__init__()
        self.imgRoot = imgRoot
        if imgNames:
            # Temporary solution since i don't really use this for more than single images.
            if get_gt:
                warnings.warn("Currently gt_labels are not supported when giving imgNames parameter!")
            self.gt_targets = [0] * len(imgNames)
            if len(dataClasses)>0:
                self.data = [name for name in imgNames if any(name.startswith(s) for s in dataClasses) and os.path.isfile(os.path.join(imgRoot, name))]
                self.imgPaths = np.array([os.path.join(imgRoot, name) for name in self.data])
            else:
                self.data = [name for name in imgNames if os.path.isfile(os.path.join(imgRoot, name))]
                self.imgPaths = np.array([os.path.join(imgRoot, name) for name in self.data])
        else:
            samples = [sample for sample in get_samples(annfile, imgRoot, None, dataClasses, splitSamples=not(get_gt))]
            if get_gt:
                self.data = [sample.split(" ")[0] for sample in samples]
                self.gt_targets = [int(sample.split(" ")[-1].split("_")[0]) for sample in samples]
            else:
                self.data = [sample.split(" ")[0] for sample in samples]
                self.gt_targets = [0] * len(samples)
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

        item = {'img':imgArray, 'name':os.path.basename(self.imgPaths[idx]), 'gt_target':self.gt_targets[idx]}

        return item

class BlurredImageDataset(ImageDataset):

    def __init__(self, imgRoot, blurredSegments, segData, annfile=None, imgNames=None, dataClasses=[], pipeline=None, get_gt=False,
                blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None,saveDir='blurredImgs/', saveImgs=False):
        super().__init__(imgRoot, annfile, imgNames, dataClasses, pipeline, get_gt)
        assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if classes not given.'
        classes = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)
        self.classes = classes
        if isinstance(blurredSegments, str):
            assert blurredSegments in classes, f'{blurredSegments} not in classes: {",".join(classes)}'
            self.blurredSegments = [classes.index(blurredSegments)]
        elif isinstance(blurredSegments, int):
            assert blurredSegments < len(classes), f'Can not blur segment no. {blurredSegments}. Only {len(classes)} present.'
            self.blurredSegments = [blurredSegments]
        elif isinstance(blurredSegments, list | np.ndarray):
            self.blurredSegments = []
            for segment in blurredSegments:
                if isinstance(segment, str):
                    assert segment in classes, f'{segment} not in classes: {",".join(classes)}'
                    self.blurredSegments.append(classes.index(segment))
                elif isinstance(segment, int):
                    assert segment < len(classes), f'Can not blur segment no. {segment}. Only {len(classes)} present.'
                    self.blurredSegments.append(segment)
                else:
                    raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
            # Remove potential duplicates
            self.blurredSegments = list(set(self.blurredSegments))
        else:
            raise ValueError(f'blurredSegments must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegments)}')
        self.segData = np.load(segData)
        self.saveImgs = saveImgs
        self.saveDir = saveDir
        self.blurKernel = blurKernel
        self.blurSigmaX = blurSigmaX

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ori_img = item['img']

        segmentation = self.segData[item['name']]

        assert segmentation.shape == ori_img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {ori_img.shape[:-1]}'

        mask = np.zeros_like(ori_img)
        for blur in self.blurredSegments:
            mask[segmentation==blur, :] = np.array([255,255,255])

        blur_img = cv2.GaussianBlur(ori_img, self.blurKernel, self.blurSigmaX)

        img = np.where(mask==np.array([255,255,255]), blur_img, ori_img)

        item['img'] = img

        if self.saveImgs:
            pil = convert_numpy_to_PIL(item['img'])
            savePIL(pil, fileName=item['name'], dir=self.saveDir, logSave=False)

        return item