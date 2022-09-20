import warnings
import numpy as np
import os
from torch.utils.data import Dataset
import torch
import mmcv
import cv2

from .io import get_samples
from .preprocessing import load_classes

from mmcls.datasets.builder import PIPELINES

importSwitch = False

if 'BlurSegments' not in PIPELINES and importSwitch==False:
    print('Importing BlurSegments step into Pipeline')
    from .customPipelineSteps import BlurSegments
    importSwitch = True

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
                #self.data = [name for name in imgNames if any(name.startswith(s) for s in dataClasses) and os.path.isfile(os.path.join(imgRoot, name))]
                #self.imgNames = np.array([os.path.join(imgRoot, name) for name in self.data])
                self.imgNames = np.array([name for name in imgNames if any(name.startswith(s) for s in dataClasses) and os.path.isfile(os.path.join(imgRoot, name))])
            else:
                # self.data = [name for name in imgNames if os.path.isfile(os.path.join(imgRoot, name))]
                # self.imgNames = np.array([os.path.join(imgRoot, name) for name in self.data])
                self.imgNames = np.array([name for name in imgNames if os.path.isfile(os.path.join(imgRoot, name))])
        else:
            samples = [sample for sample in get_samples(annfile, imgRoot, None, dataClasses, splitSamples=not(get_gt))]
            if get_gt:
                # self.data = [sample.split(" ")[0] for sample in samples]
                self.gt_targets = [int(sample.split(" ")[-1].split("_")[0]) for sample in samples]
            else:
                # self.data = [sample.split(" ")[0] for sample in samples]
                self.gt_targets = [0] * len(samples)
            # self.imgNames = np.array([os.path.join(imgRoot, name) for name in self.data])
            self.imgNames = np.array([sample.split(" ")[0] for sample in get_samples(annfile, imgRoot, None, dataClasses, splitSamples=not(get_gt))])
        self.pipeline = pipeline

    def __len__(self):
        return self.imgNames.size

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # imgArray = mmcv.imread(self.imgNames[idx])

        # if self.pipeline:
        #     imgArray = self.pipeline(imgArray)

        # item = {'img':imgArray, 'name':os.path.basename(self.imgNames[idx]), 'gt_target':self.gt_targets[idx]}

        item = {'img':[], 'name':os.path.basename(self.imgNames[idx]), 'gt_target':self.gt_targets[idx]}

        return item

def add_blurring_pipeline_step(cfg, blurredSegments, segData, segConfig, segCheckpoint, classes=None,
                                blurKernel=(55,55), blurSigmaX=0, saveDir='blurredImgs/', saveImgs=False,
                                singleColor=None):
    """
    Adds the BlurSegment Step into the Pipeline for the given cfg (under cfg.data.test.pipeline)
    This step is inserted directly after the LoadFromFile step.

    :param cfg: Config into which the step is added
    :type cfg: _type_
    :param blurredSegments: Blurred Segments. Will be passed to constructor of pipeline which will adapt to the format given
    :type blurredSegments: str | int | List(str|int)
    :param segData: Path to file containing the segmentation data. The data must match the original size at it will be applied on the original image
    :type segData: str | Path
    :param blurKernel: Kernel used for gaussian blurring, defaults to (55,55)
    :type blurKernel: tuple(int,int), optional
    :param blurSigmaX: SigmaX used for gaussian blurring, defaults to 0
    :type blurSigmaX: int, optional
    :param segConfig: Path to Config file of segmentation model used for loading the segmentation Categories/classes, defaults to None
    :type segConfig: str | Path, optional
    :param segCheckpoint: Path to Checkpoint file of segmentation model used for loading the segmentation Categories/classes, defaults to None
    :type segCheckpoint: str | Path, optional
    :param saveDir: Directory where blurred images will be saved to if saveImgs=True, defaults to 'blurredImgs/'
    :type saveDir: str | Path, optional
    :param saveImgs: Whether or not to save the blurred images, defaults to False
    :type saveImgs: bool, optional
    :param singleColor: Maked blurred segments of single color. Must be bgr channel order.
    :type singleColor: list(int) | np.ndarray(int) | None

    :return Modified config
    """
    pipeline = cfg.data.test.pipeline
    indexLoad = pipeline.index({'type': 'LoadImageFromFile'}) + 1
    if classes is None:
        assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if classes not given.'
        classes = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)
    # If just boolean is given set it to all white.
    if singleColor == True:
        singleColor = [116.28, 103.53, 123.675]
    pipeline = pipeline[:indexLoad] + \
                [{'type':'BlurSegments', 'blurredSegments':blurredSegments, 'segData':segData,
                'segCategories':classes, 'blurKernel':blurKernel, 'singleColor':singleColor,
                'blurSigmaX':blurSigmaX, 'saveImgs':saveImgs, 'saveDir':saveDir, 'logInfos':False}] + \
                pipeline[indexLoad:]
    cfg.data.test.pipeline = pipeline
    return cfg