from torch.utils.data import DataLoader
import numpy as np
import mmcv
import torch
import os.path as osp

from .io import generate_filtered_annfile

from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb

from mmseg.datasets.builder import DATASETS

"""Here all model Classes used in classification must be registered into the DATASET Registry of mmseg"""
# @DATASETS.register_module('CompCars')
# def CompCarsWrapper(**kwargs):
#     return CompCars(**kwargs)

# @DATASETS.register_module('CompCarsWeb')
# def CompCarsWrapper(**kwargs):
#     return CompCarsWeb(**kwargs)



def get_wrongly_classified(imgRoot, camConfig, camCheckpoint, annfile, imgNames, **kwargs):
    """
    :param imgRoot: Path to the root directory containing all samples.
    :param camConfig: Path to Config of CAM Model
    :param camCheckpoint: Path to CHeckpoint of CAM Model
    :param annfile: Path to annotation file containing all samples and corresponding ground truth. 
    :param imgNames: List of all image Names that should be considered. An updated annfile will be generated
                     based on those image Names and deleted afterwards.
    """
    #print(DATASETS)
    assert osp.isdir(imgRoot), f'No such directory {imgRoot}.'
    assert osp.isfile(osp.abspath(annfile)), f'No such file {osp.abspath(annfile)} for annfile.'

    print(imgNames)
    filteredAnnfilePath = generate_filtered_annfile(annfile, imgNames)

    cfg = mmcv.Config.fromfile(camConfig)

    cfg.data.test['ann_file'] = annfile
    cfg.data.test['data_prefix'] = imgRoot
    print(cfg.data.test)