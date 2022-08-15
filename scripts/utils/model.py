from torch.utils.data import DataLoader
import numpy as np
import mmcv
import torch
import os.path as osp
import os

from .io import generate_filtered_annfile

from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb
from mmcls.models.builder import build_classifier

from mmseg.datasets.builder import DATASETS, build_dataset, build_dataloader

from mmcv.runner import load_checkpoint

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

    filteredAnnfilePath = generate_filtered_annfile(annfile, imgNames)

    cfg = mmcv.Config.fromfile(camConfig)

    cfg.data.test['ann_file'] = filteredAnnfilePath
    cfg.data.test['data_prefix'] = imgRoot
    print(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False
    )
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, camCheckpoint, map_location='cpu')
    # Delete custom annfile after finish.
    #os.remove(filteredAnnfilePath)