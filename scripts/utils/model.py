from torch.utils.data import DataLoader
import numpy as np
import mmcv
import torch
import os.path as osp
import os

from .io import generate_filtered_annfile, generate_ann_file

from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb
from mmcls.models.builder import build_classifier
from mmcls.apis.test import single_gpu_test

from mmseg.datasets.builder import DATASETS, build_dataset, build_dataloader

from mmcv.runner import load_checkpoint

"""Here all model Classes used in classification must be registered into the DATASET Registry of mmseg"""
@DATASETS.register_module('CompCars')
def CompCarsWrapper(**kwargs):
    return CompCars(**kwargs)

@DATASETS.register_module('CompCarsWeb')
def CompCarsWrapper(**kwargs):
    return CompCarsWeb(**kwargs)



def get_wrongly_classified(imgRoot, camConfig, camCheckpoint, annfile, imgNames, saveDir, **kwargs):
    """
    :param imgRoot: Path to the root directory containing all samples.
    :param camConfig: Path to Config of CAM Model
    :param camCheckpoint: Path to CHeckpoint of CAM Model
    :param annfile: Path to annotation file containing all samples and corresponding ground truth. 
    :param imgNames: List of all image Names that should be considered. An updated annfile will be generated
                     based on those image Names and deleted afterwards.
    :param saveDir: Path where annotation files be saved to

    :return (pathCorrect, pathIncorrect) Path to annotation files specifying correct and incorretly classified samples. 
    """
    #print(DATASETS)
    assert osp.isdir(imgRoot), f'No such directory {imgRoot}.'
    assert osp.isfile(osp.abspath(annfile)), f'No such file {osp.abspath(annfile)} for annfile.'

    filteredAnnfilePath = generate_filtered_annfile(annfile, imgNames, saveDir=saveDir)

    cfg = mmcv.Config.fromfile(camConfig)

    cfg.data.test['ann_file'] = filteredAnnfilePath
    cfg.data.test['data_prefix'] = imgRoot

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False
    )
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, camCheckpoint, map_location='cpu')
    print("Determining wrongly classified samples.")
    results = single_gpu_test(model, data_loader)
    print("") # Just a new line after the progress bar
    results = np.vstack(results)

    pred_labels = np.argmax(results, axis=1)
    gt_labels = dataset.get_gt_labels()
    correct = pred_labels == gt_labels
    
    correctClassifiedList = [match['img_info']['filename'] + " " + str(match['gt_label'].item()) + "_0" 
                            for match in np.array(dataset.data_infos)[correct]]
    incorrectClassifiedList = [match['img_info']['filename'] + " " + str(match['gt_label'].item()) + "_0" 
                            for match in np.array(dataset.data_infos)[np.logical_not(correct)]]

    print(f'Number of correctly classified samples: {len(correctClassifiedList)}')
    print(f'Number of incorrectly classified samples: {len(incorrectClassifiedList)}')

    # Delete filtered annfile after finish since it is not needed anymore.
    print(f'Removing filtered annotation file {filteredAnnfilePath}')
    os.remove(filteredAnnfilePath)
    
    # Generate the annotation files for further work.
    annfileCorrectPath = generate_ann_file(correctClassifiedList, work_dir=saveDir, fileprefix="annfile_correct")
    annfileIncorrectPath = generate_ann_file(incorrectClassifiedList, work_dir=saveDir, fileprefix="annfile_incorrect")
    return annfileCorrectPath, annfileIncorrectPath