from torch.utils.data import DataLoader
import numpy as np
import mmcv
import torch
import os.path as osp
import os

from .io import generate_filtered_annfile, generate_ann_file

from mmcls.models.builder import build_classifier
from mmcls.apis.test import single_gpu_test

from mmseg.datasets.builder import DATASETS, build_dataset, build_dataloader

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

"""Here all model Classes used in classification must be registered into the DATASET Registry of mmseg"""
# @DATASETS.register_module('CompCars')
# def CompCarsWrapper(**kwargs):
#     return CompCars(**kwargs)

# @DATASETS.register_module('CompCarsWeb')
# def CompCarsWrapper(**kwargs):
#     return CompCarsWeb(**kwargs)

if not('CompCars' in DATASETS and 'CompCarsWeb' in DATASETS and 'StanfordCarsOriginal' in DATASETS):
    from .registrerDatasets import *


def get_wrongly_classified(imgRoot, camConfig, camCheckpoint, annfile, imgNames, saveDir, use_gpu=True, **kwargs):
    """
    :param imgRoot: Path to the root directory containing all samples.
    :param camConfig: Path to Config of CAM Model
    :param camCheckpoint: Path to CHeckpoint of CAM Model
    :param annfile: Path to annotation file containing all samples and corresponding ground truth. 
    :param imgNames: List of all image Names that should be considered. An updated annfile will be generated
                     based on those image Names and deleted afterwards.
    :param saveDir: Path where annotation files be saved to
    :param use_gpu: (default True) compute results / eval model on GPU. If False results will be computed over CPU

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
    if use_gpu:
        print('Evaluating Model on GPU')
        model = MMDataParallel(model, device_ids=[0])
    else:
        print("Evaluating Model on CPU")
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

def get_shared_classes(cfg1, checkpoint1, cfg2, checkpoint2):
    """
    Determines what classes are shared between the models that are specified by
    cfg1, checkpoint1 and cfg2, checkpoint2
    The results are returned as a list of tuples (name, index1, index2)
    where index1 gives the index of the class in cfg1 and index2 in cfg2

    :param cfg1: Path to config File for first model
    :type cfg1: str | Path
    :param checkpoint1: Path to checkpoint File for first model
    :type checkpoint1: str | Path
    :param cfg2: Path to config File for second model
    :type cfg2: str | Path
    :param checkpoint2: Path to checkpoint File for second model
    :type checkpoint2: str | Path
    """
    print('Determining shared classes between given model configurations.')
    cfgFirst = mmcv.Config.fromfile(cfg1)
    cfgSecond = mmcv.Config.fromfile(cfg2)
    modelFirst = build_classifier(cfgFirst.model)
    modelSecond = build_classifier(cfgSecond.model)
    checkpointFirst = load_checkpoint(modelFirst, checkpoint1)
    checkpointSecond = load_checkpoint(modelSecond, checkpoint2)

    shared = []
    for index1, className in enumerate(checkpointFirst['meta']['CLASSES']):
        if className in checkpointSecond['meta']['CLASSES']:
            shared.append((className, index1, checkpointSecond['meta']['CLASSES'].index(className)))

    print(f'{len(shared)} potential shared classes found.')
    return shared