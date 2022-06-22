import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import torch
import mmcv
from mmcv import Config
from .visualization_seg_masks import generateUnaryMasks
from . import transformations, utils, generate_cams, generate_segmentation
import copy
import heapq


def generate_statistic_single(cam, segmentation, segmentCount):
    """Generate intersections stastic both absolute and percentual for a single instance.

    :param cam: CAM Activation. Must match shape of segmentation. (h,w)
    :type cam: np.ndarray
    :param segmentation: Segmentation Mask. Must match shape of cam. (h,w)
    :type segmentation: np.ndarray
    :param segmentCount: Amount of segmentation classes. If some classes are not present in segmentation.
    :type segmentCount: int
    """
    assert cam.shape == segmentation.shape, f"cam.shape {cam.shape} does not "\
        f"match segmentation.shape {segmentation.shape}"
    assert len(cam.shape)==2,f"Expected shape (h,w) "\
        f"but got cam.shape:{cam.shape} and segmentation.shape:{segmentation.shape}"
    width = segmentation.shape[1]
    height = segmentation.shape[0]
    binaryMasks = generateUnaryMasks(segmentation, width, height, segmentCount)

    segmentedCAM = [cam*mask for mask in binaryMasks]
    totalCAMActivation = cam.sum()
    segmentedCAMActivation = np.sum(segmentedCAM, axis=(1,2))
    percentualSegmentedCAMActivation = segmentedCAMActivation/totalCAMActivation

    return segmentedCAMActivation, percentualSegmentedCAMActivation

def accumulate_statistics(imgNames, cams, segmentations, segmentCount):
    """Accumulates all intersections for the given imgNames.
    Returns both absolut and percentual statistics.

    :param imgNames: List of imgNames. Keys into the cams and segmentations
    :type imgNames: list or tuple
    :param cams: CAMs stored in a dictionary keyed by the imgNames
    :type cams: dict
    :param segmentations: Segmentations stored in a dictionary keyed by the imgNames
    :type segmentations: dict
    :param segmentCount: Amount of segmentation classes. If some classes are not present in segmentation.
    :type segmentCount: int
    """
    totalCAMActivations = []
    segmentedCAMActivations = []
    percentualSegmentedCAMActivations = []

    for name in imgNames:
        totalCAMActivations.append(cams[name].sum())
        segmentedCAMActivation, percentualSegmentedCAMActivation = generate_statistic_single(
            cams[name], segmentations[name], segmentCount)
        segmentedCAMActivations.append(segmentedCAMActivation)
        percentualSegmentedCAMActivations.append(percentualSegmentedCAMActivation)
    
    return np.array(totalCAMActivations), np.array(segmentedCAMActivations), np.array(percentualSegmentedCAMActivations)

def generate_statistic(imgNames, cams, segmentations, classes, pipeline=None):
    """Accumulates all intersections for the given imgNames.
    Returns both absolut and percentual statistics.

    :param imgNames: List of imgNames. Keys into the cams and segmentations
    :type imgNames: list or tuple
    :param cams: CAMs stored in a dictionary keyed by the imgNames
    :type cams: dict
    :param segmentations: Segmentations stored in a dictionary keyed by the imgNames
    :type segmentations: dict
    :param classes: Segmentation classes. 
    :type classes: tuple/list
    :param pipeline: Pipeline that is applied to the segmentations. If pipeline=None nothing is applied
    """
    segs = copy.deepcopy(segmentations)
    if pipeline is not None:
        for name in imgNames:
            segs[name] = transformations.pipeline_transform(segs[name],pipeline)

    classArray = np.array(classes)
    numSamples = len(imgNames)
    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations = accumulate_statistics(imgNames, cams, segs, classArray.size)
    totalActivation = totalCAMActivations.sum()

    summarizedSegmentedCAMActivations = segmentedCAMActivations.mean(axis=0)
    summarizedPercSegmentedCAMActivations = percentualSegmentedCAMActivations.mean(axis=0)

    dominantSegmentsRaw = heapq.nlargest(3,summarizedSegmentedCAMActivations)
    dominantMaskRaw = summarizedSegmentedCAMActivations >= np.min(dominantSegmentsRaw)

    dominantSegmentsPerc = heapq.nlargest(3,summarizedPercSegmentedCAMActivations)
    dominantMaskPerc = summarizedPercSegmentedCAMActivations >= np.min(dominantSegmentsPerc)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[0,1])

    # For the absolut we take the total Activations, sum them and only afterwards get a percentage relative to the total CAM Activations
    # across all sampels.
    bars = ax0.bar(classArray, summarizedSegmentedCAMActivations)
    ticks_loc = ax0.get_xticks()
    ax0.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax0.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMaskRaw):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax0.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height*numSamples/totalActivation:.1%}', ha='center', va='bottom' )
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.set_title('Average absolut CAM Activations')

    # For the relative we immediately take the percentage activations of each sample and average them afterwards.
    bars = ax1.bar(classArray, summarizedPercSegmentedCAMActivations)
    ticks_loc = ax1.get_xticks()
    ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax1.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMaskPerc):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height:.1%}', ha='center', va='bottom' )
    ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)
    ax1.set_title('Average relative CAM Activations')

    return ax0, ax1

def generate_statistics_infer(imgRoot, classes, camConfig=None, camCheckpoint=None, segConfig=None, segCheckpoint=None, annfile=None, genClasses=None, pipeline=None, **kwargs):
    """Generate statistics for average absolute and average relative Intersection of CAM with 
    segmentation Mask. See @generate_statistics. Infers the CAMs and Segmentation from parameters
    and calls generate_statistics with generated objects.

    :param imgRoot: Root the directory of the images
    :type imgRoot: str
    :param camConfig: Path to the Config of the CAM Model
    :type camConfig: str
    :param camCheckpoint: Path to the Checkpoint of the CAM Model
    :type camCheckpoint: str
    :param segConfig: Path to the config of the Segmentation Model
    :type segConfig: str
    :param segCheckpoint: Path to the checkpoint of the Segmentation Model
    :type segCheckpoint: str
    :param classes: Segmentation classes.
    :type classes: tuple/list
    :param annfile: Path to annfile defining which images will be used from imgRoot. If None all images in 
    imgRoot will be used, defaults to None
    :type annfile: str, optional
    :param genClasses: Class for which the statistic will be generated. Will match all images in imgRoot
    containing the specified Class. Can be combined with annfile., defaults to None
    :type genClasses: str, optional
    :param pipeline: Pipeline that will be applied to segmentations in generate_statistics.

    For kwargs:
    getPipelineFromConfig: (bool) Load the pipeline from CAMConfig. Requires the pipeline to be located under
    data.test.pipeline
    scalePipelineToInt: (bool) Defines whether to scale the pipeline to Int see @transformations.get_pipeline_from_config_pipeline
    imgNames: (list like) Preloaded list of imageNames. If specified will be passed but MUST match what is generated from other parameters.
    cams: (dict(str,np.ndarray)) Pregenerated dict of CAMs. If specified will be passed but MUST contain what is generated from other parameters.
    segmentations: (dict(str, np.ndarray)) Pregenerated dict of Segmentations. If specified will be passed but MUST contain what is generated from other parameters.
    """
    assert os.path.isdir(imgRoot), f'imgRoot does not lead to a directory {imgRoot}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        imgNames = utils.getImageList(imgRoot, annfile, classes=genClasses, addRoot=False)

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    if 'cams' in kwargs:
        assert set(imgNames).issubset(set(kwargs['cams'].keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
        cams = kwargs['cams']
    else:
        assert os.path.isfile(camConfig), f'camConfig is no file {camConfig}'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint is no file {camCheckpoint}'
        cams = generate_cams.main([imgRoot, camConfig, camCheckpoint, '--ann-file', annfile, '--classes', genClasses])

    if 'segmentations' in kwargs:
        assert set(imgNames).issubset(set(kwargs['segmentations'].keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'
        segmentations = kwargs['segmentations']
    else:
        assert os.path.isfile(segConfig), f'segConfig is no file {segConfig}'
        assert os.path.isfile(segCheckpoint), f'segCheckpoint is no file {segCheckpoint}'
        if genClasses:
            segmentations,_ = generate_segmentation.main([imgRoot, segConfig, segCheckpoint,'--types', 'masks', '--ann-file', annfile, '--classes', genClasses])
        else:
            segmentations,_ = generate_segmentation.main([imgRoot, segConfig, segCheckpoint,'--types', 'masks', '--ann-file', annfile])

    if 'getPipelineFromConfig' in kwargs:
        cfg = Config.fromfile(camConfig)
        if 'scalePipelineToInt' in kwargs:
            pipeline = transformations.get_pipeline_from_config_pipeline(cfg.data.test.pipeline, scaleToInt=True)
        else:
            pipeline = transformations.get_pipeline_from_config_pipeline(cfg.data.test.pipeline)

    ax0,ax1 = generate_statistic(imgNames=imgNames, cams=cams, segmentations=segmentations, classes=classes, pipeline=pipeline)

    if genClasses is not None:
        ax0.set_xlabel(genClasses, fontsize='x-large')
        ax1.set_xlabel(genClasses, fontsize='x-large')