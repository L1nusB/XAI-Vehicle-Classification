import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import torch
import mmcv
from mmcv import Config
from .visualization_seg_masks import generateUnaryMasks
from . import transformations
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
    :param classes: Amount of segmentation classes. If some classes are not present in segmentation.
    :type classes: int
    :param pipeline: Pipeline that is applied to the segmentations. If pipeline=None nothing is applied
    """
    segs = copy.deepcopy(segmentations)
    if pipeline is not None:
        for name in imgNames:
            segs[name] = transformations.pipeline_transform(segs[name],pipeline)

    classArray = np.array(classes)
    numClasses = classArray.size
    numSamples = len(imgNames)
    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations = accumulate_statistics(imgNames, cams, segs, numClasses)
    totalActivation = totalCAMActivations.sum()

    summarizedSegmentedCAMActivations = segmentedCAMActivations.sum(axis=0)/numSamples
    summarizedPercSegmentedCAMActivations = percentualSegmentedCAMActivations.sum(axis=0)/numSamples

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
    ax1.set_title('Average relative CAM Activations')

    return totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations