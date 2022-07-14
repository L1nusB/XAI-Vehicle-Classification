import numpy as np
import heapq

from ..visualization_seg_masks import generateUnaryMasks


def accumulate_single(cam, segmentation, classes, percentualArea=False):
    """Generate intersections stastic both absolute and percentual for a single instance.

    :param cam: CAM Activation. Must match shape of segmentation. (h,w)
    :type cam: np.ndarray
    :param segmentation: Segmentation Mask. Must match shape of cam. (h,w)
    :type segmentation: np.ndarray
    :param classes: Classes of the segmentation used for calculating proportion of category to entire area.
    :type classes: List like object
    :param percentualArea: (default false) Flag whether to calcuate the percentualArea of each category with respect to the segmentation
    :type percentualArea: boolean

    :return: Tuple containing (segmentedCAMActivation, percentualSegmentedCAMActivation, segmentPercentualArea)
    """
    assert cam.shape == segmentation.shape, f"cam.shape {cam.shape} does not "\
        f"match segmentation.shape {segmentation.shape}"
    assert len(cam.shape)==2,f"Expected shape (h,w) "\
        f"but got cam.shape:{cam.shape} and segmentation.shape:{segmentation.shape}"
    width = segmentation.shape[1]
    height = segmentation.shape[0]
    segmentCount = len(classes)
    binaryMasks = generateUnaryMasks(segmentation, width, height, segmentCount)

    segmentedCAM = [cam*mask for mask in binaryMasks]
    totalCAMActivation = cam.sum()
    segmentedCAMActivation = np.sum(segmentedCAM, axis=(1,2))
    percentualSegmentedCAMActivation = segmentedCAMActivation/totalCAMActivation

    segmentPercentualArea = None
    if percentualArea:
        segmentPercentualArea = [segmentation[segmentation==c].size for c in np.arange(len(classes))] / np.prod(segmentation.shape)

    return segmentedCAMActivation, percentualSegmentedCAMActivation, segmentPercentualArea

def accumulate_statistics(imgNames, cams, segmentations, classes, percentualArea=False):
    """Accumulates all intersections for the given imgNames.
    Returns both absolut and percentual statistics.

    :param imgNames: List of imgNames. Keys into the cams and segmentations
    :type imgNames: list or tuple
    :param cams: CAMs stored in a dictionary keyed by the imgNames
    :type cams: dict
    :param segmentations: Segmentations stored in a dictionary keyed by the imgNames
    :type segmentations: dict
    :param classes: Classes of the segmentation used for calculating proportion of category to entire area.
    :type classes: List like object
    :param percentualArea: (default false) Flag whether to calcuate the percentualArea of each category with respect to the segmentation
    :type percentualArea: boolean

    :return: Tuple containing (totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations, (percentualSegmentAreas))
    """
    totalCAMActivations = []
    segmentedCAMActivations = []
    percentualSegmentedCAMActivations = []
    percentualSegmentAreas = []

    for name in imgNames:
        totalCAMActivations.append(cams[name].sum())
        segmentedCAMActivation, percentualSegmentedCAMActivation, percentualSegmentArea = accumulate_single(
            cams[name], segmentations[name], classes)
        segmentedCAMActivations.append(segmentedCAMActivation)
        percentualSegmentedCAMActivations.append(percentualSegmentedCAMActivation)
        if percentualArea:
            percentualSegmentAreas.append(percentualSegmentArea)

    results = [np.array(totalCAMActivations), np.array(segmentedCAMActivations), np.array(percentualSegmentedCAMActivations)]
    if percentualArea:
        results.append(np.array(percentualSegmentAreas))
    
    return results

def generate_stats_abs(segmendtedActivations):
    """
    Creates array containing values for the absolute statistics for plotting of multiple samples.
    :param segmendtedActivations: CAM Activations per Segment
    :type segmendtedActivations: np.ndarray(np.ndarray(float))

    :return  summarizedSegmentedCAMActivations, dominantMask
    """
    summarizedSegmentedCAMActivations = segmendtedActivations.mean(axis=1).mean(axis=0)

    dominantSegmentsRaw = heapq.nlargest(3,summarizedSegmentedCAMActivations)
    dominantMask = summarizedSegmentedCAMActivations >= np.min(dominantSegmentsRaw)

    return summarizedSegmentedCAMActivations, dominantMask

def generate_stats_rel(percentualActivations):
    """
    Creates array containing values for the relative statistics for plotting of multiple samples.
    :param percentualActivations: Percentual CAM Activations per Segment
    :type percentualActivations: np.ndarray(np.ndarray(float))

    :return summarizedPercSegmentedCAMActivations, dominantMask
    """
    summarizedPercSegmentedCAMActivations = percentualActivations.mean(axis=1).mean(axis=0)

    dominantSegmentsPercRaw = heapq.nlargest(3,summarizedPercSegmentedCAMActivations)
    dominantMaskPercentual = summarizedPercSegmentedCAMActivations >= np.min(dominantSegmentsPercRaw)

    return summarizedPercSegmentedCAMActivations, dominantMaskPercentual

def generate_stats_rel_area(percentualAreas):
    """
    Creates array containing values for the relative statistics for plotting the relative areas of each segment.
    :param percentualAreas: Relative Area of each segment w.r.t the entire mask/image
    :type percentualAreas: np.ndarray(np.ndarray(float))

    :return summarizedPercSegmentedAreas
    """
    
    summarizedPercSegmentedAreas = percentualAreas.mean(axis=0)

    return summarizedPercSegmentedAreas

def generate_stats(classes, segmendtedActivations=None, percentualActivations=None, totalCAM=None, percentualAreas=None):
    """
    Creates arrays containing values for plotting of multiple samples.
    :param classes: Classes corresponding to the categories of the segmentations.
    :type classes: list/tuple like object.
    :param segmendtedActivations: CAM Activations per Segment
    :type segmendtedActivations: np.ndarray(np.ndarray(float))
    :param percentualActivations: Percentual CAM Activations per Segment
    :type percentualActivations: np.ndarray(np.ndarray(float))
    :param totalCAM: Total CAM Activations (per batch)
    :type totalCAM: np.ndarray(np.ndarray(float))
    :param percentualAreas: Relative Area of each segment w.r.t the entire mask/image
    :type percentualAreas: np.ndarray(np.ndarray(float))

    :return: depending on what was specified: [classArray, totalActivation, summarizedSegmentedCAMActivations, summarizedPercSegmentedCAMActivations, dominantMask, dominantMaskPercentual,summarizedPercAreas] 
        
    """
    results = [np.array(classes)]

    if totalCAM is not None:
        totalActivation = np.sum(totalCAM)
        results.append(totalActivation)
    if segmendtedActivations is not None:
        summarizedSegmentedCAMActivations, dominantMask = generate_stats_abs(segmendtedActivations)
        results.append(summarizedSegmentedCAMActivations)
        results.append(dominantMask)
    if percentualActivations is not None:
        summarizedPercSegmentedCAMActivations, dominantMaskPercentual = generate_stats_rel(percentualActivations)
        results.append(summarizedPercSegmentedCAMActivations)
        results.append(dominantMaskPercentual)
    if percentualAreas is not None:
        summarizedPercAreas = generate_stats_rel_area(percentualAreas)
        results.append(summarizedPercAreas)
    
    return results

def generate_stats_single(segmentation, camHeatmap, classes, proportions=False):
    """
    Creates arrays containing values for plotting of a single instance.
    :param segmentation: Raw Segmentation Mask. Must match shape (224,224) or (224,224,1)
    :type segmentation: np.ndarray
    :param camHeatmap: Raw CAM Data. Must macht shape (224,224) or (224,224,1)
    :type camHeatmap: np.ndarray
    :param classes: Classes corresponding to the categories of the segmentation.
    :type classes: list/tuple like object.
    :param proportions: Calculate proportions of each segment to the entire image region.
    :type proportions: bool

    :return list of [classArray, segmentedCAMActivations, percentualCAMActivations, dominantMask, (proportions)]
    """
    assert segmentation.shape[0] == segmentation.shape[1] == 224, f'Expected Shape (224,224) or (224,224,1) but got {segmentation.shape}'
    assert camHeatmap.shape[0] == camHeatmap.shape[1] == 224, f'Expected Shape (224,224) or (224,224,1) but got {camHeatmap.shape}'
    classArray = np.array(classes)
    segmentation = segmentation.squeeze()
    camHeatmap = camHeatmap.squeeze()
    binaryMasks = generateUnaryMasks(segmentation=segmentation, segmentCount=len(classes))

    segmentedCAM = [camHeatmap*mask for mask in binaryMasks]
    totalCAMActivation = camHeatmap.sum()
    segmentedCAMActivation = np.sum(segmentedCAM, axis=(1,2))
    percentualSegmentedCAMActivation = segmentedCAMActivation / totalCAMActivation

    dominantSegments = heapq.nlargest(3,segmentedCAMActivation)
    dominantMask = segmentedCAMActivation >= np.min(dominantSegments)

    results = [classArray, segmentedCAMActivation, percentualSegmentedCAMActivation, dominantMask]
    if proportions:
        results.append([segmentation[segmentation==c].size for c in np.arange(len(classes))] / np.prod(segmentation.shape))
    
    return results