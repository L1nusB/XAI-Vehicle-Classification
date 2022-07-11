import numpy as np
from ..visualization_seg_masks import generateUnaryMasks
import heapq

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
