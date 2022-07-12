import numpy as np

from ..visualization_seg_masks import generateUnaryMasks


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