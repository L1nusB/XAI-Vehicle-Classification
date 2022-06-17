import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import mmcv
from mmcv import Config
from visualization_seg_masks import generateUnaryMasks
import transformations


def generate_statistc(cam, segmentation, segmentCount):
    """Generate intersections stastics both absolute and percentual.

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