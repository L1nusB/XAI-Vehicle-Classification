from mmseg.apis import init_segmentor
import warnings
import numpy as np

from .calculations import accumulate_statistics

def load_classes(classes=None, backgroundCls='background', addBackground=False, **kwargs):
    """
    Gets the classes for the given segmentation model and if specified adds a background category
    via segConfig and segCheckpoint and then adds the background class.
    The background class will ONLY be added if the parameter addBackground is True.
    The background class will not be added if it already is in the determined classes.
    """
    if classes is None:
        assert 'segConfig' in kwargs and 'segCheckpoint' in kwargs, 'Required segConfig and segCheckpoint if classes are not specified.'
        model = init_segmentor(kwargs['segConfig'], kwargs['segCheckpoint'])
        classes = model.CLASSES

    if backgroundCls in classes and addBackground:
        warnings.warn(f'Background Class {backgroundCls} already present in given Classes. Will not be added again.')
    else:
        classes = classes + (backgroundCls,) if addBackground else classes
    return classes

def batch_statistics(classes, imgNames, cams, segmentations, forceAll=False, limit=10000, percentualArea=False, **kwargs):
    """Splits the statistical Data for the given cams and segmentations into batches of size limit.
    The batches are collected into a list and then returned as a numpy array. If no batching should be
    done forceAll can be set to True then the results will be lists that only contain one element.

    :param classes: Classes of the segmentation
    :type classes: List like object
    :param imgNames: Names of all images for which statistics will be computed
    :type imgNames: List like
    :param cams: The CAM Data
    :type cams: dict
    :param segmentations: The Segmentation Masks
    :type segmentations: dict(str, np.ndarray)
    :param forceAll: Force all statistics into a single element without batching, defaults to False
    :type forceAll: Boolean, optional
    :param limit: Batch size, defaults to 10000
    :type limit: int, optional
    :return: List containing [totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations]
    """
    classArray = np.array(classes)
    numSamples = len(imgNames)
    numSplits = numSamples // limit + 1
    # Decrease by one if exact match
    if numSplits*limit == numSamples:
        numSplits-=1
    totalCAMActivations = [None]*numSplits
    segmentedCAMActivations = [None]*numSplits
    percentualSegmentedCAMActivations = [None]*numSplits
    percentualSegmentAreas = [None]*numSplits
    if numSamples > limit:
        warnings.warn(f'Statistics computed over {numSamples} items. Reverting to using batches of size {limit} '
        'to avoid overflows. Can be overriden by using forceAll=True')
    for batch in range(numSplits):
        lower = limit*batch
        higher = limit*(batch+1)
        if forceAll:
            result= accumulate_statistics(imgNames, cams, segmentations, classArray, percentualArea=percentualArea)
            if percentualArea:
                totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation, percentualSegmentArea = result
            else:
                totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation = result
        else:
            print(f'Generating data for Batch {batch+1}')
            result = accumulate_statistics(imgNames[lower:higher], cams, segmentations, classArray, percentualArea=percentualArea)
            if percentualArea:
                totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation, percentualSegmentArea = result
            else:
                totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation = result
        totalCAMActivations[batch] = totalCAMActivation
        segmentedCAMActivations[batch] = segmentedCAMActivation
        percentualSegmentedCAMActivations[batch] = percentualSegmentedCAMActivation
        if percentualArea:
            percentualSegmentAreas[batch] = percentualSegmentArea
    
    print('Data generated.')
    results = [totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations]

    if percentualArea:
        results.append(percentualSegmentAreas)

    return results