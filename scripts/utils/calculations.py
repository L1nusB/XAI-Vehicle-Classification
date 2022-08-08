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

    :return: Tuple containing (totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation, segmentPercentualArea)
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
        segmentPercentualArea = np.array([segmentation[segmentation==c].size for c in np.arange(len(classes))]) / np.prod(segmentation.shape)

    return totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation, segmentPercentualArea

def accumulate_statistics(imgNames, cams, segmentations, classes, percentualArea=False):
    """Accumulates all intersections and totalCAMActivations for the given imgNames.
    Here only the individual datapoints are gathered and are not immediatly interpretable by themselves without further computations.
    Returns accumulated data both for absolut and percentual statistics.

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
    print('Accumulating Statistics for given imgNames.')
    totalCAMActivations = []
    segmentedCAMActivations = []
    percentualSegmentedCAMActivations = []
    percentualSegmentAreas = []
    classArray = np.array(classes)

    for name in imgNames:
        totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation, percentualSegmentArea = accumulate_single(
            cams[name], segmentations[name], classArray, percentualArea=percentualArea)
        totalCAMActivations.append(totalCAMActivation)
        segmentedCAMActivations.append(segmentedCAMActivation)
        percentualSegmentedCAMActivations.append(percentualSegmentedCAMActivation)
        if percentualArea:
            percentualSegmentAreas.append(percentualSegmentArea)

    results = [np.array(totalCAMActivations), np.array(segmentedCAMActivations), np.array(percentualSegmentedCAMActivations)]
    if percentualArea:
        results.append(np.array(percentualSegmentAreas))
    
    return results

def generate_stats_abs(segmentedActivations):
    """
    Creates array containing values for the absolute statistics for plotting of multiple samples.
    :param segmentedActivations: CAM Activations per Segment
    :type segmentedActivations: np.ndarray(np.ndarray(float))

    :return  summarizedSegmentedCAMActivations, dominantMask
    """
    # #summarizedSegmentedCAMActivations = segmentedActivations.mean(axis=1).mean(axis=0)
    # # This simulates mean(axis=1).mean(axis=0) for lists that do not have to match sizes which can be the case.
    # n_samples = sum([x.shape[0] for x in segmentedActivations])
    # summarizedSegmentedCAMActivations= [sum([sum(batch[:,i])/n_samples for batch in segmentedActivations]) for i in range(segmentedActivations[0].shape[-1])]

    # New Version without any batches.
    summarizedSegmentedCAMActivations = segmentedActivations.mean(axis=0)

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
    # #summarizedPercSegmentedCAMActivations = percentualActivations.mean(axis=1).mean(axis=0)
    # # This simulates mean(axis=1).mean(axis=0) for lists that do not have to match sizes which can be the case.
    # n_samples = sum([x.shape[0] for x in percentualActivations])
    # summarizedPercSegmentedCAMActivations= [sum([sum(batch[:,i])/n_samples for batch in percentualActivations]) for i in range(percentualActivations[0].shape[-1])]

    # """
    # This is the more truthful batched calculations where the means are calculated within each batch by dividing by the number of elemnts of the batch and then averaging those.
    # In the above implementation while the sums are calculated within each batch the division is always by the true number of samples no matter which batch is used.
    # In theory this could lead to issues when the amount of samples is extremly high and the activations are comparably low but that is not really an issue.
    # Additionally by summing through the batches the divident will be much larger which makes divions not prone to underflows or similar.
    # And by using the true number to normalize we actually recreate the true mean instead by moving the divison inside a sum instead of introducing rounding errors.
    
    # batch_n_samples = [x.shape[0] for x in percentualActivations]
    # summarizedPercSegmentedCAMActivations= [sum([sum(batch[:,i])/batch_n_samples[index] for index, batch in enumerate(percentualActivations)]) / len(batch_n_samples) for i in range(percentualActivations[0].shape[-1])]
    # """

    summarizedPercSegmentedCAMActivations = percentualActivations.mean(axis=0)

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
    # # Ensure conversion to np.array
    # areas = np.array(percentualAreas)
    # # This simulates mean(axis=1).mean(axis=0) for lists that do not have to match sizes which can be the case.
    # n_samples = sum([x.shape[0] for x in areas])
    # summarizedPercSegmentedAreas = [sum([sum(batch[:,i])/n_samples for batch in areas]) for i in range(areas[0].shape[-1])]

    summarizedPercSegmentedAreas = percentualAreas.mean(axis=0)

    return summarizedPercSegmentedAreas

def generate_stats(classes, segmentedActivations=None, percentualActivations=None, totalCAM=None, percentualAreas=None):
    """
    Creates arrays containing values for plotting of multiple samples.
    :param classes: Classes corresponding to the categories of the segmentations.
    :type classes: list/tuple like object.
    :param segmentedActivations: CAM Activations per Segment
    :type segmentedActivations: np.ndarray(np.ndarray(float))
    :param percentualActivations: Percentual CAM Activations per Segment
    :type percentualActivations: np.ndarray(np.ndarray(float))
    :param totalCAM: Total CAM Activations (per batch)
    :type totalCAM: np.ndarray(np.ndarray(float))
    :param percentualAreas: Relative Area of each segment w.r.t the entire mask/image
    :type percentualAreas: np.ndarray(np.ndarray(float))

    :return: depending on what was specified: [classArray, totalActivation, summarizedSegmentedCAMActivations, summarizedPercSegmentedCAMActivations, dominantMask, dominantMaskPercentual,summarizedPercAreas] 
        
    """
    print('Generate Statistics Data')
    results = [np.array(classes)]

    # if totalCAM is not None:
    #     totalActivation = sum(np.sum(batch) for batch in totalCAM)
    #     # totalActivation = np.sum(totalCAM)
    #     results.append(totalActivation)
    # if segmentedActivations is not None:
    #     summarizedSegmentedCAMActivations, dominantMask = generate_stats_abs(segmentedActivations)
    #     results.append(summarizedSegmentedCAMActivations)
    #     results.append(dominantMask)
    # if percentualActivations is not None:
    #     summarizedPercSegmentedCAMActivations, dominantMaskPercentual = generate_stats_rel(percentualActivations)
    #     results.append(summarizedPercSegmentedCAMActivations)
    #     results.append(dominantMaskPercentual)
    # if percentualAreas is not None:
    #     summarizedPercAreas = generate_stats_rel_area(percentualAreas)
    #     results.append(summarizedPercAreas)

    if totalCAM is not None:
        # If an overflow here occurs then so be it. This is HIGHLY UNLIKELY
        totalActivation = np.sum(totalCAM)
        results.append(totalActivation)
    if segmentedActivations is not None:
        summarizedSegmentedCAMActivations, dominantMask = generate_stats_abs(segmentedActivations)
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