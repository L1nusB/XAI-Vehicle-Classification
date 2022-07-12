import numpy as np
from ..visualization_seg_masks import generateUnaryMasks
import heapq
import matplotlib.ticker as mticker

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


def highlight_dominants(bars, dominantMask, color='red'):
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color(color)


def add_text(bars, ax, format, adjust_ypos=False, dominantMask=None, **kwargs):
    for index, bar in enumerate(bars):
        height = bar.get_height()
        ypos = height
        if adjust_ypos:
            if dominantMask[index]:
                ypos /= 2
            else:
                ypos += 0.01
        ax.text(bar.get_x()+bar.get_width()/2.0, ypos, f'{height:{format}}', ha='center', va='bottom' , **kwargs)


def plot_bar(ax, x_ticks=None, data=None, dominantMask=None, hightlightDominant=True, x_tick_labels=None, 
            bars=None, addText=True,  format='.1%', **kwargs):
    if bars is None:
        barkwargs = {key[len('bar'):]:value for key,value in kwargs.items() if key.startswith('bar')}
        bars = ax.bar(x_ticks, data, **barkwargs)
    ticks_loc = ax.get_xticks()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    if x_tick_labels is not None:
        ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(x_ticks, rotation=45, ha="right")

    if hightlightDominant:
        highlight_dominants(bars, dominantMask)

    if addText:
        textkwargs = {key[len('text'):]:value for key,value in kwargs.items() if key.startswith('text')}
        add_text(bars, ax, format, dominantMask=dominantMask, **textkwargs)
    