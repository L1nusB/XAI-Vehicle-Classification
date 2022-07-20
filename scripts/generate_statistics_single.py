import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os.path as osp
from .generate_cams import generate_cam_overlay

from .utils.pipeline import get_pipeline_torchvision, apply_pipeline
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.plot import plot_bar
from .utils.preprocessing import load_classes
from .utils.calculations import generate_stats_single
from .utils.io import get_save_figure_name, saveFigure

def generate_bar_cam_intersection(classes=None, **kwargs):
    """Generate a bar plot showing the intersection of each segmentation region 
    with the given CAM.
    :param classes: Classes corresponding to the categories of the segmentation. If not specified loaded from segConfig, segCheckpoint in kwargs.
    :type classes: list/tuple like object.

    for kwargs see :func:`prepareInput`
    """
    classes = load_classes(classes, **kwargs)
    
    segmentation,_, camHeatmap = prepareInput(prepImg=False, **kwargs)
    # Manual pipelineCfg can be set via kwargs parameter pipelineCfg
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipelineScale = get_pipeline_torchvision(cfg.data.test.pipeline, workPIL=True)
        segmentation = pipelineScale(segmentation)
    classArray, segmentedCAMActivation, percentualSegmentedCAMActivation, dominantMask = generate_stats_single(segmentation, camHeatmap, classes)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.set_title('Absolut CAM Activations')
    ax1 = fig.add_subplot(grid[0,1])
    ax1.set_title('Percentage CAM Activations')

    # Plot Absolute CAM Activations
    plot_bar(ax=ax0, x_ticks=classArray, data=segmentedCAMActivation, dominantMask=dominantMask, format='.0f')
    # Plot Relative CAM Activations
    plot_bar(ax=ax1, x_ticks=classArray, data=percentualSegmentedCAMActivation, dominantMask=dominantMask, format='.1%')

    figure_name = get_save_figure_name(statType='Single', **kwargs)

    saveFigure(savePath=osp.join("results", figure_name), figure=fig)



def generate_bar_cam_intersection_prop_area(classes=None, showPropPercent=False, **kwargs):
    """Generate a bar plot showing the intersection of each segmentation region 
    with the given CAM as well as the proportion of the segment to the area. 

    :param classes: Classes corresponding to the categories of the segmentation.  If not specified loaded from segConfig, segCheckpoint in kwargs.
    :type classes: list/tuple like object.
    :param showPropPercent: show Percentage values of the proportional areas. Formatting is not great so default false.
    :type showPropPercent: bool

    for kwargs see :func:`prepareInput`
    """
    classes = load_classes(classes, **kwargs)

    segmentation,_, camHeatmap = prepareInput(prepImg=False, **kwargs)

    # Manual pipelineCfg can be set via kwargs parameter pipelineCfg
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipelineScale = get_pipeline_torchvision(cfg.data.test.pipeline, workPIL=True)
        segmentation = pipelineScale(segmentation)

    classArray, _, percentualSegmentedCAMActivation, dominantMask, proportions = generate_stats_single(segmentation, camHeatmap, classes, proportions=True)

    fig = plt.figure(figsize=(13,5),constrained_layout=True)

    ax0 = fig.add_subplot()
    ax0.set_title('Relative CAM Activations')

    # Default width is 0.8 and since we are plotting two bars side by side avoiding overlap requires
    # reducing the width
    barwidth = 0.4

    bars = ax0.bar(np.arange(classArray.size), percentualSegmentedCAMActivation, width=barwidth, label='CAM Activations')
    ax0.set_xticks([tick+barwidth/2 for tick in range(classArray.size)], classArray)

    rotation = 90 if showPropPercent else 0
    # Format main Data with generated bar graph
    plot_bar(ax=ax0, bars=bars, dominantMask=dominantMask, x_tick_labels=classArray, format='.1%', textadjust_ypos=showPropPercent,
        textrotation=rotation)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, data=proportions, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMask, addText=showPropPercent, hightlightDominant=False,
        textadjust_ypos=showPropPercent, format='.1%', textrotation=rotation)

    legendMap = {
        'b':'CAM Activations',
        'r':'Top CAM Activations',
        'g':'Proportional Segment Coverage'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax0.legend(handles=handles)

    figure_name = get_save_figure_name(statType='Single', additional='ShowPropArea', **kwargs)

    saveFigure(savePath=osp.join("results", figure_name), figure=fig)


def generate_overview(sourceImg, segmentationImg, camHeatmap, camOverlay):
    """
    Generate a visualization of the different parts that are relevant for generating statistics and 
    a corresponding bar graph showing the overlap of the CAM and the segmentation regions

    :param sourceImg: The original Img
    :param segmentationImg: The segmentation Mask overlayed over the sourceImg
    :param camHeatmap: The raw values of the CAM
    :param camOverlay: The CAM overlayed over the sourceImg

    """


    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(4,2)
    axl0 = fig.add_subplot(gs[0,0])
    axl1 = fig.add_subplot(gs[1,0])
    axl2 = fig.add_subplot(gs[2,0])
    axl3 = fig.add_subplot(gs[3,0])
    axr = fig.add_subplot(gs[:,1])

    axl0.imshow(sourceImg, interpolation='nearest')
    axl0.axis('off')

    axl1.imshow(segmentationImg, interpolation='nearest')
    axl1.axis('off')

    axl2.imshow(camHeatmap, interpolation='nearest')
    axl2.axis('off')

    axl3.imshow(camOverlay, interpolation='nearest')
    axl3.axis('off')

    axr.imshow(sourceImg)
    axr.axis('off')

    saveFigure(osp.join("results", "Overview.jpg"), fig)

#def plot(imgName, imgRoot,camConfig, camCheckpoint=None, camData=None,  imgData=None, annfile='', method='gradcam', 
#    segmentationImgData=None, segConfig=None, segCheckpoint=None, segDevice='cuda',  camDevice='cpu'):
def plot(imgName, **kwargs):
    """
    Generates an overview and plot for a single Image. 
    The Segmentation and CAM can be provided or will be generated.

    :param imgName: Name of the Image.
    :param imgRoot: Path to the directory containing the image.
    :param imgData: (np.ndarray) Array containing the image data. Must match shape (_,_,1) or (_,_,3)
    :param segmentationData: (Dictionary or np.ndarray) Image Data of the segmentation overlayed with the Image. If not given will be generated.
    :param camData: (Dictionary or np.ndarray) Raw CAM Data. If not given will be generated.
    :param segConfig: Path to Config file used for Segmentation. Required if no segmentationImgData is provided.
    :param segCheckpoint: Path to Checkpoint file used for Segmentation. Required if no segmentationImgData is provided.
    :param segDevice: Device used to generate Segmentation. (default 'cuda')
    :param camConfig: Path to Config File used for generating the CAM. Required if no camData is provided.
    :param camCheckpoint: Path tot Checkpoint File used for generating the CAM. Required if no camData is provided.
    :param camDevice: Device used for generating the CAM. (default 'cuda')
    """
    kwargs['imgName'] = imgName # Add imgName to kwargs for consolidation in input.
    sourceImg, _, segmentationImgData, camData = prepareInput(**kwargs)
    assert segmentationImgData is not None, "segmentationImgData must not be none if generate is False"
    assert camData is not None, "Cam Overlay must not be none if generate is False"
    if isinstance(segmentationImgData, dict):
        segmentationImg = segmentationImgData[imgName]
    else:
        segmentationImg = np.copy(segmentationImgData)
    if isinstance(camData, dict):
        camHeatmap = camData[imgName]
    else:
        camHeatmap = np.copy(camData)

    cfg = get_pipeline_cfg(**kwargs)

    pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=False , workPIL=True)

    transformedSourceImg, transformedSegmentationImg = apply_pipeline(pipeline, sourceImg, segmentationImg)
    camOverlay = generate_cam_overlay(transformedSourceImg, camHeatmap)
    generate_overview(transformedSourceImg,transformedSegmentationImg,camHeatmap,camOverlay)

    