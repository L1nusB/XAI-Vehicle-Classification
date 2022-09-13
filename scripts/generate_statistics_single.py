import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os.path as osp

import mmcv

from .generate_cams import generate_cam_overlay
from .utils.pipeline import get_pipeline_torchvision, apply_pipeline
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.plot import plot_bar
from .utils.preprocessing import load_classes
from .utils.calculations import generate_stats_single
from .utils.io import get_save_figure_name, saveFigure
from .utils.constants import RESULTS_PATH, FIGUREFORMATS

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

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(statType='Single', **kwargs)

    saveFigure(savePath=osp.join(RESULTS_PATH, figure_name), figure=fig)



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
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray ,data=proportions, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMask, addText=showPropPercent, hightlightDominant=False,
        textadjust_ypos=showPropPercent, format='.1%', textrotation=rotation)

    legendMap = {
        'b':'CAM Activations',
        'r':'Top CAM Activations',
        'g':'Proportional Segment Coverage'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax0.legend(handles=handles)

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(statType='Single', additional='ShowPropArea', **kwargs)

    saveFigure(savePath=osp.join(RESULTS_PATH, figure_name), figure=fig)


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

    saveFigure(osp.join(RESULTS_PATH, "Overview.pdf"), fig)

#def plot(imgName, imgRoot,camConfig, camCheckpoint=None, camData=None,  imgData=None, annfile='', method='gradcam', 
#    segmentationImgData=None, segConfig=None, segCheckpoint=None, segDevice='cuda',  camDevice='cpu'):
def plot(imgName, **kwargs):
    """
    Generates an overview and plot for a single Image. 
    The Segmentation and CAM can be provided or will be generated.

    :param imgName: Name of the Image.
    :param imgRoot: Path to the directory containing the image.
    :param imgPath: Alternative to combining imgName and imgRoot
    :param imgData: (np.ndarray) Array containing the image data. Must match shape (_,_,1) or (_,_,3)
    :param camData: (Dictionary or np.ndarray) Raw CAM Data. If not given will be generated.
    :param segConfig: Path to Config file used for Segmentation. Required if no segmentationImgData is provided.
    :param segCheckpoint: Path to Checkpoint file used for Segmentation. Required if no segmentationImgData is provided.
    :param segDevice: Device used to generate Segmentation. (default 'cuda')
    :param camConfig: Path to Config File used for generating the CAM. Required if no camData is provided.
    :param camCheckpoint: Path tot Checkpoint File used for generating the CAM. Required if no camData is provided.
    :param camDevice: Device used for generating the CAM. (default 'cuda')
    :param channel_order: Order of channels when loading an image. Default here is rgb
    """
    kwargs['imgName'] = imgName # Add imgName to kwargs for consolidation in input.
    if 'segData' in kwargs:
        warnings.simplefilter('always')
        warnings.warn('Specifiyng segData will lead to image Data not being generated. Ignoring parameter.')
        kwargs['segData'] = None
    kwargs['channel_order'] = kwargs.get('channel_order', 'rgb')  # Specify rgb order since default is bgr
    sourceImgDict, _, segmentationImgData, camData = prepareInput(**kwargs)
    sourceImg = list(sourceImgDict.values())[0] # Need to key index here because we now return dictionaries
    if isinstance(segmentationImgData, dict):
        segmentationImg = segmentationImgData[imgName]
    else:
        segmentationImg = np.copy(segmentationImgData)
    if isinstance(camData, dict):
        camHeatmap = camData[imgName]
    else:
        camHeatmap = np.copy(camData)

    cfg = get_pipeline_cfg(**kwargs)

    assert cfg is not None, 'No pipeline could be determined. Specify either pipelineCfg or camConfig'

    pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=False , workPIL=True)

    transformedSourceImg, transformedSegmentationImg = apply_pipeline(pipeline, sourceImg, segmentationImg)
    camOverlay = generate_cam_overlay(transformedSourceImg, camHeatmap)
    generate_overview(transformedSourceImg,transformedSegmentationImg,camHeatmap,camOverlay)

def plot_cam(fileName='CamOverview', **kwargs):
    """
    Plots cam that is generated based on given parameters along side its overlay on top of the original image

    :param pipelineCfg: Path to config from which to load the pipeline.
    :param camConfig: Path from which the cam model will be loaded and pipeline is loaded if pipelineCfg not given.
    :param camCheckpoint: Path from which the cam checkpoint will be loaded.
    :param camData: (Dictionary or np.ndarray) Raw CAM Data. If not given will be generated.
    :param camDevice: Device used for generating the CAM. (default 'cuda')
    :param imgPath: Full path to image.
    :param imgName: Name of the image.
    :param imgRoot: Path to directory where image is located at.
    :param channel_order: Order of channels when loading an image. Default here is rgb
    :param targetCategory: For what class the CAM is generated.
    """
    if kwargs.get('imgPath'):
        imgName = osp.basename(kwargs['imgPath'])
    else:
        assert 'imgName' in kwargs and 'imgRoot' in kwargs, 'If imgPath is not specified imgName and imgRoot must be given.'
        imgName = kwargs.get('imgName')

    cfg = get_pipeline_cfg(**kwargs)
    assert cfg is not None, 'No pipeline could be determined. Specify either pipelineCfg or camConfig'
    pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=False , workPIL=True)

    kwargs['channel_order'] = kwargs.get('channel_order', 'rgb')  # Specify rgb order since default is bgr

    fig = plt.figure(constrained_layout=True)

    if isinstance(kwargs.get('targetCategory'), list):
        categories = [''] + kwargs.get('targetCategory') # Add default class at front
        count = len(categories)
        gs = fig.add_gridspec(count,3)

        for index, category in enumerate(categories):
            kwargs['targetCategory'] = category
            sourceImgDict, camData = prepareInput(prepSeg=False, **kwargs)
            sourceImg = list(sourceImgDict.values())[0] # Need to key index here because we now return dictionaries
            camHeatmap = camData[imgName]

            transformedSourceImg = apply_pipeline(pipeline, sourceImg)[0] # Need to index here because we return a list
            camOverlay = generate_cam_overlay(transformedSourceImg, camHeatmap)

            aximg = fig.add_subplot(gs[index,0])
            axcam = fig.add_subplot(gs[index,1])
            axoverlay = fig.add_subplot(gs[index,2])

            aximg.imshow(transformedSourceImg, interpolation='nearest')
            #aximg.axis('off')
            aximg.set_xlabel(f'({3*index+1})')
            aximg.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)

            axcam.imshow(camHeatmap, interpolation='nearest')
            #axcam.axis('off')
            axcam.set_xlabel(f'({3*index+2})')
            axcam.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)

            axoverlay.imshow(camOverlay, interpolation='nearest')
            #axoverlay.axis('off')
            axoverlay.set_xlabel(f'({3*index+3})')
            axoverlay.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)
    else:
        sourceImgDict, camData = prepareInput(prepSeg=False, **kwargs)
        sourceImg = list(sourceImgDict.values())[0] # Need to key index here because we now return dictionaries
        camHeatmap = camData[imgName]

        transformedSourceImg = apply_pipeline(pipeline, sourceImg)[0] # Need to index here because we return a list
        camOverlay = generate_cam_overlay(transformedSourceImg, camHeatmap)

        gs = fig.add_gridspec(1,3)
        aximg = fig.add_subplot(gs[0,0])
        axcam = fig.add_subplot(gs[0,1])
        axoverlay = fig.add_subplot(gs[0,2])

        aximg.imshow(transformedSourceImg, interpolation='nearest')
        #aximg.axis('off')
        aximg.set_xlabel('(1)')
        aximg.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)

        axcam.imshow(camHeatmap, interpolation='nearest')
        #axcam.axis('off')
        axcam.set_xlabel('(2)')
        axcam.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)

        axoverlay.imshow(camOverlay, interpolation='nearest')
        #axoverlay.axis('off')
        axoverlay.set_xlabel('(3)')
        axoverlay.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False, length=0)

    if osp.basename(fileName).split(".")[-1] in FIGUREFORMATS:
        fileName = fileName
    else:
        fileName = fileName + '.pdf'

    saveFigure(osp.join(RESULTS_PATH, fileName), fig)
