import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import os
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import mmcv
from mmcv import Config
from .visualization_seg_masks import generateUnaryMasks
import heapq

from . import new_gen_seg, generate_cams
from .utils.pipeline import get_pipeline_torchvision

def generate_bar_cam_intersection(segmentation, camHeatmap, classes):
    """Generate a bar plot showing the intersection of each segmentation region 
    with the given CAM.

    :param segmentation: Raw Segmentation Mask. Must match shape (224,224) or (224,224,1)
    :type segmentation: np.ndarray
    :param camHeatmap: Raw CAM Data. Must macht shape (224,224) or (224,224,1)
    :type camHeatmap: np.ndarray
    :param classes: Classes corresponding to the categories of the segmentation.
    :type classes: list/tuple like object.
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

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[0,1])

    bars = ax0.bar(classArray, segmentedCAMActivation)
    ticks_loc = ax0.get_xticks()
    ax0.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax0.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax0.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height:.0f}', ha='center', va='bottom' )
    ax0.set_title('Absolut CAM Activations')

    bars = ax1.bar(classArray, percentualSegmentedCAMActivation)
    ticks_loc = ax1.get_xticks()
    ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax1.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height:.1%}', ha='center', va='bottom' )
    ax1.set_title('Percentage CAM Activations')

def generate_bar_cam_intersection_prop_area(segmentation, camHeatmap, classes, showPropPercent=False):
    """Generate a bar plot showing the intersection of each segmentation region 
    with the given CAM as well as the proportion of the segment to the area. 

    :param segmentation: Raw Segmentation Mask. Must match shape (224,224) or (224,224,1)
    :type segmentation: np.ndarray
    :param camHeatmap: Raw CAM Data. Must macht shape (224,224) or (224,224,1)
    :type camHeatmap: np.ndarray
    :param classes: Classes corresponding to the categories of the segmentation.
    :type classes: list/tuple like object.
    :param showPropPercent: show Percentage values of the proportional areas. Formatting is not great so default false.
    :type showPropPercent: bool
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

    proportions = [segmentation[segmentation==c].size for c in np.arange(len(classes))] / np.prod(segmentation.shape)

    dominantSegments = heapq.nlargest(3,segmentedCAMActivation)
    dominantMask = segmentedCAMActivation >= np.min(dominantSegments)

    fig = plt.figure(figsize=(13,5),constrained_layout=True)

    ax0 = fig.add_subplot()

    # Default width is 0.8 and since we are plotting two bars side by side avoiding overlap requires
    # reducing the width
    barwidth = 0.4

    bars = ax0.bar(np.arange(classArray.size), percentualSegmentedCAMActivation, width=barwidth, label='CAM Activations')
    ax0.set_xticks([tick+barwidth/2 for tick in range(classArray.size)], classArray)
    ticks_loc = ax0.get_xticks()
    ax0.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax0.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMask):
        if dominant:
            bars[index].set_color('red')
    for index, bar in enumerate(bars):
        height = bar.get_height()
        # Dominant columns are located in center of bar.
        if dominantMask[index]:
            ypos = height/2.0
        else:
            # Move height up a tiny bit
            ypos = height + 0.01
        # Rotate Text by 90° if showPropPercent
        if showPropPercent:
            ax0.text(bar.get_x()+bar.get_width()/2.0, ypos , f'{height:.1%}', ha='center', va='bottom', rotation=90)
        else:
            ax0.text(bar.get_x()+bar.get_width()/2.0, height , f'{height:.1%}', ha='center', va='bottom')
    bars = ax0.bar(np.arange(classArray.size)+barwidth, proportions, width=barwidth, color='g', label='Proportional Segment Coverage')
    if showPropPercent:
        for index, bar in enumerate(bars):
            height = bar.get_height()
            # Dominant columns are located in center of bar.
            if dominantMask[index]:
                ypos = height/2.0
            else:
                # Move height up a tiny bit
                ypos = height + 0.01
            ax0.text(bar.get_x()+bar.get_width()/2.0, ypos , f'{height:.1%}', ha='center', va='bottom', rotation=90)
    ax0.set_title('Relative CAM Activations')

    legendMap = {
        'b':'CAM Activations',
        'r':'Top CAM Activations',
        'g':'Proportional Segment Coverage'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax0.legend(handles=handles)



def generate_image_instances(sourceImg, segmentationImg, camHeatmap, pipeline):
    """
    Transforms the given images to match the shape of the CAM and creates the CAM Overlay

    :param sourceImg: Original Imgae
    :param segmentationImg: Original segmentation Mask overlayed with source Image
    :param camHeatmap: The raw values of the CAM
    :param pipeline: The pipeline to be used for the transformations

    :return: Tuple (transformedSourceImg, transformedSegmentationImg, camOverlay) with same shapes
    """
    assert isinstance(sourceImg, np.ndarray) or isinstance(sourceImg, torch.Tensor), "sourceImg must be a np.ndarray or Tensor"
    assert isinstance(segmentationImg, np.ndarray) or isinstance(segmentationImg, torch.Tensor), "segmentationImg mus be a np.ndarray or Tensor"
    transformedSourceImg = pipeline(sourceImg)
    transformedSegmentationImg = pipeline(segmentationImg)
    assert transformedSourceImg.shape == transformedSegmentationImg.shape
    assert transformedSourceImg.shape[:-1] == camHeatmap.shape, "Shape of camHeatmap and transformed sourceImg after Pipeline does not match"
    if isinstance(transformedSourceImg, torch.Tensor):
        transformedSourceImg = transformedSourceImg.numpy()
    camOverlay = show_cam_on_image(transformedSourceImg, camHeatmap, use_rgb=False)
    return (transformedSourceImg,transformedSegmentationImg, camOverlay)


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

def plot(imgName, imgRoot,camConfig, camCheckpoint=None, camData=None,  imgData=None, 
    segmentationImgData=None, segConfig=None, segCheckpoint=None, segDevice='cuda',  camDevice='cpu'):
    """
    Generates an overview and plot for a single Image. 
    The Segmentation and CAM can be provided or will be generated.

    :param imgName: Name of the Image.
    :param imgRoot: Path to the directory containing the image.
    :param imgData: (np.ndarray) Array containing the image data. Must match shape (_,_,1) or (_,_,3)
    :param segmentationImgData: (Dictionary or np.ndarray) Image Data of the segmentation overlayed with the Image. If not given will be generated.
    :param camData: (Dictionary or np.ndarray) Raw CAM Data. If not given will be generated.
    :param segConfig: Path to Config file used for Segmentation. Required if no segmentationImgData is provided.
    :param segCheckpoint: Path to Checkpoint file used for Segmentation. Required if no segmentationImgData is provided.
    :param segDevice: Device used to generate Segmentation. (default 'cuda')
    :param camConfig: Path to Config File used for generating the CAM. Required if no camData is provided.
    :param camCheckpoint: Path tot Checkpoint File used for generating the CAM. Required if no camData is provided.
    :param camDevice: Device used for generating the CAM. (default 'cuda')
    """
    imgPath = os.path.join(imgRoot, imgName)
    assert os.path.isfile(imgPath), f'Specified Path does not yield valid file:{imgPath}'
    if imgData is not None:
        assert isinstance(imgData, np.ndarray), f'Expected type np.ndarray got {type(imgData)}'
        assert len(imgData.shape) == 3 and (imgData.shape[-1]==1 or imgData.shape[-1]==3), f'expected Shape (_,_,1) or (_,_,3) for imgData but got {imgData.shape}'
        sourceImg = np.copy(imgData)
    else:
        sourceImg = mmcv.imread(imgPath)
    if segmentationImgData is None:
        assert os.path.isfile(segConfig), f'segConfig:{segConfig} does not lead to a file'
        assert os.path.isfile(segCheckpoint), f'segCheckpoint:{segCheckpoint} does not lead to a file'
        temp_filePath = 'temp_ann.txt'
        with open(temp_filePath,'w') as tmp:
            tmp.write(imgName)
        #segmentationMasks, segmentationImgData, palette = new_gen_seg.main([imgPath, segConfig, segCheckpoint,'--types', 'masks', 'images','--device',segDevice])
        segmentationMasks, segmentationImgData = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--ann-file',os.path.abspath(temp_filePath), '-r','--types', 'masks', 'images','--device',segDevice])
        os.remove(temp_filePath)
    if camData is None:
        assert os.path.isfile(camConfig), f'camConfig:{camConfig} does not lead to a file'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint:{camCheckpoint} does not lead to a file'
        camData = generate_cams.main([imgPath, camConfig, camCheckpoint,'--device',camDevice])
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
    
    camConfig = Config.fromfile(camConfig)
    pipeline = get_pipeline_torchvision(camConfig.data.test.pipeline, scaleToInt=False , workPIL=True)

    transformedSourceImg, transformedSegmentationImg, camOverlay = generate_image_instances(sourceImg,segmentationImg,camHeatmap, pipeline)
    generate_overview(transformedSourceImg,transformedSegmentationImg,camHeatmap,camOverlay)
    