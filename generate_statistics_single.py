import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import generate_segmentation
import generate_cams
import mmcv
from mmcv import Config
from visualization_seg_masks import generateUnaryMasks

import transformations

def generate_bar_cam_intersection(segmentation, camHeatmap, classes):
    """Generate a bar plot showing the intersection of each segmentation region 
    with the given CAM.

    :param segmentation: Raw Segmentation Mask. Must match shape (244,244) or (244,244,1)
    :type segmentation: np.ndarray
    :param camHeatmap: Raw CAM Data. Must macht shape (244,244) or (244,244,1)
    :type camHeatmap: np.ndarray
    :param classes: Classes corresponding to the categories of the segmentation.
    :type classes: list/tuple like object.
    """
    binaryMasks = generateUnaryMasks(segmentation=segmentation, segmentCount=len(classes))
    segmentation = segmentation.squeeze()
    camHeatmap = camHeatmap.squeeze()

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
        segmentationMasks, segmentationImgData, palette = generate_segmentation.main([imgPath, segConfig, segCheckpoint,'--types', 'masks', 'images','--device',segDevice])
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
    pipeline = transformations.get_pipeline_from_config_pipeline(camConfig.data.test.pipeline)

    transformedSourceImg, transformedSegmentationImg, camOverlay = generate_image_instances(sourceImg,segmentationImg,camHeatmap, pipeline)
    generate_overview(transformedSourceImg,transformedSegmentationImg,camHeatmap,camOverlay)
    