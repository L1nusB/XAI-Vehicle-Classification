from matplotlib.pyplot import plt
import os
import numpy as np
import torchvision.transforms as transforms
import torch
from pytorch_grad_cam.utils.image import show_cam_on_image
import generate_segmentation
import generate_cams

IMAGE_TRANSFORMS = {'Resize':'size', 'CenterCrop':'crop_size'}

def get_transform_instance(cls):
    """
    Generates an instance of the given class from torchvision.transforms.

    :param cls: Class name of the torchvision.transforms object

    :return: Uninstantiated Class object of specified type 
    """
    m = __import__('torchvision')
    m = getattr(m, 'transforms')
    m = getattr(m, cls)
    return m

def get_pipeline_from_config_pipeline(pipeline, img_transforms = IMAGE_TRANSFORMS, tensorize=True, convertToPIL=True, transpose=True):
    """
    Constructs a torchvision pipeline from a config dictionary describing the pipeline from a model.

    :param pipeline: Dictionary containing the pipeline components.
    :param img_transforms: Dictionary of relevant transformations that are taken over from the pipeline dictionary.
    :param tensorize: Add a ToTensor Step in the pipeline (default True)
    :param convertToPIL: Convert input to PIL image requires input to be ndarray or tensor (default True)
    :param transpose: Transpose resulting image shape from (x1,x2,x3) to (x2,x3,x1) (default True)

    :return: Pipeline object
    """
    components = []
    if convertToPIL:
        components.append(transforms.ToPILImage())
    for step in pipeline:
        if step['type'] in img_transforms.keys():
            transform = get_transform_instance(step['type'])
            param = step[img_transforms[step['type']]]
            if isinstance(param,tuple) and param[-1]==-1:
                param = param[:-1]
            components.append(transform(param))
    if tensorize:
        components.append(transforms.ToTensor())
    if transpose:
        components.append(transforms.Lambda(lambda x: np.transpose(x,(1,2,0))))
    return transforms.Compose(components)

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
    camOverlay = show_cam_on_image(transformedSourceImg.numpy(), camHeatmap, use_rgb=False)
    return (transformedSourceImg,transformedSegmentationImg, camOverlay)


def generate_single_overview(sourceImg, segmentationImg, camHeatmap, camOverlay):
    """
    Generate a visualization of the different parts that are relevant for generating statistics and 
    a corresponding bar graph showing the overlap of the CAM and the segmentation regions

    :param sourceImg: The original Img
    :param segmentationImg: The segmentation Mask overlayed over the sourceImg
    :param camHeatmap: The raw values of the CAM
    :param camOverlay: The CAM overlayed over the sourceImg
    """
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gripspec(4,2)
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

def plot_single(imgName, imgRoot, segmentationImgData=None, camData=None, 
    segConfig=None, segCheckpoint=None, segDevice='cuda',camConfig=None, camCheckpoint=None, camDevice='cuda'):
    """
    Generates an overview and plot for a single Image. 
    The Segmentation and CAM can be provided or will be generated.

    :param imgName: Name of the Image.
    :param imgRoot: Path to the directory containing the image.
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
    if segmentationImgData is None:
        assert os.path.isfile(segConfig), f'segConfig:{segConfig} does not lead to a file'
        assert os.path.isfile(segCheckpoint), f'segCheckpoint:{segCheckpoint} does not lead to a file'
        _, segmentationImgData = generate_segmentation.main([imgPath, segConfig, segCheckpoint,'--types', 'masks', 'images'])
    if camData is None:
        assert os.path.isfile(camConfig), f'camConfig:{camConfig} does not lead to a file'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint:{camCheckpoint} does not lead to a file'
        camData = generate_cams.main([imgPath, camConfig, camCheckpoint])
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
    