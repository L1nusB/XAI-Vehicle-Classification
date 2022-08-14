import numpy as np
import os

from .utils.io import get_samples
from .utils.preprocessing import load_classes
from .utils.prepareData import prepareInput, get_pipeline_cfg, get_pipeline_torchvision

VISUALIZE_TYPES=['heatmap','overlay']

def visualize_cpu_vs_gpu(saveDir='', type='overlay',**kwargs):
    """Create images showing the CAM results for CPU and GPU comparing them.
    The type can be the raw CAM Heatmap or the CAM Overlay over the original image

    :param saveDir: Directory where the images will be saved to
    :type saveDir: str
    :param type: Type of the visualizations images. Must be included in VISUALIZE_TYPES (Currently ['heatmap','overlay'])
    :type type: str

    The kwargs for the CAMs are specified by prefixing them with 'gpu' or 'cpu' respectively
    Relevant kwargs are:
    imgRoot: Path to root folder where images/samples lie
    (cpu/gpu)camConfig: Path to config of the used CAM Model
    (cpu/gpu)camCheckpoint: Path to Checkpoint of the used CAM Model
    (cpu/gpu)camData: Path to file containing generated CAMs (or dictionary). Can be used instead of Config and Checkpoint
    method: Method used for generating the CAMs. Defaults to 'gradcam'
    annfile: Path to annotation file specifng which samples should be used.
    dataClasses: Array of Class Prefixes that specify which sample classes should be used. If not specified everything will be generated.
    """
    assert type in VISUALIZE_TYPES, f'Specified type {type} not the available types: {",".join(VISUALIZE_TYPES)}'

    gpuKwargs = {key[len('gpu'):]:value for key, value in kwargs.items() if key.lower.startswith('gpu')}
    cpuKwargs = {key[len('cpu'):]:value for key, value in kwargs.items() if key.lower.startswith('cpu')}

    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'imgRoot' in kwargs, 'imgRoot must be specified.'
        imgNames = get_samples(**kwargs) # Required annfile or (imgRoot) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')


    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    sourceImgs, camsGPU = prepareInput(prepImg=True, prepSeg=False, prepCam=True, **kwargs, **gpuKwargs)
    camsGPU = prepareInput(prepImg=False, prepSeg=False, prepCam=True, **kwargs, **cpuKwargs)
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given GPU CAM Dictionary does not contain all imgNames as keys.'
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given CPU CAM Dictionary does not contain all imgNames as keys.'

    transformedSourceImgs = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
        print('Tranforming segmentation masks with the given pipeline.')
    for name in imgNames:
        transformedSourceImgs[name] = pipeline(sourceImgs[name]) if cfg else sourceImgs[name]