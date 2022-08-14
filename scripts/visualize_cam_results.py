import warnings
import numpy as np
import os

from .generate_cams import generate_cam_overlay
from .utils.io import get_samples, savePIL
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.imageProcessing import add_text, concatenate_images, convert_numpy_to_PIL
from .utils.pipeline import get_pipeline_torchvision, apply_pipeline

VISUALIZE_TYPES=['heatmap','overlay']

def visualize_cpu_vs_gpu(saveDir='', visType='overlay',**kwargs):
    """Create images showing the CAM results for CPU and GPU comparing them.
    The type can be the raw CAM Heatmap or the CAM Overlay over the original image

    :param saveDir: Directory where the images will be saved to. It will be appended by '/imgs/'
    :type saveDir: str
    :param visType: Type of the visualizations images. Must be included in VISUALIZE_TYPES (Currently ['heatmap','overlay'])
    :type visType: str

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
    assert visType in VISUALIZE_TYPES, f'Specified type {visType} not the available types: {",".join(VISUALIZE_TYPES)}'

    gpuKwargs = {key[len('gpu'):]:value for key, value in kwargs.items() if key.lower().startswith('gpu')}
    cpuKwargs = {key[len('cpu'):]:value for key, value in kwargs.items() if key.lower().startswith('cpu')}

    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'imgRoot' in kwargs, 'imgRoot must be specified.'
        imgNames = get_samples(**kwargs) # Required annfile or (imgRoot) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    print(f'Generate results for type {visType}')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    camsGPU = prepareInput(prepImg=False, prepSeg=False, prepCam=True, **kwargs, **gpuKwargs)[0] # Need to index here since we return a list
    camsCPU = prepareInput(prepImg=False, prepSeg=False, prepCam=True, **kwargs, **cpuKwargs)[0] # Need to index here since we return a list
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given GPU CAM Dictionary does not contain all imgNames as keys.'
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given CPU CAM Dictionary does not contain all imgNames as keys.'

    if visType == VISUALIZE_TYPES[1]:
        # Only load sourceImgs if we need them for overlays
        sourceImgs = prepareInput(prepImg=True, prepSeg=False, prepCam=False, **kwargs)[0] # Need to index here since we return a list
        transformedSourceImgs = {}
        cfg = get_pipeline_cfg(**kwargs)
        if cfg:
            pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=False, workPIL=True)
            print('Tranforming sourceImages with the given pipeline.')
        for name in imgNames:
            transformedSourceImgs[name] = pipeline(sourceImgs[name]) if cfg else sourceImgs[name]

    saveDir = os.path.join(saveDir, 'imgs')
    print(f'Saving images into directory {saveDir}')
    for name in imgNames:
        if visType == VISUALIZE_TYPES[0]:
            # Use CAM Heatmaps
            cpuImg = convert_numpy_to_PIL(camsCPU[name])
            gpuImg = convert_numpy_to_PIL(camsGPU[name])
        elif visType == VISUALIZE_TYPES[1]:    
            cpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsCPU[name]))
            gpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsGPU[name]))
        else:
            raise ValueError(f'Given type {visType} is unknown. Please use one of {",".join(VISUALIZE_TYPES)}')
        add_text(cpuImg, 'CPU')
        add_text(gpuImg, 'GPU')
        combinedImg = concatenate_images(cpuImg, gpuImg)
        nameParts = name.split('.')
        if len(nameParts) > 1:
            fName = ".".join(nameParts[:-1])
            fExtension = "." + nameParts[-1]
        else:
            fName = name
            fExtension = ""
        savePIL(combinedImg, fileName=fName + "_" + visType + fExtension, dir=saveDir)