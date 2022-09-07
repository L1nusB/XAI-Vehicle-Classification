import torchvision.transforms as transforms
import numpy as np
import warnings
import mmcv
import torch
import os.path as osp
import copy
from pathlib import Path

IMAGE_TRANSFORMS = {
'Resize':'size', 
'CenterCrop':'crop_size', 
'RandomCrop':'size'
}

from .constants import PIPELINEMAPS,PIPELINES,PIPELINETRANSFORMS
from .preprocessing import load_classes

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

def get_pipeline_torchvision(pipeline,scaleToInt=True, workPIL=False):
    components = []
    rescaleFactor = 255 if scaleToInt else 1
    resultType = 'uint8' if scaleToInt else 'float'
    expansion_dims = lambda x:tuple(range(4-len(x.shape)))  # This ensures that input is always of shape [NxCxHxW]
    if workPIL:
        components.append(transforms.Lambda(lambda x:x.astype('uint8')))
        components.append(transforms.ToPILImage())
    else:
        components.append(transforms.Lambda(lambda x:torch.from_numpy(np.expand_dims(x/255,expansion_dims(x)))))  # Rescale into [0,1] interval for Tensor Arithmetic and add two dimensions
    for step in pipeline:
        if step['type'] in IMAGE_TRANSFORMS:
            transform = get_transform_instance(step['type'])
            if IMAGE_TRANSFORMS[step['type']] in step:
                param = step[IMAGE_TRANSFORMS[step['type']]]
                if isinstance(param,tuple) and param[-1]==-1:
                    param = param[:-1]
                components.append(transform(param))
            else:
                warnings.warn(f"Required key {IMAGE_TRANSFORMS[step['type']]} not in pipeline step {step}!")
    if workPIL:
        # Convert PIL back to Tensor
        components.append(transforms.ToTensor())
        # Remove extra dims, Convert to Numpy, switch axes, Rescale to [0,255] and then set type to uint8 
        components.append(lambda x:(x.squeeze().numpy()*rescaleFactor))
        components.append(lambda x: (x.transpose(1,2,0) if len(x.shape)==3 else x).astype(resultType))
    else:
        components.append(transforms.Lambda(lambda x:x.squeeze().numpy()*rescaleFactor))  # Remove extra dims and scale up before translating to numpy array.
    return transforms.Compose(components)

def get_pipeline_pre_post(args, default_mapping='cls', scaleToInt=True, workPIL=True):
    prePipeline = None
    postPipeline = None
    if len(args.pipeline)>0:
        posSpecifier = [specifier for specifier in args.pipeline if (specifier.lower() == 'pre' or specifier.lower() == 'post')]
        assert len(posSpecifier)==1,'Either pre or post must be specified when using pipeline.'
        posSpecifier = posSpecifier[0]
        predefined = [specifier for specifier in args.pipeline if specifier in PIPELINES]
        assert len(predefined) < 2, 'Only one predefined Pipeline can be specified.'
        if len(predefined)>0:
            pipeline = PIPELINES[predefined[0]][posSpecifier]
            print(f'Using predefined pipeline: {predefined[0]}')
        else:
            configPath = [specifier for specifier in args.pipeline if osp.isfile(specifier)]
            assert len(configPath)==1, 'Exactly one config Path must be specified when using pipeline.'
            pipeline = mmcv.Config.fromfile(configPath[0]).data.test.pipeline
            print(f'Loading Pipeline from Config {configPath[0]}')

            mapSpecifier = [specifier for specifier in args.pipeline if specifier in PIPELINEMAPS]
            if len(mapSpecifier)>0:
                assert len(mapSpecifier)==1,'Only one Map Specifier can be applied.'
                if posSpecifier.lower()=='post':
                    warnings.warn('Using a Map Specifier for post processing is not advised and will likely raise Exceptions.')
                mapping = PIPELINEMAPS[mapSpecifier[0]]
            else:
                # Only add a default mapping if it is not post since there we need a torchvision pipeline will does not like the custom mapped names.
                # Otherwise set it to none
                if posSpecifier.lower()=='post':
                    mapping = PIPELINEMAPS['none']
                else:
                    print(f'No pipeline map specified. Using default {default_mapping}. Specify None if non is desired.')
                    mapping = PIPELINEMAPS[default_mapping]
            for step in pipeline:
                if step['type'] in mapping:
                    step['type'] = mapping[step['type']]
            
        transformPipelineSteps = []
        for step in pipeline:
                if step['type'] in PIPELINETRANSFORMS:
                    transformPipelineSteps.append(step)
        if posSpecifier.lower() == 'pre':
            prePipeline = transformPipelineSteps
            postPipeline = None
        else:
            prePipeline = None
            postPipeline = get_pipeline_torchvision(transformPipelineSteps, scaleToInt=scaleToInt, 
                                workPIL=workPIL)
    return prePipeline, postPipeline

def get_pipeline_from_cfg(cfg, scaleToInt=True, workPIL=True, transposeResult=True):
    """
    Creates a torchvisions pipeline from the given pipeline configuration
    based on the contained transformation elements. 
    The cfg can be either given as a dictionary or as a config file which
    will be loaded.
    """
    cfgPipeline = None
    if isinstance(cfg, str) or isinstance(cfg, Path):
        print(f'Loading config from File {cfg}')
        cfgPipeline = mmcv.Config.fromfile(cfg).data.test.pipeline
    else:
        assert isinstance(cfg,dict), 'Given cfg is no path like object or dictionary.'
        cfgPipeline = copy.copy(cfg)
    
    transformPipelineSteps = []
    for step in cfgPipeline:
        if step['type'] in PIPELINETRANSFORMS:
            transformPipelineSteps.append(step)
    pipeline = get_pipeline_torchvision(transformPipelineSteps, scaleToInt=scaleToInt, workPIL=workPIL)
    return pipeline
    


def apply_pipeline(pipeline, *args):
    """
    Applies the given pipeline to all objects passed in args and returns them 
    in the same order.
    """
    results = []
    for obj in args:
        assert isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor), f"object {obj} must be a np.ndarray or Tensor"
        results.append(pipeline(obj))
    return results

def add_blurring_pipeline_step(cfg, blurredSegments, segData, segConfig=None, segCheckpoint=None, segCategories=None,
                                blurKernel=(33,33), blurSigmaX=0, saveDir='blurredImgs/', saveImgs=False,
                                randomBlur=False, **kwargs):
    """
    Adds the BlurSegment Step into the Pipeline for the given cfg (under cfg.data.test.pipeline)
    This step is inserted directly after the LoadFromFile step.

    :param cfg: Config into which the step is added
    :type cfg: _type_
    :param blurredSegments: Blurred Segments. Will be passed to constructor of pipeline which will adapt to the format given
    :type blurredSegments: str | int | List(str|int)
    :param segData: Path to file containing the segmentation data. The data must match the original size at it will be applied on the original image
    :type segData: str | Path
    :param blurKernel: Kernel used for gaussian blurring, defaults to (33,33)
    :type blurKernel: tuple(int,int), optional
    :param blurSigmaX: SigmaX used for gaussian blurring, defaults to 0
    :type blurSigmaX: int, optional
    :param segConfig: Path to Config file of segmentation model used for loading the segmentation Categories/classes, defaults to None
    :type segConfig: str | Path, optional
    :param segCheckpoint: Path to Checkpoint file of segmentation model used for loading the segmentation Categories/classes, defaults to None
    :type segCheckpoint: str | Path, optional
    :param saveDir: Directory where blurred images will be saved to if saveImgs=True, defaults to 'blurredImgs/'
    :type saveDir: str | Path, optional
    :param saveImgs: Whether or not to save the blurred images, defaults to False
    :type saveImgs: bool, optional
    :param randomBlur: Blur random pixels with the same area as the blurred segments
    :type randomBlur: bool, optional 
    :param segCategories: List of segment categories. Used to validate blurred Segments are valid.
    :type segCategories: list | np.ndarray

    :return Modified config
    """
    pipeline = cfg.data.test.pipeline
    indexLoad = pipeline.index({'type': 'LoadImageFromFile'}) + 1
    if segCategories is None:
        assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if segCategories not given.'
        segCategories = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)
    pipeline = pipeline[:indexLoad] + \
                [{'type':'BlurSegments', 'blurredSegments':blurredSegments, 'segData':segData,
                'segCategories':segCategories, 'blurKernel':blurKernel, 'randomBlur' : randomBlur,
                'blurSigmaX':blurSigmaX, 'saveImgs':saveImgs, 'saveDir':saveDir}] + \
                pipeline[indexLoad:]
    cfg.data.test.pipeline = pipeline
    return cfg