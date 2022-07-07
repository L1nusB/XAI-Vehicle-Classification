import torchvision.transforms as transforms
import numpy as np
import warnings
import mmcv
import torch
import os.path as osp

IMAGE_TRANSFORMS = {
'Resize':'size', 
'CenterCrop':'crop_size', 
'RandomCrop':'size'
}

from .constants import PIPELINEMAPS,PIPELINES,PIPELINETRANSFORMS

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

def get_pipeline_torchvision(pipeline):
    components = []
    components.append(transforms.Lambda(lambda x:torch.from_numpy(np.expand_dims(x/255,(0,1)))))  # Rescale into [0,1] interval for Tensor Arithmetic and add two dimensions
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
    components.append(transforms.Lambda(lambda x:x.squeeze().numpy()*255))  # Remove extra dims and scale up before translating to numpy array.
    return transforms.Compose(components)

def get_pipeline(args):
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
                    print("ASDASD")
                    warnings.warn('Using a Map Specifier for post processing is not advised and will likely raise Exceptions.')
                mapping = PIPELINEMAPS[mapSpecifier[0]]
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
            postPipeline = get_pipeline_torchvision(transformPipelineSteps)
    return prePipeline, postPipeline