import torchvision.transforms as transforms
import numpy as np
import warnings
import mmcv
import torch

IMAGE_TRANSFORMS = {
'Resize':'size', 
'CenterCrop':'crop_size', 
'RandomCrop':'size'
}

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

def get_pipeline(pipeline):
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