import torchvision.transforms as transforms
import numpy as np
import mmcv
import torch

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

def get_pipeline_from_config_pipeline(pipeline, img_transforms = IMAGE_TRANSFORMS, toNumpy=True,
                            tensorize=True, convertToPIL=True, transpose=True, scaleToInt=False):
    """
    Constructs a torchvision pipeline from a config dictionary describing the pipeline from a model.

    :param pipeline: Dictionary containing the pipeline components.
    :param img_transforms: Dictionary of relevant transformations that are taken over from the pipeline dictionary.
    :param toNumpy: Convert results into a numpy ndarray. Only works if tensorize=True.
    :param tensorize: Add a ToTensor Step in the pipeline (default True)
    :param convertToPIL: Convert input to PIL image requires input to be ndarray or tensor (default True)
    :param transpose: Transpose resulting image shape from (x1,x2,x3) to (x2,x3,x1) (default True)
    :param scaleToInt: Multiply resulting tensor object by 255 (default False)

    :return: Pipeline object
    """
    components = []
    if convertToPIL:
        components.append(transforms.ToPILImage())
    for step in pipeline:
        if step['type'] in img_transforms.keys():
            transform = get_transform_instance(step['type'])
            # Make sure parameter is in component of pipeline
            if img_transforms[step['type']] in step:
                param = step[img_transforms[step['type']]]
                if isinstance(param,tuple) and param[-1]==-1:
                    param = param[:-1]
                components.append(transform(param))
    if tensorize:
        components.append(transforms.ToTensor())
        if toNumpy:
            components.append(transforms.Lambda(lambda x: x.numpy()))
    if transpose:
        components.append(transforms.Lambda(lambda x: np.transpose(x,(1,2,0))))
    if scaleToInt:
        components.append(transforms.Lambda(lambda x:x*255))
    return transforms.Compose(components)




def pipeline_transform(source, pipeline):
    """Transforms the given source Image using the specified pipeline.
    If type of source is integer it is casted to uint8 before applying the pipeline.

    :param source: Source Image
    :type source: np.ndarray/torch.Tensor or str
    :param pipeline: Pipeline that is applied.
    """
    assert isinstance(source, np.ndarray) or isinstance(source, torch.Tensor) or isinstance(source, str), "sourceImg must be a np.ndarray, Tensor or string"
    if isinstance(source,np.ndarray) and np.issubdtype(source.dtype, np.integer):
        source = source.astype('uint8')
    if isinstance(source,str):
        source = mmcv.imread(source)
    return pipeline(source)