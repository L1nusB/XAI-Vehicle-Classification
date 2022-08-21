import numpy as np
import cv2
from mmcv import Config
import copy

from mmseg.datasets.builder import build_dataset, DATASETS
from mmcls.datasets.pipelines import Compose

from .preprocessing import load_classes
from ..ImageDataset import ImageDataset
from .constants import DATASETWRAPPERS
from .imageProcessing import convert_numpy_to_PIL
from .io import savePIL

@DATASETS.register_module('BlurDataset')
def get_blurred_dataset(cfgPath, imgRoot, annfile, blurredSegments, segData, classes=None, saveDir='blurredImgs/', saveImgs=False,
                        blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None, **kwargs):
    """Creates a Dataset that inherits from the given CAM Cfg but blurres specified parts of the image.

    :param cfgPath: Path to config of the CAM Model from which the dataset is inferred.
    :type cfgPath: str | Path
    :param imgRoot: Path to the directory where samples are stored. Will overwrite data_prefix entry in cfg.
    :type imgRoot: str | Path
    :param annfile: Path to the annotation file. Will overwrite ann_file entry in cfg.
    :type annfile: str | Path
    :param classes: List or array containing the classes of the segmentation. Used for ensuring blurredSegment is valid
                    If not specified classes will be loaded from segConfig and segCheckpoint
    :type classes: list(str) | np.ndarray(str)
    :param blurredSegments: Index specifying what segements to blur. Either by name or indexed
    :type blurredSegments: str | int | list(str|int)
    :param segData: Path to numpy file containig the segmentation data
    :type segData: str | Path
    :param saveDir: Path where to save blurredImages to, defaults to './'
    :type saveDir: str | Path, optional
    :param saveImgs: Whether to save the blurredImages to saveDir, defaults to False
    :type saveImgs: bool, optional
    :param blurKernel: Kernel used for gaussian Blurring, defaults to (33,33)
    :type blurKernel: tuple(int,int), optional
    :param blurSigmaX: sigmaX used for gaussian Blurring, defaults to 0
    :type blurSigmaX: int, optional
    :param segConfig: Path to segmentation Config. Required to load classes if not specified.
    :type segConfig: str | Path
    :param segCheckpoint: Path to segmentation Checkpoint. Required to load classes if not specified.
    :type segCheckpoint: str | Path
    """
    #Need to add ann_file setting for cfg.data.test
    cfg = Config.fromfile(cfgPath)
    datasetType = cfg.data.test.type
    assert datasetType in DATASETWRAPPERS or datasetType in DATASETS, f'Datasettype {datasetType} not supported.'
    # Differentiate between Wrapper because datasets from mmcls are added via Functions and therefore can not be get
    # from Registry and inherited from
    if datasetType in DATASETWRAPPERS:
        datasetType = DATASETWRAPPERS.get(datasetType)
    else:
        datasetType = DATASETS.get(datasetType)
    cfg.data.test['data_prefix'] = imgRoot
    cfg.data.test['ann_file'] = annfile
    if classes is None:
        assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if classes not given.'
        classes = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)

    class BlurDataset(datasetType):

        def __init__(self, **kwargs) -> None:
            super(BlurDataset, self).__init__(**kwargs)
            self.classes = classes
            self.prepipeline = Compose([{'type':'LoadImageFromFile'}])
            self.postpipeline = Compose([step for step in kwargs['pipeline'] if step['type'] != 'LoadImageFromFile'])
            if isinstance(blurredSegments, str):
                assert blurredSegments in classes, f'{blurredSegments} not in classes: {",".join(classes)}'
                self.blurredSegments = [classes.index(blurredSegments)]
            elif isinstance(blurredSegments, int):
                assert blurredSegments < len(classes), f'Can not blur segment no. {blurredSegments}. Only {len(classes)} present.'
                self.blurredSegments = [blurredSegments]
            elif isinstance(blurredSegments, list | np.ndarray):
                self.blurredSegments = []
                for segment in blurredSegments:
                    if isinstance(segment, str):
                        assert segment in classes, f'{segment} not in classes: {",".join(classes)}'
                        self.blurredSegments.append(classes.index(segment))
                    elif isinstance(segment, int):
                        assert segment < len(classes), f'Can not blur segment no. {segment}. Only {len(classes)} present.'
                        self.blurredSegments.append(segment)
                    else:
                        raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
                # Remove potential duplicates
                self.blurredSegments = list(set(self.blurredSegments))
            else:
                raise ValueError(f'blurredSegments must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegments)}')
            self.segData = np.load(segData)
            self.saveImgs = saveImgs
            self.saveDir = saveDir
            self.blurKernel = blurKernel
            self.blurSigmaX = blurSigmaX

        def __getitem__(self, idx):
            results = copy.deepcopy(self.data_infos[idx])
            results = self.prepipeline(results)
            ori_img = results['img']

            segmentation = self.segData[results['ori_filename']]

            assert segmentation.shape == ori_img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {ori_img.shape[:-1]}'

            mask = np.zeros_like(ori_img)
            for blur in self.blurredSegments:
                mask[segmentation==blur, :] = np.array([255,255,255])

            blur_img = cv2.GaussianBlur(ori_img, self.blurKernel, self.blurSigmaX)

            img = np.where(mask==np.array([255,255,255]), blur_img, ori_img)

            results['img'] = img

            if self.saveImgs:
                pil = convert_numpy_to_PIL(results['img'])
                savePIL(pil, fileName=results['ori_filename'], dir=self.saveDir, logSave=False)

            return self.postpipeline(results)
        
    cfg.data.test.type='BlurDataset'

    params = cfg.data.test.copy()
    params.pop('type')

    return BlurDataset(**params)