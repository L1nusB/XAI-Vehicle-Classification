import numpy as np
import cv2
import copy

from mmseg.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose
from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb

from .preprocessing import load_classes
from .constants import DATASETWRAPPERS, DATASETWRAPPERSBLURRED
from .imageProcessing import convert_numpy_to_PIL
from .io import savePIL

@DATASETS.register_module('BlurDataset')
def get_blurred_dataset(cfg, imgRoot, annfile, blurredSegments, segData, classes=None, saveDir='blurredImgs/', saveImgs=False,
                        blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None, **kwargs):
    """Creates a Dataset that inherits from the given CAM Cfg but blurres specified parts of the image.

    :param cfg: Config of the CAM Model from which the dataset is inferred.
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
    datasetType = cfg.data.test.type
    assert datasetType in DATASETWRAPPERS or datasetType in DATASETS, f'Datasettype {datasetType} not supported.'
    cfg.data.test['data_prefix'] = imgRoot
    cfg.data.test['ann_file'] = annfile
    # Differentiate between Wrapper because datasets from mmcls are added via Functions and therefore can not be get
    # from Registry and inherited from
    if datasetType in DATASETWRAPPERS:
        datasetType = DATASETWRAPPERS.get(datasetType)
    else:
        datasetType = DATASETS.get(datasetType)
    if classes is None:
        assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if classes not given.'
        classes = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)

    class BlurDataset(datasetType):

        def __init__(self, **kwargs) -> None:
            super(BlurDataset, self).__init__(**kwargs)
            self.segClasses = classes
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
                    elif isinstance(segment, int| np.integer):
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
            if saveImgs:
                print(f'Blurred Images will be saved to {saveDir}')

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
        

    params = copy.deepcopy(cfg.data.test)
    params.pop('type')

    return BlurDataset(**params)

def setup_config_blurred(cfg, imgRoot, annfile, blurredSegments, segData, segClasses=None, saveDir='blurredImgs/', saveImgs=False,
                        blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None, randomBlur=False, singleColor=None, **kwargs):
    """
    Sets the fields in cfg.data.test required for BlurDataset

    :param imgRoot: Path to the directory where samples are stored. Will overwrite data_prefix entry in cfg.
    :type imgRoot: str | Path
    :param annfile: Path to the annotation file. Will overwrite ann_file entry in cfg.
    :type annfile: str | Path
    :param blurredSegments: Index specifying what segements to blur. Either by name or indexed
    :type blurredSegments: str | int | list(str|int)
    :param segData: Path to numpy file containig the segmentation data
    :type segData: str | Path
    :param segClasses: List or array containing the classes of the segmentation. Used for ensuring blurredSegment is valid
                    If not specified classes will be loaded from segConfig and segCheckpoint
    :type segClasses: list(str) | np.ndarray(str)
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
    :param randomBlur: Blur random pixels with the same area as the blurred segments
    :type randomBlur: bool
    :param singleColor: Maked blurred segments of single color. Must be bgr channel order.
    :type singleColor: list(int) | np.ndarray(int) | None
    """
    print('Setting up configs for Blurred Dataset')
    datasetType = cfg.data.test.type
    assert datasetType in DATASETWRAPPERSBLURRED, f'Datasettype {datasetType} not supported. Supported are {",".join(DATASETWRAPPERSBLURRED)}'
    if datasetType in DATASETWRAPPERSBLURRED:
        datasetType = DATASETWRAPPERSBLURRED.get(datasetType)
    params = copy.deepcopy(cfg)
    params.data.test['data_prefix'] = imgRoot
    params.data.test['ann_file'] = annfile
    params.data.test['type'] = datasetType
    params.data.test['blurredSegments'] = blurredSegments
    params.data.test['segData'] = segData
    params.data.test['segClasses'] = segClasses
    params.data.test['saveDir'] = saveDir
    params.data.test['saveImgs'] = saveImgs
    params.data.test['blurKernel'] = blurKernel
    params.data.test['blurSigmaX'] = blurSigmaX
    params.data.test['segConfig'] = segConfig
    params.data.test['segCheckpoint'] = segCheckpoint
    params.data.test['randomBlur'] = randomBlur
    params.data.test['singleColor'] = singleColor
    
    return params

# @DATASETS.register_module('BlurredCompCars')
# class CompCarsBlurred(CompCars):
#     def __init__(self, ann_file, blurredSegments, segData, data_prefix, segClasses=None, saveDir='blurredImgs/', saveImgs=False,
#                         blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None, randomBlur=False, **kwargs):
#         super().__init__(data_prefix=data_prefix, ann_file=ann_file, **kwargs)
#         if segClasses is None:
#             assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if segClasses not given.'
#             self.segClasses = load_classes(segConfig = segConfig, segCheckpoint= segCheckpoint)
#         else:
#             self.segClasses = segClasses
        
#         self.prepipeline = Compose([{'type':'LoadImageFromFile'}])
#         self.postpipeline = Compose([step for step in kwargs['pipeline'] if step['type'] != 'LoadImageFromFile'])
#         if isinstance(blurredSegments, str):
#             assert blurredSegments in self.segClasses, f'{blurredSegments} not in segClasses: {",".join(self.segClasses)}'
#             self.blurredSegments = [self.segClasses.index(blurredSegments)]
#         elif isinstance(blurredSegments, int):
#             assert blurredSegments < len(self.segClasses), f'Can not blur segment no. {blurredSegments}. Only {len(self.segClasses)} present.'
#             self.blurredSegments = [blurredSegments]
#         elif isinstance(blurredSegments, list | np.ndarray):
#             self.blurredSegments = []
#             for segment in blurredSegments:
#                 if isinstance(segment, str):
#                     assert segment in self.segClasses, f'{segment} not in segClasses: {",".join(self.segClasses)}'
#                     self.blurredSegments.append(self.segClasses.index(segment))
#                 elif isinstance(segment, int | np.integer):
#                     assert segment < len(self.segClasses), f'Can not blur segment no. {segment}. Only {len(self.segClasses)} present.'
#                     self.blurredSegments.append(segment)
#                 else:
#                     raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
#             # Remove potential duplicates
#             self.blurredSegments = list(set(self.blurredSegments))
#         else:
#             raise ValueError(f'blurredSegments must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegments)}')
#         self.randomBlur = randomBlur
#         if self.randomBlur:
#             print('Blurring random parts of the image')
#         self.segData = segData
#         self.saveImgs = saveImgs
#         self.saveDir = saveDir
#         self.blurKernel = blurKernel
#         self.blurSigmaX = blurSigmaX
#         if saveImgs:
#             print(f'Blurred Images will be saved to {saveDir}')

#     def __getitem__(self, idx):
#         results = copy.deepcopy(self.data_infos[idx])
#         results = self.prepipeline(results)
#         ori_img = results['img']

#         with np.load(self.segData) as segData:
#             segmentation = segData[results['ori_filename']]

#         assert segmentation.shape == ori_img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {ori_img.shape[:-1]}'

#         mask = np.zeros_like(ori_img)
#         if self.randomBlur:
#                 blurredPixels = np.sum([np.prod((segmentation==blur).shape) for blur in self.blurredSegments])
#                 rawMask = np.zeros_like(segmentation)
#                 rawMask[:blurredPixels] = 1
#                 np.random.shuffle(rawMask)
#                 rawMask = rawMask.astype(bool)
#                 mask[rawMask==1,:] = np.array([255,255,255])
#         else:
#             for blur in self.blurredSegments:
#                 mask[segmentation==blur, :] = np.array([255,255,255])

#         blur_img = cv2.GaussianBlur(ori_img, self.blurKernel, self.blurSigmaX)

#         img = np.where(mask==np.array([255,255,255]), blur_img, ori_img)

#         results['img'] = img

#         if self.saveImgs:
#             pil = convert_numpy_to_PIL(results['img'])
#             savePIL(pil, fileName=results['ori_filename'], dir=self.saveDir, logSave=False)

#         return self.postpipeline(results)

# @DATASETS.register_module('BlurredCompCarsWeb')
# class BlurredCompCars(CompCarsWeb):
#     def __init__(self, ann_file, blurredSegments, segData, data_prefix, segClasses=None, saveDir='blurredImgs/', saveImgs=False,
#                         blurKernel=(33,33), blurSigmaX=0, segConfig=None, segCheckpoint=None, randomBlur=False, **kwargs):
#         super().__init__(data_prefix=data_prefix, ann_file=ann_file, **kwargs)
#         if segClasses is None:
#             assert segConfig is not None and segCheckpoint is not None, f'segConfig and segCheckpoint must be specified if segClasses not given.'
#             self.segClasses = load_classes(segConfig=segConfig, segCheckpoint=segCheckpoint)
        
#         self.prepipeline = Compose([{'type':'LoadImageFromFile'}])
#         self.postpipeline = Compose([step for step in kwargs['pipeline'] if step['type'] != 'LoadImageFromFile'])
#         if isinstance(blurredSegments, str):
#             assert blurredSegments in self.segClasses, f'{blurredSegments} not in segClasses: {",".join(self.segClasses)}'
#             self.blurredSegments = [self.segClasses.index(blurredSegments)]
#         elif isinstance(blurredSegments, int):
#             assert blurredSegments < len(self.segClasses), f'Can not blur segment no. {blurredSegments}. Only {len(self.segClasses)} present.'
#             self.blurredSegments = [blurredSegments]
#         elif isinstance(blurredSegments, list | np.ndarray):
#             self.blurredSegments = []
#             for segment in blurredSegments:
#                 if isinstance(segment, str):
#                     assert segment in self.segClasses, f'{segment} not in segClasses: {",".join(self.segClasses)}'
#                     self.blurredSegments.append(self.segClasses.index(segment))
#                 elif isinstance(segment, int| np.integer):
#                     assert segment < len(self.segClasses), f'Can not blur segment no. {segment}. Only {len(self.segClasses)} present.'
#                     self.blurredSegments.append(segment)
#                 else:
#                     raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
#             # Remove potential duplicates
#             self.blurredSegments = list(set(self.blurredSegments))
#         else:
#             raise ValueError(f'blurredSegments must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegments)}')
#         self.randomBlur = randomBlur
#         if self.randomBlur:
#             print('Blurring random parts of the image')
#         self.segData = segData
#         self.saveImgs = saveImgs
#         self.saveDir = saveDir
#         self.blurKernel = blurKernel
#         self.blurSigmaX = blurSigmaX
#         if saveImgs:
#             print(f'Blurred Images will be saved to {saveDir}')

#     def __getitem__(self, idx):
#         results = copy.deepcopy(self.data_infos[idx])
#         results = self.prepipeline(results)
#         ori_img = results['img']

#         if self.randomBlur:
#             mask = np.zeros(10000, dtype=int)
#             mask[:1000] = 1
#             np.random.shuffle(mask)
#             mask = mask.astype(bool)
#         else:
#             with np.load(self.segData) as segData:
#                 segmentation = segData[results['ori_filename']]

#             assert segmentation.shape == ori_img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {ori_img.shape[:-1]}'

#             mask = np.zeros_like(ori_img)
#             if self.randomBlur:
#                 blurredPixels = np.sum([np.prod((segmentation==blur).shape) for blur in self.blurredSegments])
#                 rawMask = np.zeros_like(segmentation)
#                 rawMask[:blurredPixels] = 1
#                 np.random.shuffle(rawMask)
#                 rawMask = rawMask.astype(bool)
#                 mask[rawMask==1,:] = np.array([255,255,255])
#             else:
#                 for blur in self.blurredSegments:
#                     mask[segmentation==blur, :] = np.array([255,255,255])

#             blur_img = cv2.GaussianBlur(ori_img, self.blurKernel, self.blurSigmaX)

#             img = np.where(mask==np.array([255,255,255]), blur_img, ori_img)

#         results['img'] = img

#         if self.saveImgs:
#             pil = convert_numpy_to_PIL(results['img'])
#             savePIL(pil, fileName=results['ori_filename'], dir=self.saveDir, logSave=False)

#         return self.postpipeline(results)