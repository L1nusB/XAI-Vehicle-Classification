from wsgiref.util import setup_testing_defaults
import numpy as np
import cv2

from ..ImageDataset import ImageDataset

class BlurDataset(ImageDataset):

    def __init__(self, imgRoot,classes, blurredSegment, segData, saveDir='./', saveImgs=False, 
                annfile=None, imgNames=None, dataClasses=[], pipeline=None, get_gt=False,
                blurKernel=(33,33), blurSigmaX=0):
        super(BlurDataset, self).__init__(imgRoot, annfile=None, 
                                        imgNames=None, dataClasses=[], 
                                        pipeline=None, get_gt=False)
        self.classes = classes
        if isinstance(blurredSegment, str):
            assert blurredSegment in classes, f'{blurredSegment} not in classes: {",".join(classes)}'
            self.blurredSegment = [classes.index(blurredSegment)]
        elif isinstance(blurredSegment, int):
            assert blurredSegment < len(classes), f'Can not blur segment no. {blurredSegment}. Only {len(classes)} present.'
            self.blurredSegment = [blurredSegment]
        elif isinstance(blurredSegment, list | np.ndarray):
            self.blurredSegment = []
            for segment in blurredSegment:
                if isinstance(segment, str):
                    assert segment in classes, f'{segment} not in classes: {",".join(classes)}'
                    self.blurredSegment.append(classes.index(segment))
                elif isinstance(segment, int):
                    assert segment < len(classes), f'Can not blur segment no. {segment}. Only {len(classes)} present.'
                    self.blurredSegment.append(segment)
                else:
                    raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
            # Remove potential duplicates
            self.blurredSegment = list(set(self.blurredSegment))
        else:
            raise ValueError(f'blurredSegment must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegment)}')
        self.segData = np.load(segData)
        self.saveImgs = saveImgs
        self.saveDir = saveDir
        self.blurKernel = blurKernel
        self.blurSigmaX = blurSigmaX

    def __getitem__(self, idx):
        item = super().__getitem__(idx) 
        ori_img = item['img']

        segmentation = self.segData[item['name']]

        assert segmentation.shape == ori_img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {ori_img.shape[:-1]}'

        mask = np.zeros_like(ori_img)
        for blur in self.blurredSegment:
            mask[segmentation==blur, :] = np.array([255,255,255])

        blur_img = cv2.GaussianBlur(ori_img, self.blurKernel, self.blurSigmaX)

        img = np.where(mask==np.array([255,255,255]), blur_img, ori_img)

        item['img'] = img

        return item