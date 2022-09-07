import numpy as np
import cv2
import os.path as osp

from mmcls.datasets.builder import PIPELINES

from .io import savePIL
from .imageProcessing import convert_numpy_to_PIL

@PIPELINES.register_module()
class BlurSegments(object):
    """
    Blurs the specified segments out using gaussian blurring from cv2

    """

    def __init__(self, blurredSegments, 
                segData, segCategories, 
                blurKernel=(33,33), blurSigmaX=0, 
                saveImgs=False, saveDir='blurredImgs/',
                randomBlur=False):
        self.segCategories = segCategories
        self.segData = segData
        self.blurKernel = blurKernel
        self.blurSigmaX = blurSigmaX
        self.saveImgs = saveImgs
        self.saveDir = saveDir
        self.randomBlur = randomBlur
        if randomBlur:
            print('Blurring random parts of the image')
        if saveImgs:
            print(f'Blurred Images will be saved to {saveDir}')
        if isinstance(blurredSegments, str):
            assert blurredSegments in segCategories, f'{blurredSegments} not in segCategories: {",".join(segCategories)}'
            self.blurredSegments = [segCategories.index(blurredSegments)]
        elif isinstance(blurredSegments, int):
            assert blurredSegments < len(segCategories), f'Can not blur segment no. {blurredSegments}. Only {len(segCategories)} present.'
            self.blurredSegments = [blurredSegments]
        elif isinstance(blurredSegments, list | np.ndarray):
            self.blurredSegments = []
            for segment in blurredSegments:
                if isinstance(segment, str):
                    assert segment in segCategories, f'{segment} not in segCategories: {",".join(segCategories)}'
                    self.blurredSegments.append(segCategories.index(segment))
                elif isinstance(segment, int| np.integer):
                    assert segment < len(segCategories), f'Can not blur segment no. {segment}. Only {len(segCategories)} present.'
                    self.blurredSegments.append(segment)
                else:
                    raise ValueError(f'Encounter invalid type {type(segment)} for {segment}')
            # Remove potential duplicates
            self.blurredSegments = list(set(self.blurredSegments))
        else:
            raise ValueError(f'blurredSegments must be of type "str" or "int" or "list" or "np.ndarray" not {type(blurredSegments)}')
    
    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(blurredSegments={",".join(self.blurredSegments)}'
        repr_str += f', segCategories={",".join(self.segCategories)}'
        repr_str += f', blurKernel={self.blurKernel}'
        repr_str += f', blurSigmaX={self.blurSigmaX}'
        repr_str += f', saveImgs={self.saveImgs})'
        repr_str += f', saveDir={self.saveDir})'
        return repr_str

    def __call__(self, results):
        img = results['img']

        with np.load(self.segData) as segData:
            segmentation = segData[osp.basename(results['filename'])]

        assert segmentation.shape == img.shape[:-1], f'Shape of Segmentation {segmentation.shape} does not match image shape {img.shape[:-1]}'

        mask = np.zeros_like(img)
        if self.randomBlur:
            blurredPixels = np.sum([np.prod((segmentation==blur).shape) for blur in self.blurredSegments])
            rawMask = np.zeros_like(segmentation)
            rawMask[:blurredPixels] = 1
            np.random.shuffle(rawMask)
            rawMask = rawMask.astype(bool)
            mask[rawMask==1,:] = np.array([255,255,255])
        else:
            for blur in self.blurredSegments:
                mask[segmentation==blur, :] = np.array([255,255,255])

        blur_img = cv2.GaussianBlur(img, self.blurKernel, self.blurSigmaX)

        img = np.where(mask==np.array([255,255,255]), blur_img, img)

        results['img'] = img

        if self.saveImgs:
            pil = convert_numpy_to_PIL(results['img'])
            savePIL(pil, fileName=osp.basename(results['filename']), dir=self.saveDir, logSave=False)

        return results