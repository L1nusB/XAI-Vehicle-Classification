import mmcv
import os
import numpy as np

from .. import new_gen_seg, generate_cams


def prepareSegmentation(imgRoot='', segConfig='', segCheckpoint='', segImgData=None, segDevice='cuda', annfile='', **kwargs):
    assert (segConfig and segCheckpoint) or segImgData, 'Either config + checkpoint or data must be provided for segmentation.'
    if segImgData:
        return np.copy(segImgData)
    if 'imgName' in kwargs:
        assert os.path.isfile(segConfig), f'segConfig:{segConfig} does not lead to a file'
        assert os.path.isfile(segCheckpoint), f'segCheckpoint:{segCheckpoint} does not lead to a file'
        temp_filePath = 'temp_ann.txt'
        with open(temp_filePath,'w') as tmp:
            tmp.write(kwargs['imgName'])
        segmentationMasks, segmentationImgData = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--ann-file',os.path.abspath(temp_filePath), '-r','--types', 'masks', 'images','--device',segDevice])
        os.remove(temp_filePath)
    else:
        segmentationMasks, segmentationImgData = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--ann-file',os.path.abspath(annfile), '-r','--types', 'masks', 'images','--device',segDevice])
    return segmentationMasks, segmentationImgData

def prepareImg(imgPath='', imgData=None, **kwargs):
    if imgData:
        assert isinstance(imgData, np.ndarray), f'Expected type np.ndarray got {type(imgData)}'
        assert len(imgData.shape) == 3 and (imgData.shape[-1]==1 or imgData.shape[-1]==3), f'expected Shape (_,_,1) or (_,_,3) for imgData but got {imgData.shape}'
        return np.copy(imgData)
    if os.path.isfile(imgPath):
        return mmcv.imread(imgPath)
    else:
        raise ValueError('imgPath must be specified through imgRoot and imgName if no imgData is provided.')

def prepareCams(imgPath='', camConfig='', camCheckpoint='', camImgData=None, camDevice='cpu', method='gradcam', **kwargs):
    assert (camConfig and camCheckpoint) or camImgData, 'Either config + checkpoint or data must be provided for CAM generation.'
    if camImgData:
        return np.copy(camImgData)
    if os.path.isfile(imgPath)
        assert os.path.isfile(camConfig), f'camConfig:{camConfig} does not lead to a file'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint:{camCheckpoint} does not lead to a file'
        camData = generate_cams.main([imgPath, camConfig, camCheckpoint,'--device',camDevice])
    else:
        raise ValueError('imgName must be specified if no camData is provided.')
    return camData
    
    

def prepareInput(prepImg=False, prepSeg=False, prepCam=False, **kwargs):
    if 'imgName' in kwargs:
        kwargs['imgPath'] = os.path.join(kwargs['imgRoot'], kwargs['imgName'])
    results = []
    if prepImg:
        imgData = prepImg(**kwargs)
        results.append(imgData)
    if prepSeg:
        segmentationMasks, segmentationsImgData = prepareSegmentation(**kwargs)
        results.append(segmentationMasks)
        results.append(segmentationsImgData)
    if prepCam:
        camData = prepareCams(**kwargs)
        results.append(camData)
    return results