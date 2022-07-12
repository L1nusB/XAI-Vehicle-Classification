import mmcv
import os
import numpy as np
import warnings

from mmcv import Config


from .. import new_gen_seg, generate_cams


def prepareSegmentation(imgRoot='', segConfig='', segCheckpoint='', segData=None, segDevice='cuda', annfile='', dataClasses=[], **kwargs):
    assert (segConfig and segCheckpoint) or (segData is not None), 'Either config + checkpoint or data must be provided for segmentation.'
    segmentationMasks = None
    segmentationImgData = None
    if segData is not None:
        segmentationMasks = np.copy(segData)
    else:
        if 'imgName' in kwargs:
            assert os.path.isfile(segConfig), f'segConfig:{segConfig} does not lead to a file'
            assert os.path.isfile(segCheckpoint), f'segCheckpoint:{segCheckpoint} does not lead to a file'
            temp_filePath = 'temp_ann.txt'
            with open(temp_filePath,'w') as tmp:
                tmp.write(kwargs['imgName'])
            segmentationMasks, segmentationImgData = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--ann-file',os.path.abspath(temp_filePath), '-r','--types', 'masks', 'images','--device',segDevice])
            os.remove(temp_filePath)
        else:
            classParamAddition=[]
            if len(dataClasses)>0:
                classParamAddition=['--classes'] + dataClasses
                annfileParamAddition=[]
                if annfile:
                    annfileParamAddition=['--ann-file', os.path.abspath(annfile)]
            segmentationMasks, segmentationImgData = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'-r','--types', 'masks', 'images','--device',segDevice] + classParamAddition + annfileParamAddition)
    return segmentationMasks, segmentationImgData

def prepareImg(imgPath='', imgData=None, **kwargs):
    if imgData is not None:
        assert isinstance(imgData, np.ndarray), f'Expected type np.ndarray got {type(imgData)}'
        assert len(imgData.shape) == 3 and (imgData.shape[-1]==1 or imgData.shape[-1]==3), f'expected Shape (_,_,1) or (_,_,3) for imgData but got {imgData.shape}'
        return np.copy(imgData)
    if os.path.isfile(imgPath):
        return mmcv.imread(imgPath)
    else:
        raise ValueError('imgPath must be specified through imgRoot and imgName if no imgData is provided.')

def prepareCams(imgPath='', camConfig='', camCheckpoint='', camData=None, camDevice='cpu', method='gradcam', dataClasses=[], annfile='', **kwargs):
    assert (camConfig and camCheckpoint) or (camData is not None), 'Either config + checkpoint or data must be provided for CAM generation.'
    if camData is not None:
        return np.copy(camData)
    if os.path.isfile(imgPath):
        assert os.path.isfile(camConfig), f'camConfig:{camConfig} does not lead to a file'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint:{camCheckpoint} does not lead to a file'
        camData = generate_cams.main([imgPath, camConfig, camCheckpoint,'--device',camDevice, '--method', method])
    elif os.path.isdir(imgPath):
        assert os.path.isfile(camConfig), f'camConfig:{camConfig} does not lead to a file'
        assert os.path.isfile(camCheckpoint), f'camCheckpoint:{camCheckpoint} does not lead to a file'
        classParamAddition=[]
        if len(dataClasses)>0:
            classParamAddition=['--classes'] + dataClasses
        annfileParamAddition=[]
        if annfile:
            annfileParamAddition=['--ann-file', annfile]
        camData = generate_cams.main([imgPath, camConfig, camCheckpoint,'--device',camDevice, '--method', method] + classParamAddition + annfileParamAddition)
    else:
        raise ValueError('imgName must be a file or directory if no camData is provided.')
    return camData


def get_pipeline_cfg(pipelineCfg=None, **kwargs):
    cfg = None
    if pipelineCfg and os.path.isfile(pipelineCfg):
        cfg = Config.fromfile(pipelineCfg)
    elif 'camConfig' in kwargs:
        if 'segData' in kwargs:
            warnings.warn('No pipeline is applied since segData is provided. If pipeline should be applied specify '
            'by pipelineCfg parameter.')
        else:
            cfg = Config.fromfile(kwargs['camConfig'])
    else:
        if not 'segData' in kwargs:
            warnings.warn('No Pipeline specified and segData parameter not given. Shape must match automatically.')
        else:
            warnings.warn('No pipeline is applied since segData is provided. If pipeline should be applied specify '
            'by pipelineCfg parameter.')
    return cfg
    

def prepareInput(prepImg=True, prepSeg=True, prepCam=True, **kwargs):
    """
    Prepares the input data for the given parameter options for sourceImg, Segmentation and Cams.

    Potential kwargs are:
    imgRoot: Root to img Folder
    imgName: Name of the single image file
    imgData: Preloaded img Data of source image
    segConfig: Path to Config for segmentation
    segCheckpoint: Path to Checkpoint for segmentation
    segData: Already loaded object containing segmentation
    segDevice: (default cuda) Device used for segmentation
    annfile: Path to annfile to be used when generating segmentations
    camConfig: Path to Config for CAMs
    camCheckpoint: Path to Checkpoint for CAMs
    camData: Already loaded object containing CAMs
    camDevice: (default cpu) Device used for CAMs
    method: (default gradcam) Method by which the CAMs are generated
    dataClasses: (default []) List classes for which data will be generated. [] results in everything.

    :return List of specified data objects. [sourceImg, segmentationsMasks, segmentationsImages, cam]
    """
    if 'imgName' in kwargs:
        imgPath = os.path.join(kwargs['imgRoot'], kwargs['imgName'])
        kwargs['imgPath'] = imgPath
        assert os.path.isfile(imgPath), f'Specified Path does not yield valid file:{imgPath}'
    else:
        # Set imgPath to imgRoot unless imgPath is already explicity given.
        kwargs['imgPath'] = kwargs['imgPath'] if 'imgPath' in kwargs else kwargs['imgRoot']
    results = []
    if prepImg:
        imgData = prepareImg(**kwargs)
        results.append(imgData)
    if prepSeg:
        segmentationMasks, segmentationsImgData = prepareSegmentation(**kwargs)
        if 'imgName' in kwargs:
            results.append(segmentationMasks[kwargs['imgName']])
            results.append(segmentationsImgData[kwargs['imgName']])
        else:
            results.append(segmentationMasks)
            results.append(segmentationsImgData)
    if prepCam:
        camData = prepareCams(**kwargs)
        if 'imgName' in kwargs:
            results.append(camData[kwargs['imgName']])
        else:
            results.append(camData)
    return results