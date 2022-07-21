import os.path as osp
import shutil
import mmcv
from mmcv import Config
from pathlib import Path
import os
import numpy as np

from .constants import DATASETSDATAPREFIX

def get_dir_and_file_path(path, defaultName='results.npz', defaultDir='./output/'):
    directory = defaultDir
    fileName = defaultName
    if osp.isdir(path):
        # Path is a directory
        # Just use default Name
        directory = path
    elif osp.dirname(path):
        # Base Directory Specified
        # Override default Dir
        directory = osp.dirname(path)
    # No else since otherwise default Dir is used

    if osp.basename(path):
        fileName = osp.basename(path)
        if osp.basename(path)[:-4] != '.npz':
            # Change file extension to .npz
            fileName = fileName + ".npz"
    # Again no else needed since default is used otherwise
    return directory, fileName

def get_samples(annfile=None, imgRoot=None, fc=None, dataClasses=[], **kwargs):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if annfile:
        if len(dataClasses)>0:
            samples = [i for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk'))]
    else:
        if dataClasses:
            samples = [i for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)]
    return samples

def get_sample_count(args, fc=None, dataClasses=[]):
    return len(get_samples(annfile=args.ann_file, imgRoot=osp.join(args.root, args.imgDir), fc=fc, dataClasses=dataClasses))

def generate_split_files(sample_iterator, batch_count, batch_size, work_dir, dataClasses=[], fileprefix='results'):
    sample_list = list(sample_iterator)
    if len(dataClasses)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in dataClasses)]
    if batch_size == -1:
        with open(osp.join(work_dir, f'{fileprefix}_{0}.txt'),'w') as f:
            f.write('\n'.join(sample_list))
        return
    for i in range(batch_count):
        with open(osp.join(work_dir, f'{fileprefix}_{i}.txt'),'w') as f:
            f.write('\n'.join(sample_list[i*batch_size:(i+1)*batch_size]))

def saveResults(savePath, defaultName='generated_result.npz', **results):
    print(f'Saving results in: {savePath}')
    Path(os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(savePath) or not os.path.basename(savePath):
        print(f'No filename specified. Generating file {defaultName} in directory {savePath}')
        path = os.path.join(savePath,defaultName)
    else: 
        if os.path.basename(savePath)[-4:] == ".npz":
            path = savePath
        else:
            path = os.path.join(os.path.dirname(savePath), os.path.basename(savePath)+".npz")
    
    np.savez(path,**results)

def copyFile(srcPath, dstPath):
    """
    Copys the file at srcPath into dstPath. 
    Creates folders as necessery for dstPath.
    If dstPath has no basename/is a dictionary the name of srcPath will be used.
    """
    assert osp.isfile(srcPath),f'No such file {srcPath}'
    if osp.basename(dstPath).split('.')[0] != '.txt':
        # Ensure dstPath leads to .txt file
        outpath = osp.join(osp.dirname(dstPath), osp.basename(dstPath).split('.')[0]+'.txt')
    else:
        outpath = dstPath
    if osp.isdir(dstPath) or osp.basename(dstPath) is None:
        print(f'No Basename for dstPath. Using srcPath basename.')
        outpath = dstPath + osp.basename(srcPath)
    Path(os.path.dirname(outpath)).mkdir(parents=True, exist_ok=True)
    print(f'Copying file from {srcPath} to {outpath}')
    shutil.copy(srcPath, outpath)

def writeArrayToFile(path, arr, seperator=','):
    """
    Writes the given elements of the array into a .txt file at the given Path.
    If the file does not exist it will be created as well as necessary directories.
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    if osp.basename(path).split('.')[0] != '.txt':
        # Ensure path leads to .txt file
        filePath = osp.join(osp.dirname(path), osp.basename(path).split('.')[0]+'.txt')
    else:
        filePath = path
    print(f'Writing data to file at {path}')
    with open(filePath, 'w') as f:
        f.write(seperator.join(arr))


def saveFigure(savePath, figure, defaultName='figure.jpg'):
    """
    Saves the given figure under the specified Path.
    If the Basename of savePath is not a directory its base will be used.
    If savePath is a directory the defaultName will be added as the filename into that directory.
    Otherwise it is ensured that it is .jpg or .png and if necessary a .jpg extension is added.
    """
    Path(os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    base = os.path.dirname(savePath)
    if not os.path.isdir(savePath):
        print(f'Output path is not a directory. Using base directory: {os.path.dirname(savePath)}.')
        if os.path.basename(savePath):
            if os.path.basename(savePath)[-4:] == ".jpg" or os.path.basename(savePath)[-4:] == ".png":
                outPath = savePath
            else:
                outPath = savePath + ".jpg"
        else:
            outPath = os.path.join(base, defaultName)
    else:
        outPath = os.path.join(savePath, defaultName)
            
    Path(os.path.dirname(outPath)).mkdir(parents=True, exist_ok=True)
    print(f'Saving images to: {outPath}')
    figure.savefig(outPath)


def saveCAMs(args, cams):
    print("Save generated CAMs to " + args.save_path)
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(args.save_path) or not os.path.basename(args.save_path):
        print(f'No filename specified. Generating file "generated_cams.npz" in directory {args.save_path}')
        path = os.path.join(args.save_path,"generated_cams.npz")
    else: 
        if os.path.basename(args.save_path)[-4:] == ".npz":
            path = args.save_path
        else:
            path = os.path.join(os.path.dirname(args.save_path), os.path.basename(args.save_path)+".npz")
    print(f'Output path to CAM file:{path}')
    
    np.savez(path,**cams)

def get_save_figure_name(statType,dataClasses=[], annfile='', method='gradcam', additional = '', **kwargs):
    """
    Determines the name under which the figure and potential other files will be saved.
    Determines if another file must be saved.
    Filenames follow the strucutre:
    statType_selectionCriterion_camMethod_camDataset_camModel_segModel_segDataset_additional_dd_MM_yyyy

    :param statType: (Single, Full, Multiple) What data went into the statistic. Will be multiple if dataClasses or annfile are given
    :param dataClasses: List of Classes that was sampled from. Will be saved
    :param annfile: Annotation file containing all samples used. Will be saved
    :param method: Method used to generate the CAMs
    :param additional: Additional statistics information see Excel Sheet.

    :return: Tuple (figure_name, saveDataClasses, saveAnnfile)
    """
    from datetime import date
    selectionCriterion = ''
    saveAnnfile = False
    saveDataClasses = False
    if len(dataClasses)>0:
        selectionCriterion='classes'
        saveDataClasses = True
        statType = 'Multiple' if statType.lower() != 'single' else statType
    if annfile != '':
        statType = 'Multiple' if statType.lower() != 'single' else statType
        selectionCriterion=selectionCriterion+"+annfile" if selectionCriterion != '' else 'annfile'
        saveAnnfile = True


    if 'camData' in kwargs:
        camMethod = 'CAM-Predefined'
        camDataset = 'CAM-Predefined'
        camModel = 'CAM-Predefined'
    else:
        # Now camConfig and camCheckpoint must be in kwargs
        camMethod = method

        cfg = Config.fromfile(kwargs['camConfig'])
        camDataset = cfg.data.train.type # Load the general type. Used if not more detailed found
        # Check if a detailed Dataset can be identified.
        for k,v in DATASETSDATAPREFIX.items():
            if v.lower() in cfg.data.train.data_prefix.lower():
                camDataset = k
                break
        camModel = cfg.model.backbone.type

    if 'segData' in kwargs:
        segModel = 'SEG-Predefined'
        segDataset = 'SEG-Predefined'
    else:
        cfg = Config.fromfile(kwargs['segConfig'])
        segDataset = cfg.data.train.type # Load the general type. Used if not more detailed found
        # Check if a detailed Dataset can be identified.
        for k,v in DATASETSDATAPREFIX.items():
            if v.lower() in cfg.data.train.data_root.lower():
                segDataset = k
                break
        segModel = cfg.model.backbone.type

    dateStr = date.today().strftime("%d_%m_%Y")

    components = [statType, selectionCriterion, camMethod, camDataset, camModel, segModel, segDataset, additional, dateStr]

    figure_name = "_".join([component for component in components if component != '']) + '.png'

    return figure_name, saveDataClasses, saveAnnfile