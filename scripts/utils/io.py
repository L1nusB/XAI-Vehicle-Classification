import os.path as osp
import shutil
import mmcv
from mmcv import Config
from pathlib import Path
import os
import numpy as np

from .constants import DATASETSDATAPREFIX, RESULTS_PATH_ANN,RESULTS_PATH, RESULTS_PATH_DATACLASS

def get_dir_and_file_path(path, defaultName='results.npz', defaultDir='./output/', removeFileExtensions=False):
    """Splits the given path into directory and basename covering the case, that the
    path is a directory without a trailing slash. Missing areas in the path will be 
    filled with the given defaults.
    If specified possible file extensions will be removed by removing everything after the last found . in the filename.

    :return: (directory, filename) tuple of paths.
    """
    directory = defaultDir
    fileName = defaultName
    if osp.isdir(path):
        # Path is a directory
        # Just use default Name
        directory = path
    else:
        if osp.dirname(path):
            # Base Directory Specified
            # Override default Dir
            directory = osp.dirname(path)
        if osp.basename(path):
            fileName = osp.basename(path)
            if removeFileExtensions:
                # Ensure dot in fileName otherwise result will be empty
                if '.' in fileName:
                    fileName = ".".join(fileName.split(".")[:-1])
    return directory, fileName

def get_samples(annfile=None, imgRoot=None, fc=None, dataClasses=[], **kwargs):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if isinstance(dataClasses, str):
        dataClasses = [dataClasses]
    if annfile:
        if len(dataClasses)>0:
            samples = [i.strip().rsplit(" ",1)[0] for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i.strip().rsplit(" ",1)[0] for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk'))]
    else:
        if dataClasses:
            samples = [i.strip().rsplit(" ",1)[0] for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i.strip().rsplit(" ",1)[0] for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)]
    return samples

def get_sample_count(args, fc=None, dataClasses=[]):
    return len(get_samples(annfile=args.ann_file, imgRoot=osp.join(args.root, args.imgDir), fc=fc, dataClasses=dataClasses))

# def generate_split_files(sample_iterator, batch_count, batch_size, work_dir, dataClasses=[]):
#     sample_list = list(sample_iterator)
#     if len(dataClasses)>0:
#         sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in dataClasses)]
#     if batch_size == -1:
#         with open(osp.join(work_dir, f'split_{0}.txt'),'w') as f:
#             f.write('\n'.join(sample_list))
#         return
#     for i in range(batch_count):
#         with open(osp.join(work_dir, f'split_{i}.txt'),'w') as f:
#             f.write('\n'.join(sample_list[i*batch_size:(i+1)*batch_size]))

def generate_split_file(sample_iterator, work_dir, dataClasses=[], fileprefix='resultsSeg'):
    """Generates a .txt file containing all items of the given sample_iterator
    that will be used to tell the dataset which samples to select.
    Via dataClasses one can restrict the samples to ones based on those classes.

    :param sample_iterator: Iterable containing all sample names/paths
    :param work_dir: Path to the directory where the file will be saved to
    :type work_dir: str
    :param dataClasses: List of classes restricting the samples unless empty, defaults to []
    :type dataClasses: List[str], optional
    """
    sample_list = list(sample_iterator)
    if len(dataClasses)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in dataClasses)]
    with open(osp.join(work_dir, f'{fileprefix}.txt'),'w') as f:
        f.write('\n'.join(sample_list))

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

def savePIL(img, fileName, dir='./', logSave=True):
    """Saves the given PIL Image under the fileName in the specified diretory

    :param img: Image to be saved
    :type img: PIL Image
    :param dir: Directory where to save the image to
    :type dir: str | Path
    :param fileName: Name of the file. If no extension a .png will be created.
    :type fileName: str | Path
    :param logSave: (default True)Log a message where the file is saved.
    :type logSave: bool
    """
    Path(dir).mkdir(parents=True, exist_ok=True)
    if fileName[-4:] != '.jpg' and fileName[-4:] != '.png':
        fileName = fileName + '.png'
    absPath = os.path.join(dir, fileName)
    if logSave:
        print(f'Saving image to: {absPath}')
    img.save(absPath)


def get_save_figure_name(statType='' ,dataClasses=[], annfile='', method='gradcam', additional = '', **kwargs):
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
    
    # Set to Full if no dataClasses, annfile or manually specified statType 'single'
    if len(dataClasses) == 0 and annfile == '' and statType.lower() != 'single':
        statType = 'Full'


    if 'camData' in kwargs:
        camMethod = 'CAM-Predefined'
        camDataset = 'CAM-Predefined'
        camModel = 'CAM-Predefined'
        # If camData is a path use that name for Method
        if isinstance(kwargs['camData'], str | os.PathLike):
            camMethod = osp.basename(kwargs['camData']).split(".")[0]
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
        # If segData is a path use that name for Dataset
        if isinstance(kwargs['segData'], str | os.PathLike):
            segDataset = osp.basename(kwargs['segData']).split(".")[0]
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

    figure_name = "_".join([component for component in components if component != '']) + '.jpg'

    return figure_name, saveDataClasses, saveAnnfile

def save_result_figure_data(figure, save_dir="", path_intermediate="", fileNamePrefix="", default_Path=RESULTS_PATH, default_Ann_Path=RESULTS_PATH_ANN, default_DataClasses_Path=RESULTS_PATH_DATACLASS, **kwargs):
    """Saves the given figure and potential corresponding annfile and dataClasses File in the given base saveDir.
    An optional intermediate can be specified which will create a directory after the saveDir in which results will be saved to.

    :param figure: Figure to save
    :param save_dir: Path to base directory to which results will be saved to.
    :param path_intermediate: Optional intermediate that will add a directory after save_dir
    :param fileNamePrefix: Optional prefix that will be added in front of the fileName
    :param default_Path: Default Path where the figure will be saved to
    :param default_Ann_Path: Default Path where the annfiles will be saved to
    :param default_DataClasses_Path: Default Path where the dataClasses file will be saved to
    """

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(**kwargs)
    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    if save_dir:
        results_path = os.path.join(save_dir, path_intermediate)
        results_path_ann = os.path.join(save_dir, path_intermediate, 'annfiles')
        results_path_dataclasses = os.path.join(save_dir, path_intermediate, 'dataClasses')
    else:
        results_path = os.path.join(RESULTS_PATH, path_intermediate)
        results_path_ann = os.path.join(RESULTS_PATH_ANN, path_intermediate)
        results_path_dataclasses = os.path.join(RESULTS_PATH_DATACLASS, path_intermediate)

    saveFigure(savePath=os.path.join(results_path, figure_name), figure=figure)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(results_path_ann, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(results_path_dataclasses, figure_name), kwargs['dataClasses'])

def generate_filtered_annfile(annfile, imgNames, fileName='annfile_filtered.txt'):
    """Creates an updated annotation file under the given name that is a filtered version of the specified
    annotation file by the given image Names.
    The file be created in the same directory as the original file and the filename will be returned

    :param annfile: Path to the original annotation file.
    :type annfile: str | Path
    :param imgNames: Collection of the image Names that should be kept after filtering.
    :type imgNames: List | np.ndarray
    :param fileName: Name under which the filtered file will be saved to.
    :type fileName: str | Path
    """
    assert os.path.isfile(annfile), f'No such file {annfile}'
    filteredEntries = []
    with open(annfile, encoding='utf-8') as f:
        for entry in f.readlines():
            if entry.strip().rsplit(' ')[0] in imgNames:
                filteredEntries.append(entry)

    if fileName[-4:] != '.txt':
        fileName = fileName + '.txt'

    with open(os.path.join(os.path.dirname(annfile), fileName), mode='w', encoding='utf-8') as f:
        f.write("".join(filteredEntries))
    
    print(f"Created filtered annotation file at {os.path.join(os.path.dirname(annfile), fileName)}")
    return os.path.join(os.path.dirname(annfile), fileName)