from fileinput import filename
import os.path as osp
import shutil
import mmcv
from mmcv import Config
from pathlib import Path
import os
import numpy as np
import warnings
import pandas
import json

from .constants import DATASETSDATAPREFIX, RESULTS_PATH_ANN,RESULTS_PATH, RESULTS_PATH_DATACLASS, FIGUREFORMATS

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

def get_samples(annfile=None, imgRoot=None, fc=None, dataClasses=[], splitSamples=True, **kwargs):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if isinstance(dataClasses, str):
        dataClasses = [dataClasses]
    if annfile:
        if len(dataClasses)>0:
            samples = [i.strip() for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i.strip() for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk'))]
    else:
        if dataClasses:
            samples = [i.strip() for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i.strip() for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)]
    if splitSamples:
        samples = [i.rsplit(" ",1)[0] for i in samples]
    return samples

def get_sample_count(args, fc=None, dataClasses=[]):
    return len(get_samples(annfile=args.ann_file, imgRoot=osp.join(args.root, args.imgDir), fc=fc, dataClasses=dataClasses))

def generate_ann_file(sample_iterator, work_dir, dataClasses=[], fileprefix='results'):
    """Generates a .txt file containing all items of the given sample_iterator
    that will be used to tell the dataset which samples to select.
    Via dataClasses one can restrict the samples to ones based on those classes.

    :param sample_iterator: Iterable containing all sample names/paths
    :param work_dir: Path to the directory where the file will be saved to
    :type work_dir: str
    :param dataClasses: List of classes restricting the samples unless empty, defaults to []
    :type dataClasses: List[str], optional
    :param fileprefix: Name of the file that will be created. File extension is NOT required.
    :type fileprefix: str

    :return filePath of the created file.
    """
    sample_list = list(sample_iterator)
    if len(dataClasses)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in dataClasses)]
    # Remove .txt file extension if given.
    if fileprefix[-4:] == '.txt':
        fileprefix = fileprefix[-4:]
    filePath = osp.join(work_dir, f'{fileprefix}.txt')
    print(f'Creating annotation file at {filePath}')
    with open(filePath,'w') as f:
        f.write('\n'.join(sample_list))
    return filePath

def generate_split_file(sample_iterator, work_dir, dataClasses=[], fileprefix='resultsSeg'):
    """
    This function directly calles generate_ann_files
    Generates a .txt file containing all items of the given sample_iterator
    that will be used to tell the dataset which samples to select.
    Via dataClasses one can restrict the samples to ones based on those classes.

    :param sample_iterator: Iterable containing all sample names/paths
    :param work_dir: Path to the directory where the file will be saved to
    :type work_dir: str
    :param dataClasses: List of classes restricting the samples unless empty, defaults to []
    :type dataClasses: List[str], optional
    :param fileprefix: Name of the file that will be created. File extension is NOT required.
    :type fileprefix: str

    :return filePath of the created file.
    """
    warnings.warn("Using generate_split_files will redirect the call to generate_ann_file. Consider directly calling that function")
    return generate_ann_file(sample_iterator, work_dir, dataClasses, fileprefix)


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


def saveFigure(savePath, figure, defaultName='figure.jpg', fileExtension='.jpg'):
    """
    Saves the given figure under the specified Path.
    If the Basename of savePath is not a directory its base will be used.
    If savePath is a directory the defaultName will be added as the filename into that directory.
    Otherwise it is ensured that it is .jpg or .png or .svg or .pdf and if necessary a extension is added.
    """
    Path(os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    base = os.path.dirname(savePath)
    # Ensure format works if no dot is specified.
    fileExtension = "." + fileExtension.split(".")[-1]
    assert fileExtension.split(".")[-1] in FIGUREFORMATS, f'Given fileExtension {fileExtension} is not supported.'
    if not os.path.isdir(savePath):
        print(f'Output path is not a directory. Using base directory: {os.path.dirname(savePath)}.')
        if os.path.basename(savePath):
            if os.path.basename(savePath).split(".")[-1] in FIGUREFORMATS:
                outPath = savePath
            else:
                outPath = savePath + fileExtension
        else:
            outPath = os.path.join(base, defaultName)
    else:
        outPath = os.path.join(savePath, defaultName)
            
    Path(os.path.dirname(outPath)).mkdir(parents=True, exist_ok=True)
    print(f'Saving images to: {outPath}')
    figure.savefig(outPath, bbox_inches='tight')

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


def get_save_figure_name(statType='' ,dataClasses=[], annfile='', method='gradcam', additional = '', fileExtension='.jpg', **kwargs):
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
    :param fileExtension: FileExtension added in the end to figure_name (default .jpg)

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

    figure_name = "_".join([component for component in components if component != '']) + fileExtension

    return figure_name, saveDataClasses, saveAnnfile

def save_result_figure_data(figure, save_dir="", path_intermediate="", fileNamePrefix="", default_Path=RESULTS_PATH, saveAdditional=True,
                            default_Ann_Path=RESULTS_PATH_ANN, default_DataClasses_Path=RESULTS_PATH_DATACLASS, fileName='',
                            fileExtension='.jpg', **kwargs):
    """Saves the given figure and potential corresponding annfile and dataClasses File in the given base saveDir.
    An optional intermediate can be specified which will create a directory after the saveDir in which results will be saved to.

    :param figure: Figure to save
    :param save_dir: Path to base directory to which results will be saved to.
    :param path_intermediate: Optional intermediate that will add a directory after save_dir
    :param fileNamePrefix: Optional prefix that will be added in front of the fileName
    :param saveAdditional: save annfile and dataClasses if they were specified.
    :param default_Path: Default Path where the figure will be saved to
    :param default_Ann_Path: Default Path where the annfiles will be saved to
    :param default_DataClasses_Path: Default Path where the dataClasses file will be saved to
    :param fileName: Given filename. If specified no further inference for the name will be done and no additional files saved.
    :param fileExtension: Extension under which the figure will be saved as.
    """

    if fileName:
        figure_name = fileName
        saveDataClasses = False
        saveAnnfile = False
    else:
        figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(fileExtension=fileExtension, **kwargs)
    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    if save_dir:
        results_path = os.path.join(save_dir, path_intermediate)
        results_path_ann = os.path.join(save_dir, path_intermediate, 'annfiles')
        results_path_dataclasses = os.path.join(save_dir, path_intermediate, 'dataClasses')
    else:
        results_path = os.path.join(default_Path, path_intermediate)
        results_path_ann = os.path.join(default_Ann_Path, path_intermediate)
        results_path_dataclasses = os.path.join(default_DataClasses_Path, path_intermediate)

    saveFigure(savePath=os.path.join(results_path, figure_name), figure=figure, fileExtension=fileExtension)
    if saveAnnfile and saveAdditional:
        copyFile(kwargs['annfile'], os.path.join(results_path_ann, figure_name))
    if saveDataClasses and saveAdditional:
        writeArrayToFile(os.path.join(results_path_dataclasses, figure_name), kwargs['dataClasses'])

def generate_filtered_annfile(annfile, imgNames, fileName='annfile_filtered.txt', saveDir="./"):
    """Creates an updated annotation file under the given name that is a filtered version of the specified
    annotation file by the given image Names.
    The file be created in the same directory as the original file and the filename will be returned

    :param annfile: Path to the original annotation file.
    :type annfile: str | Path
    :param imgNames: Collection of the image Names that should be kept after filtering.
    :type imgNames: List | np.ndarray
    :param fileName: Name under which the filtered file will be saved to.
    :type fileName: str | Path
    :param saveDir: Where annotation file be saved to
    :type saveDir: str | Path
    """
    assert os.path.isfile(annfile), f'No such file {annfile}'
    filteredEntries = []
    with open(annfile, encoding='utf-8') as f:
        for entry in f.readlines():
            if entry.strip().rsplit(' ')[0] in imgNames:
                filteredEntries.append(entry)

    if fileName[-4:] != '.txt':
        fileName = fileName + '.txt'

    filePath = osp.join(saveDir, fileName)
    Path(osp.dirname(filePath)).mkdir(parents=True, exist_ok=True)
    print(f"Created filtered annotation file at {filePath}")

    with open(filePath, mode='w', encoding='utf-8') as f:
        f.write("".join(filteredEntries))
    
    return filePath

def generate_shared_annotation_file(annfile, cfg1, checkpoint1, cfg2, checkpoint2, saveDir='./'):
    """Creates an annotation file that only contains the entries of classes
    that exist in both models that are specified by the given configuration and checkpoint files.
    In addition the classIndices in the annotation file is remapped from model1 to model2
    The annfile must point to the annotation file that is the basis for the filtered file 
    and which will be reduced. It is assumed that this corresponds to an annfile of model 2

    :param annfile: Path to annotation file that is used as basis for the filtered file.
    :type annfile: str | Path
    :param cfg1: Path to config File for first model
    :type cfg1: str | Path
    :param checkpoint1: Path to checkpoint File for first model
    :type checkpoint1: str | Path
    :param cfg2: Path to config File for second model
    :type cfg2: str | Path
    :param checkpoint2: Path to checkpoint File for second model
    :type checkpoint2: str | Path
    :param saveDir: Where annotation file be saved to
    :type saveDir: str | Path
    """
    from .model import get_shared_classes
    from .preprocessing import remap_annfile
    mapping = get_shared_classes(cfg1,checkpoint1, cfg2, checkpoint2)
    dataClasses = [name for name,_,_ in mapping]
    # Use index2, index1 order here because annfile has indices of model 2
    indexMap = [(index2,index1) for _,index1,index2 in mapping]
    imgNames = get_samples(annfile=annfile, dataClasses=dataClasses)
    filteredAnnfile = generate_filtered_annfile(annfile, imgNames, fileName='annfile_shared_remapped.txt', saveDir=saveDir)
    remap_annfile(filteredAnnfile, indexMap)


def save_to_excel(arrs, filename='results.xlsx', saveDir='./', segments=None, categoryKey='segments'):
    """Saves the given arrays into an excel file. 
    If the arrays are passed in a dictionary the keys will be used as column names.
    """
    print("Generate excel file for results.")
    df = pandas.DataFrame()
    df[categoryKey] = segments
    if isinstance(arrs, dict):
        print("Using specified dictionary keys as column names")
        for key, value in arrs.items():
            df[key] = pandas.Series(value, index=df.index[:len(value)])
    elif isinstance(arrs, list):
        print("No names specified default indices will be used for columns.")
        for arr in arrs:
            df = pandas.concat((df, pandas.Series(arr, index=df.index[:len(arr)])), axis=1)
    else:
        assert isinstance(arrs, np.ndarray), f'Unsupported type passed for objects to be saved: {type(arrs)}'
        df = pandas.concate((df, pandas.Series(arrs)), axis=1)
    
    savePath = osp.join(saveDir, filename)
    if savePath[-5:] != '.xlsx':
        savePath = savePath + '.xlsx'
    Path(osp.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    print(f'Saving excel to {savePath}')
    df.to_excel(savePath)

def save_excel_auto_name(arrs, fileNamePrefix="", save_dir='', path_intermediate='', segments=None, fileName='', categoryKey='segments', **kwargs):
    """Saves the given arrays into an excel file.
    The filename will be determined dynamically based on the passed arguments like camConfig, camCheckpoint,
    camData, segConfig, segCheckpoint, segData using the get_save_figure_name function.
    After determination of the filename the saving will be performed by save_to_excel function

    :param arrs: arrays to be saved. Can either be a dictionary of a list. For dictionaries the key will be used as names.
    :param fileNamePrefix: Prefix that will be added in front of the determined filename
    :type fileNamePrefix: str
    :param save_dir: Path to dictionary where file will be saved to, if not specified it will default to RESULTS_PATH
    :type save_dir: str | Path, optional
    :param path_intermediate: Optional intermediate that will add a directory after save_dir, defaults to ''
    :type path_intermediate: str | Path, optional
    :param segments: If specified will use the given list of segments as indices (RowNames), defaults to None
    :type segments: List(str), optional
    :param fileName: Given filename. If specified no further inference for the name will be done and no additional files saved.
    :type fileName: str
    :param categoriesKey: If index_col=None the categories will be loaded under that column. Defaults to 'segments'
    :type categoriesKey: str
    """
    if fileName:
        file_name = fileName
    else:
        file_name, _, _ = get_save_figure_name(fileExtension='.xlsx',**kwargs)
    if fileNamePrefix:
        file_name = fileNamePrefix + "_" + file_name

    if save_dir:
        results_path = osp.join(save_dir, path_intermediate)
    else:
        results_path = osp.join(RESULTS_PATH, path_intermediate)

    save_to_excel(arrs, filename=file_name, saveDir=results_path, segments=segments, categoryKey=categoryKey)

def save_json(data, save_dir="", fileName="", fullPath="", fileNamePrefix=""):
    """Saves the given data into a json file at the given path.

    :param data: Data to be saved in json
    :type data: dict
    :param save_dir: Path to save directory 
    :type save_dir: str | Path
    :param fileName: Name of the file
    :type fileName: str
    :param fullPath: Fully specified Path if specified ignore save_dir and fileName
    :type fullPath: str | Path, optional
    :param fileNamePrefix: Prefix that will be used when fullPath is given.
    :type fileNamePrefix: str
    """
    if fullPath:
        save_path = osp.join(osp.dirname(fullPath), fileNamePrefix + '_' + osp.basename(fullPath))
    else:
        save_path = osp.join(save_dir, fileName)
    if save_path[-5:] != '.json':
        save_path = save_path + ".json"
    Path(osp.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        print(f'Saving json data to {save_path}')
        json.dump(data, f)

def load_results_excel(path, columnMap, index_col=0, sort=False, categoriesKey='segments'):
    """
    Loads the results from an excel file and returns the specified columns in a dictionary.
    containing the values in np.arrays under the key of the columnMap Dictionary.
    If the columnMap is a list of values(strings) the columns with these names will be returned as a list in 
    the same order as in the list 
    If the columnMap is just a single string only the column of that value will be returned.
    Additionally the index/categories will be returned as the first return value.
    If sort=True each column is sorted by its name and each entry in results will be a tuple (segment, value) 
    and returned in a zip object

    Args:
        path (str|Path): Path to excel file
        columnMap (Dict(str,str)): List of column Names that should be loaded.
        index_col (int, optional): Parameter for read_excel. What column to use for index. Defaults to 0.
        categoriesKey (str): If index_col=None the categories will be loaded under that column. Defaults to 'segments'
    """
    df = pandas.read_excel(path, index_col=index_col)
    categories = np.array(df[categoriesKey])
    if sort:
        if isinstance(columnMap, dict):
            results = {name:zip(np.array(df.sort_values(column, ascending=False)[column].index, dtype=int),
                                np.array(df.sort_values(column, ascending=False)[column])) 
                                for name, column in columnMap.items()}
        elif isinstance(columnMap, list):
            results = [zip(np.array(df.sort_values(column, ascending=False)[column].index, dtype=int),
                                np.array(df.sort_values(column, ascending=False)[column])) 
                                for column in columnMap]
        elif isinstance(columnMap, str):
            results = zip(np.array(df.sort_values(columnMap, ascending=False)[columnMap].index), np.array(df.sort_values(columnMap, ascending=False)[columnMap]))
        else:
            raise ValueError(f'Unsupported type for columnMap: {type(columnMap)}. Allowed is dict,list, str')
    else:
        if isinstance(columnMap, dict):
            results = {name:np.array(df[column]) for name, column in columnMap.items()}
        elif isinstance(columnMap, list):
            results = [np.array(df[column] for column in columnMap)]
        elif isinstance(columnMap, str):
            results = np.array(df[columnMap])
        else:
            raise ValueError(f'Unsupported type for columnMap: {type(columnMap)}. Allowed is dict,list, str')
    return categories, results

def load_result_excel_pandas_longform(path, value_vars, index_col=0, id_vars='segments', var_name='type', value_name='vals'):
    """Loads the specified excel table and converts it into long form data using pd.melt with the given parameters.

    Args:
        path (str): Path to excel file.
        value_vars (list|tuple|np.ndarray): Names of columns that are reformatted into long form and useable as hue/color
        index_col (int, optional): Parameter for read_excel. What column to use for index. Defaults to 0.
        id_vars (str, optional): Name of columns retained as indices -> Used for x parameter. Defaults to 'segments'.
        var_name (str, optional): Name of the columns the reformatted columns are under -> name of hue/color column. Defaults to 'type'.
        value_name (str, optional): Name of the columns the values of the reformatted columns are under -> used for y parameter. Defaults to 'vals'.

    Returns:
        pandas.DataFrame: Dataframe with loaded data in long-form
    """
    df = pandas.read_excel(path, index_col=index_col)
    df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)
    return df