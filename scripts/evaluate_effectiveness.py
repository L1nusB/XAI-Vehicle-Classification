import os.path as osp
import os

from .utils.evaluation import compare_original_blurred
from .utils.io import get_samples, generate_filtered_annfile, load_results_excel
from .utils.BlurDataset import setup_config_blurred

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmseg.datasets.builder import build_dataset

from mmcls.models.builder import build_classifier

def evaluate_blurred_background(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, saveImgs=False, saveDir='./', use_gpu=True, imgRootBlurred="", **kwargs):
    """
    Compares the original model on the original dataset compared to
    the model on a modified dataset, where the background is blurred out.

    IF LOADING BLURRED IMAGES THE RESULTS ARE SLIGHTY DIFFERENT FROM THE ON DEMAND BLURRED ONES BECAUSE OF IMAGE COMPRESSION

    :param imgRoot: Directory of where sample data lies.
    :param classifierConfig: Path to config for Classifier
    :param classifierCheckpoint: Path to Checkpoint for Classifier.
    :param annfile: Path to annotation file.
    :param segData: Path to numpy file containing the segmentation data. Must match to original image data shape.
    :param saveImgs: Save blurred images.
    :param saveDir: Path where results will be saved to.
    :param imgRootBlurred: Path to directory where blurred image data is. If specified original dataset will be used on that dataset.

    For kwargs:
    segCheckpoint: Path to Checkpoint for segmentation model.
    segConfig: Path to Config for segmentation model.
    classes: Already loaded classes of the segmentation model.
    dataClasses: list(str) containing classes that will be filtered for.
    blurKernel: Kernel for gaussian blur defaults to (33,33).
    blurSigmaX: sigmaX for gaussian blur defaults to 0.
    saveJson: Save json file containing metrics in addition to excel file.
    metrics: List of metrics under which the model will be evaluated. If not specified defaults to ['accuracy', 'precision', 'recall', 'f1_score', 'support']
    filename: Filename under which the excel file will be saved to. Default to 'evalBlurred.xlsx'
    eval_data_original: Path to json file containing evaluation results of the original model.
    evaluate_original: Whether to evaluate the original dataset. Defaults to True. If eval_data_original given it will be set to False.
    """
    print('Evaluating original model vs blurred background.')

    return evaluate_blurred(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, 'background', saveImgs, saveDir, use_gpu, imgRootBlurred, **kwargs)

def evaluate_blurred(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, blurredSegments, saveImgs=False, saveDir='./', use_gpu=True, imgRootBlurred="", **kwargs):
    """
    Compares the original model on the original dataset compared to
    the model on a modified dataset, where the specified Segments are blurred out.

    IF LOADING BLURRED IMAGES THE RESULTS ARE SLIGHTY DIFFERENT FROM THE ON DEMAND BLURRED ONES BECAUSE OF IMAGE COMPRESSION

    :param imgRoot: Directory of where sample data lies.
    :param classifierConfig: Path to config for Classifier
    :param classifierCheckpoint: Path to Checkpoint for Classifier.
    :param annfile: Path to annotation file.
    :param segData: Path to numpy file containing the segmentation data. Must match to original image data shape.
    :param blurredSegments: Which segment/-s to blur out. Can be either index, string or list of either.
    :param saveImgs: Save blurred images.
    :param saveDir: Path where results will be saved to.
    :param imgRootBlurred: Path to directory where blurred image data is. If specified original dataset will be used on that dataset.

    For kwargs:
    segCheckpoint: Path to Checkpoint for segmentation model.
    segConfig: Path to Config for segmentation model.
    classes: Already loaded classes of the segmentation model.
    dataClasses: list(str) containing classes that will be filtered for.
    blurKernel: Kernel for gaussian blur defaults to (33,33).
    blurSigmaX: sigmaX for gaussian blur defaults to 0.
    saveJson: Save json file containing metrics in addition to excel file.
    metrics: List of metrics under which the model will be evaluated. If not specified defaults to ['accuracy', 'precision', 'recall', 'f1_score', 'support']
    filename: Filename under which the excel file will be saved to. Default to 'evalBlurred.xlsx'
    eval_data_original: Path to json file containing evaluation results of the original model.
    evaluate_original: Whether to evaluate the original dataset. Defaults to True. If eval_data_original given it will be set to False.
    """
    print(f'Evaluating original model vs blurred where segments {blurredSegments} are blurred.')

    imgNames = get_samples(imgRoot=imgRoot, annfile=annfile, **kwargs)
    filteredAnnfilePath = generate_filtered_annfile(annfile=annfile, imgNames=imgNames, saveDir=saveDir)

    cfg = Config.fromfile(classifierConfig)
    cfg.data.test['ann_file'] = filteredAnnfilePath
    cfg.data.test['data_prefix'] = imgRoot
    dataset_original = build_dataset(cfg.data.test)
    model = build_classifier(cfg.model)
    load_checkpoint(model, classifierCheckpoint)
    if use_gpu:
        model = MMDataParallel(model, device_ids=[0])
    if 'eval_data_original' in kwargs:
        kwargs['evaluate_original'] = False
    if imgRootBlurred:
        print(f'Using standard dataset on blurred data at directory {imgRootBlurred}')
        cfg.data.test['data_prefix'] = imgRootBlurred
        dataset_blurred = build_dataset(cfg.data.test)
    else:
        print('Computing blurred images on demand.')
        blur_cfg = setup_config_blurred(cfg=cfg, imgRoot=imgRoot, annfile=annfile,saveDir=osp.join(saveDir, 'blurredImgs'), 
                                            segData=segData, saveImgs=saveImgs, blurredSegments=blurredSegments,**kwargs)
        #dataset_blurred = build_dataset(blur_cfg.data.test)
        #print(blur_cfg.data.test)
        dataset_blurred = build_dataset(blur_cfg.data.test)
    #dataset_blurred = get_blurred_dataset(cfg=cfg, imgRoot=imgRoot, annfile=annfile, blurredSegments=[0,1], segData=segData, **kwargs)

    compare_original_blurred(model=model, cfg=cfg, dataset_original=dataset_original, dataset_blurred=dataset_blurred, saveDir=saveDir, use_gpu=use_gpu, **kwargs)

    print(f'Removing temporary filtered annotation file {filteredAnnfilePath}')
    os.remove(filteredAnnfilePath)

def evaluate_blurred_rel_importance(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, importanceScoreFile, importanceScoreColumnName='RelativeCAMImportance', 
                                    numBlurred=3, saveImgs=False, saveDir='./', use_gpu=True, imgRootBlurred="", **kwargs):
    """
    Compares the original model on the original dataset compared to
    the model on a modified dataset, where the segments with the highest relative importance are blurred out.
    The amount of blurred segments is specified via numBlurred (default 3)
    The relative importance scores will be infered from the excel file under importanceScoreFile 
    in Column importanceScoreColumnName (default 'RelativeCAMImportance')

    IF LOADING BLURRED IMAGES THE RESULTS ARE SLIGHTY DIFFERENT FROM THE ON DEMAND BLURRED ONES BECAUSE OF IMAGE COMPRESSION

    :param imgRoot: Directory of where sample data lies.
    :param classifierConfig: Path to config for Classifier
    :param classifierCheckpoint: Path to Checkpoint for Classifier.
    :param annfile: Path to annotation file.
    :param segData: Path to numpy file containing the segmentation data. Must match to original image data shape.
    :param numBlurred: Amount of segment categories which will be blurred out. Defaults to 3
    :param saveImgs: Save blurred images.
    :param saveDir: Path where results will be saved to.
    :param imgRootBlurred: Path to directory where blurred image data is. If specified original dataset will be used on that dataset.

    For kwargs:
    segCheckpoint: Path to Checkpoint for segmentation model.
    segConfig: Path to Config for segmentation model.
    classes: Already loaded classes of the segmentation model.
    dataClasses: list(str) containing classes that will be filtered for.
    blurKernel: Kernel for gaussian blur defaults to (33,33).
    blurSigmaX: sigmaX for gaussian blur defaults to 0.
    saveJson: Save json file containing metrics in addition to excel file.
    metrics: List of metrics under which the model will be evaluated. If not specified defaults to ['accuracy', 'precision', 'recall', 'f1_score', 'support']
    filename: Filename under which the excel file will be saved to. Default to 'evalBlurred.xlsx'
    eval_data_original: Path to json file containing evaluation results of the original model.
    evaluate_original: Whether to evaluate the original dataset. Defaults to True. If eval_data_original given it will be set to False.
    """
    print(f'Evaluating original model vs blurred relative Importance with top {numBlurred} blurred.')
    categories, relImportanceScores = load_results_excel(importanceScoreFile, importanceScoreColumnName, sort=True)
    if numBlurred > len(categories):
        raise ValueError(f'Can not blurr {numBlurred} segments. Only {len(categories)} present. ')
    print(f'Blurring {numBlurred} segments out of {len(categories)}.')

    blurredSegmentIndices = [index for index, _ in relImportanceScores][:numBlurred]

    print(f'Blurring segment with index {",".join(blurredSegmentIndices)}')
    print(f'Blurring segments {",".join(categories[blurredSegmentIndices])}')

    return evaluate_blurred(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, blurredSegmentIndices, saveImgs, saveDir, use_gpu, imgRootBlurred, **kwargs)

def evaluate_blurred_normalized_by_rel_importance(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, importanceScoreFile, importanceScoreColumnName='PercActivationsRescaled', 
                                    numBlurred=3, saveImgs=False, saveDir='./', use_gpu=True, imgRootBlurred="", **kwargs):
    """
    Compares the original model on the original dataset compared to
    the model on a modified dataset, where the segments with the highest rescaled/normalized CAM Activation
    w.r.t the relative importance are blurred out.
    The amount of blurred segments is specified via numBlurred (default 3)
    The relative importance scores will be infered from the excel file under importanceScoreFile 
    in Column importanceScoreColumnName (default 'RelativeCAMImportance')

    IF LOADING BLURRED IMAGES THE RESULTS ARE SLIGHTY DIFFERENT FROM THE ON DEMAND BLURRED ONES BECAUSE OF IMAGE COMPRESSION

    :param imgRoot: Directory of where sample data lies.
    :param classifierConfig: Path to config for Classifier
    :param classifierCheckpoint: Path to Checkpoint for Classifier.
    :param annfile: Path to annotation file.
    :param segData: Path to numpy file containing the segmentation data. Must match to original image data shape.
    :param numBlurred: Amount of segment categories which will be blurred out. Defaults to 3
    :param saveImgs: Save blurred images.
    :param saveDir: Path where results will be saved to.
    :param imgRootBlurred: Path to directory where blurred image data is. If specified original dataset will be used on that dataset.

    For kwargs:
    segCheckpoint: Path to Checkpoint for segmentation model.
    segConfig: Path to Config for segmentation model.
    classes: Already loaded classes of the segmentation model.
    dataClasses: list(str) containing classes that will be filtered for.
    blurKernel: Kernel for gaussian blur defaults to (33,33).
    blurSigmaX: sigmaX for gaussian blur defaults to 0.
    saveJson: Save json file containing metrics in addition to excel file.
    metrics: List of metrics under which the model will be evaluated. If not specified defaults to ['accuracy', 'precision', 'recall', 'f1_score', 'support']
    filename: Filename under which the excel file will be saved to. Default to 'evalBlurred.xlsx'
    eval_data_original: Path to json file containing evaluation results of the original model.
    evaluate_original: Whether to evaluate the original dataset. Defaults to True. If eval_data_original given it will be set to False.
    """
    print(f'Evaluating original model vs blurred relative Importance with top {numBlurred} blurred.')
    categories, percActivationsNormalized = load_results_excel(importanceScoreFile, importanceScoreColumnName, sort=True)
    if numBlurred > len(categories):
        raise ValueError(f'Can not blurr {numBlurred} segments. Only {len(categories)} present. ')
    print(f'Blurring {numBlurred} segments out of {len(categories)}.')

    blurredSegmentIndices = [index for index, _ in percActivationsNormalized][:numBlurred]

    print(f'Blurring segment with index {",".join(blurredSegmentIndices)}')
    print(f'Blurring segments {",".join(categories[blurredSegmentIndices])}')

    return evaluate_blurred(imgRoot, classifierConfig, classifierCheckpoint, annfile, segData, blurredSegmentIndices, saveImgs, saveDir, use_gpu, imgRootBlurred, **kwargs)