import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np
import heapq
from mmseg.apis import init_segmentor

from .utils.io import get_samples, saveFigure, saveResults
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.pipeline import get_pipeline_torchvision
from .utils.calculations import accumulate_statistics, generate_stats
from .utils.plot import plot_bar
from .utils.preprocessing import add_background_class, batch_statistics


def generate_statistic(imgNames, cams, segmentations, classes, forceAll=False, saveResults=None):
    """Accumulates all intersections for the given imgNames.
    Returns both absolut and percentual statistics.

    :param imgNames: List of imgNames. Keys into the cams and segmentations
    :type imgNames: list or tuple
    :param cams: CAMs stored in a dictionary keyed by the imgNames
    :type cams: dict
    :param segmentations: Segmentations stored in a dictionary keyed by the imgNames
    :type segmentations: dict like
    :param classes: Segmentation classes. 
    :type classes: tuple/list
    :param pipeline: Pipeline that is applied to the segmentations. If pipeline=None nothing is applied
    :param forceAll: Forces function to take average over all objects without batching. Only relevant for large sample amounts.(default False)
    :type forceAll: boolean
    :param saveResults: Save the results in array form and the plots at the given Path. Only relevant if not None.
    """


    accumulateLimit = 10000
    classArray = np.array(classes)
    numSamples = len(imgNames)
    numSplits = numSamples // accumulateLimit + 1
    # Decrease by one if exact match
    if numSplits*accumulateLimit == numSamples:
        numSplits-=1
    totalCAMActivations = np.zeros((numSplits,accumulateLimit))
    segmentedCAMActivations = np.zeros((numSplits,accumulateLimit, classArray.size))
    percentualSegmentedCAMActivations = np.zeros((numSplits,accumulateLimit, classArray.size))
    if numSamples > accumulateLimit:
        warnings.warn(f'Statistics computed over {numSamples} items. Reverting to using batches of size {accumulateLimit} '
        'to avoid overflows. Can be overriden by using forceAll=True')
    for batch in range(numSplits):
        lower = accumulateLimit*batch
        higher = accumulateLimit*(batch+1)
        if forceAll:
            totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation = accumulate_statistics(imgNames, cams, segmentations, classArray.size)
        else:
            print(f'Generating data for Batch {batch+1}')
            totalCAMActivation, segmentedCAMActivation, percentualSegmentedCAMActivation = accumulate_statistics(imgNames[lower:higher], cams, segmentations, classArray.size)
        totalCAMActivations[batch] = np.pad(totalCAMActivation, pad_width=(0,totalCAMActivations.shape[1]-totalCAMActivation.shape[0]), mode='constant')
        segmentedCAMActivations[batch] = np.pad(segmentedCAMActivation, pad_width=((0,segmentedCAMActivations.shape[1]-segmentedCAMActivation.shape[0]),(0,0)), mode='constant')
        percentualSegmentedCAMActivations[batch] = np.pad(percentualSegmentedCAMActivation, pad_width=((0,percentualSegmentedCAMActivations.shape[1]-percentualSegmentedCAMActivation.shape[0]),(0,0)), mode='constant')
    
    print('Data generated.')

    totalCAMActivations = np.array(totalCAMActivations)
    segmentedCAMActivations = np.array(segmentedCAMActivations)
    percentualSegmentedCAMActivations = np.array(percentualSegmentedCAMActivations)

    totalActivation = totalCAMActivations.sum()
    summarizedSegmentedCAMActivations = segmentedCAMActivations.mean(axis=1).mean(axis=0)
    summarizedPercSegmentedCAMActivations = percentualSegmentedCAMActivations.mean(axis=1).mean(axis=0)

    dominantSegmentsRaw = heapq.nlargest(3,summarizedSegmentedCAMActivations)
    dominantMaskRaw = summarizedSegmentedCAMActivations >= np.min(dominantSegmentsRaw)

    dominantSegmentsPerc = heapq.nlargest(3,summarizedPercSegmentedCAMActivations)
    dominantMaskPerc = summarizedPercSegmentedCAMActivations >= np.min(dominantSegmentsPerc)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[0,1])

    # For the absolut we take the total Activations, sum them and only afterwards get a percentage relative to the total CAM Activations
    # across all sampels.
    bars = ax0.bar(classArray, summarizedSegmentedCAMActivations)
    ticks_loc = ax0.get_xticks()
    ax0.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax0.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMaskRaw):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax0.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height*numSamples/totalActivation:.1%}', ha='center', va='bottom' )
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.set_title('Average absolut CAM Activations')

    # For the relative we immediately take the percentage activations of each sample and average them afterwards.
    bars = ax1.bar(classArray, summarizedPercSegmentedCAMActivations)
    ticks_loc = ax1.get_xticks()
    ax1.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax1.set_xticklabels(classArray, rotation=45, ha="right")
    for index,dominant in enumerate(dominantMaskPerc):
        if dominant:
            bars[index].set_color('red')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x()+bar.get_width()/2.0, bar.get_height() , f'{height:.1%}', ha='center', va='bottom' )
    ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)
    ax1.set_title('Average relative CAM Activations')

    if saveResults:
        saveResults(saveResults, totalActivation=totalActivation, segmentedCAMActivations=summarizedSegmentedCAMActivations, percSegmentedCAMActivations=summarizedPercSegmentedCAMActivations)
        saveFigure(saveResults, fig)

    return ax0, ax1

def generate_statistics_infer(imgRoot, classes=None, camConfig=None, camCheckpoint=None, segConfig=None, segCheckpoint=None, annfile=None, genClasses=None, pipeline=None, forceAll=False, saveResults=None, **kwargs):
    """Generate statistics for average absolute and average relative Intersection of CAM with 
    segmentation Mask. See @generate_statistics. Infers the CAMs and Segmentation from parameters
    and calls generate_statistics with generated objects.

    :param imgRoot: Root the directory of the images
    :type imgRoot: str
    :param camConfig: Path to the Config of the CAM Model
    :type camConfig: str
    :param camCheckpoint: Path to the Checkpoint of the CAM Model
    :type camCheckpoint: str
    :param segConfig: Path to the config of the Segmentation Model
    :type segConfig: str
    :param segCheckpoint: Path to the checkpoint of the Segmentation Model
    :type segCheckpoint: str
    :param classes: Segmentation classes.
    :type classes: tuple/list
    :param annfile: Path to annfile defining which images will be used from imgRoot. If None all images in 
    imgRoot will be used, defaults to None
    :type annfile: str, optional
    :param genClasses: Class for which the statistic will be generated. Will match all images in imgRoot
    containing the specified Class. Can be combined with annfile., defaults to None
    :type genClasses: str, optional
    :param pipeline: Pipeline that will be applied to segmentations in generate_statistics.
    :param forceAll: Forces function to take average over all objects without batching. Only relevant for large sample amounts.(default False)
    :type forceAll: boolean

    For kwargs:
    getPipelineFromConfig: (bool) Load the pipeline from CAMConfig. Requires the pipeline to be located under
    data.test.pipeline
    scalePipelineToInt: (bool) Defines whether to scale the pipeline to Int see @transformations.get_pipeline_from_config_pipeline
    imgNames: (list like) Preloaded list of imageNames. If specified will be passed but MUST match what is generated from other parameters.
    cams: (dict(str,np.ndarray)) Pregenerated dict of CAMs. If specified will be passed but MUST contain what is generated from other parameters.
    segmentations: (dict(str, np.ndarray)) Pregenerated dict of Segmentations. If specified will be passed but MUST contain what is generated from other parameters.
    """
    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'ann_file' in kwargs or ('imgRoot' in kwargs and 'imgDir' in kwargs), 'Either ann_file or imgRoot and imgDir must be specified.'
        imgNames = get_samples(**kwargs) # Required ann_file or (imgRoot and imgDir) in kwargs
        #imgNames = utils.getImageList(imgRoot, annfile, classes=genClasses, addRoot=False)

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    # For Seg: Here we need segConfig, segCheckpoint or segData, imgRoot, (segDevice), (dataClasses) and (annfile)
    cams, segmentations, _ = prepareInput(prepImg=False, **kwargs)
    assert set(imgNames).issubset(set(cams.keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
    assert set(imgNames).issubset(set(segmentations.keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'

    # if 'cams' in kwargs:
    #     assert set(imgNames).issubset(set(kwargs['cams'].keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
    #     cams = kwargs['cams']
    # else:
    #     assert os.path.isfile(camConfig), f'camConfig is no file {camConfig}'
    #     assert os.path.isfile(camCheckpoint), f'camCheckpoint is no file {camCheckpoint}'
    #     # genClasses must be set to dataClasses for prepareData kwargs argument
    #     cams = generate_cams.main([imgRoot, camConfig, camCheckpoint, '--ann-file', annfile, '--classes', genClasses])

    # if 'segmentations' in kwargs:
    #     assert set(imgNames).issubset(set(kwargs['segmentations'].keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'
    #     segmentations = kwargs['segmentations']
    # else:
    #     assert os.path.isfile(segConfig), f'segConfig is no file {segConfig}'
    #     assert os.path.isfile(segCheckpoint), f'segCheckpoint is no file {segCheckpoint}'
    #     if genClasses:
    #         segmentations,_ = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--types', 'masks', '--ann-file', annfile, '--classes', genClasses])
    #     else:
    #         segmentations,_ = new_gen_seg.main([imgRoot, segConfig, segCheckpoint,'--types', 'masks', '--ann-file', annfile])

    # if 'getPipelineFromConfig' in kwargs:
    #     cfg = Config.fromfile(camConfig)
    #     if 'scalePipelineToInt' in kwargs:
    #         pipeline = transformations.get_pipeline_from_config_pipeline(cfg.data.test.pipeline, scaleToInt=True)
    #     else:
    #         pipeline = transformations.get_pipeline_from_config_pipeline(cfg.data.test.pipeline)

    # If desired pipelineCfg can be included into kwargs here
    transformedSegmentations = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    if classes is None:
        assert 'segConfig' in kwargs and 'segCheckpoint' in kwargs, 'Required segConfig and segCheckpoint if classes are not specified.'
        model = init_segmentor(kwargs['segConfig'], kwargs['segCheckpoint'])
        classes = model.CLASSES
    if 'addBackground' in kwargs:
        classes = classes + ('background',) if kwargs['addBackground']==True else classes
    else:
        classes = classes + ('background',)

    ax0,ax1 = generate_statistic(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, forceAll=forceAll, saveResults=saveResults)

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
        ax1.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')



def new(classes=None, **kwargs):
    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'ann_file' in kwargs or ('imgRoot' in kwargs and 'imgDir' in kwargs), 'Either ann_file or imgRoot and imgDir must be specified.'
        imgNames = get_samples(**kwargs) # Required ann_file or (imgRoot and imgDir) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    # For Seg: Here we need segConfig, segCheckpoint or segData, imgRoot, (segDevice), (dataClasses) and (annfile)
    cams, segmentations, _ = prepareInput(prepImg=False, **kwargs)
    assert set(imgNames).issubset(set(cams.keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
    assert set(imgNames).issubset(set(segmentations.keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'

    transformedSegmentations = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    classes = add_background_class(classes, **kwargs)

    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=segmentations, **kwargs)  # forceAll can be set in kwargs if desired

    classArray, totalActivation, summarizedSegmentedCAMActivations, summarizedPercSegmentedCAMActivations, dominantMask, dominantMaskPercentual = generate_stats(
        segmendtedActivations=segmentedCAMActivations, percentualActivations=percentualSegmentedCAMActivations, totalCAM=totalCAMActivations, classes=classes)

    numSamples = len(classArray)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=2, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    ax1 = fig.add_subplot(grid[0,1])

    # Plot segmentedCAMActivations aka the average of the absolute CAM Activations
    plot_bar(ax=ax0, x_ticks=classArray, data=summarizedSegmentedCAMActivations, dominantMask=dominantMask, format='.1%', 
            textvalueModifier=numSamples/totalActivation)
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.set_title('Average absolut CAM Activations')

    # Plot percentualSegmentedCAMActivations aka the relative of the absolute CAM Activations
    plot_bar(ax=ax1, x_ticks=classArray, data=summarizedPercSegmentedCAMActivations, dominantMask=dominantMaskPercentual, format='.1%')
    ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)
    ax1.set_title('Average relative CAM Activations')

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
        ax1.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')


def newProp(classes=None, showPropPercent=False, **kwargs):
    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'ann_file' in kwargs or ('imgRoot' in kwargs and 'imgDir' in kwargs), 'Either ann_file or imgRoot and imgDir must be specified.'
        imgNames = get_samples(**kwargs) # Required ann_file or (imgRoot and imgDir) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    # For Seg: Here we need segConfig, segCheckpoint or segData, imgRoot, (segDevice), (dataClasses) and (annfile)
    cams, segmentations, _ = prepareInput(prepImg=False, **kwargs)
    assert set(imgNames).issubset(set(cams.keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
    assert set(imgNames).issubset(set(segmentations.keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'

    transformedSegmentations = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    classes = add_background_class(classes, **kwargs)

    _, _, percentualSegmentedCAMActivations, percentualSegmentAreas =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=segmentations,percentualArea=True ,**kwargs)  # forceAll can be set in kwargs if desired

    # Pass fake segmentedActivations and totalCAM since i don't care about results.
    classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)

    numSamples = len(classArray)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)

    ax0 = fig.add_subplot()
    ax0.set_title('Average relative CAM Activations')

    # Default width is 0.8 and since we are plotting two bars side by side avoiding overlap requires
    # reducing the width
    barwidth = 0.4

    bars = ax0.bar(np.arange(classArray.size), summarizedPercSegmentedCAMActivations, width=barwidth, label='CAM Activations')
    ax0.set_xticks([tick+barwidth/2 for tick in range(classArray.size)], classArray)

    rotation = 90 if showPropPercent else 0

    # Format main Data with generated bar graph
    plot_bar(ax=ax0, bars=bars, dominantMask=dominantMaskPercentual, x_tick_labels=classArray, format='.1%', textadjust_ypos=showPropPercent,
        textrotation=rotation)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, data=summarizedPercSegmentAreas, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMaskPercentual, addText=showPropPercent, hightlightDominant=False,
        textadjust_ypos=showPropPercent, format='.1%', textrotation=rotation)

    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')