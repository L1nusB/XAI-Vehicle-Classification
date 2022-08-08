import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np

from .utils.io import get_samples, saveFigure, get_save_figure_name, copyFile, writeArrayToFile
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.pipeline import get_pipeline_torchvision
from .utils.calculations import generate_stats, accumulate_statistics
from .utils.plot import plot_bar
from .utils.preprocessing import load_classes, batch_statistics
from .utils.constants import RESULTS_PATH_ANN,RESULTS_PATH, RESULTS_PATH_DATACLASS

def generate_statistic(classes=None, fileNamePrefix="" , **kwargs):
    """Generates a plot with average absolute and average relative CAM Activations.

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param fileNamePrefix: Prefix that will be added in front of the filenames under which the results are saved.

    Relevant kwargs are:
    imgRoot: Path to root folder where images/samples lie
    camConfig: Path to config of the used CAM Model
    camCheckpoint: Path to Checkpoint of the used CAM Model
    camData: Path to file containing generated CAMs (or dictionary). Can be used instead of Config and Checkpoint
    camDevice: Device used for generating CAMs if needed. Defaults to 'cpu'
    method: Method used for generating the CAMs. Defaults to 'gradcam'
    segConfig: Path to config of the used Segmentation Model
    segCheckpoint: Path to Checkpoint of the used Segmentation Model
    segData: Path to file containing generated Segmentations (or dictionary). Can be used instead of Config and Checkpoint
    segDevice: Device used for generating the segmentations. Defaults to 'cuda'
    annfile: Path to annotation file specifng which samples should be used.
    dataClasses: Array of Class Prefixes that specify which sample classes should be used. If not specified everything will be generated.
    """
    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'annfile' in kwargs or 'imgRoot' in kwargs, 'Either annfile or imgRoot must be specified.'
        imgNames = get_samples(**kwargs) # Required annfile or (imgRoot) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')


    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    # For Seg: Here we need segConfig, segCheckpoint or segData, imgRoot, (segDevice), (dataClasses) and (annfile)
    segmentations, _, cams = prepareInput(prepImg=False, **kwargs)
    assert (isinstance(cams, dict) and set(imgNames).issubset(set(cams.keys()))) or set(imgNames).issubset(set(cams.files)), f'Given CAM Dictionary does not contain all imgNames as keys.'
    assert (isinstance(cams, dict) and set(imgNames).issubset(set(segmentations.keys()))) or set(imgNames).issubset(set(segmentations.files)), f'Given Segmentation Dictionary does not contain all imgNames as keys.'

    transformedSegmentations = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
        print('Tranforming segmentation masks with the given pipeline.')
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    classes = load_classes(classes, **kwargs)

    #totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, **kwargs)

    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations = accumulate_statistics(imgNames=imgNames, classes=classes, cams=cams, segmentations=transformedSegmentations)

    classArray, totalActivation, summarizedSegmentedCAMActivations, dominantMask, summarizedPercSegmentedCAMActivations, dominantMaskPercentual = generate_stats(
        segmentedActivations=segmentedCAMActivations, percentualActivations=percentualSegmentedCAMActivations, totalCAM=totalCAMActivations, classes=classes)

    numSamples = len(imgNames)

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
    plt.show()

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(**kwargs)
    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    saveFigure(savePath=os.path.join(RESULTS_PATH, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(RESULTS_PATH_ANN, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(RESULTS_PATH_DATACLASS, figure_name), kwargs['dataClasses'])




def generate_statistic_prop(classes=None, fileNamePrefix="", showPropPercent=False, showNormalized=False, **kwargs):
    """Generates a plot with average absolute and average relative CAM Activations.

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param showPropPercent: (default False) Determine if percentage labels will be shown on the proportional area bars as well.
    :param fileNamePrefix: Prefix that will be added in front of the filenames under which the results are saved.
    :param showNormalized: Show a normalized Plot showing the activations relative to the proportions as well.

    Relevant kwargs are:
    imgRoot: Path to root folder where images/samples lie
    camConfig: Path to config of the used CAM Model
    camCheckpoint: Path to Checkpoint of the used CAM Model
    camData: Path to file containing generated CAMs (or dictionary). Can be used instead of Config and Checkpoint
    camDevice: Device used for generating CAMs if needed. Defaults to 'cpu'
    method: Method used for generating the CAMs. Defaults to 'gradcam'
    segConfig: Path to config of the used Segmentation Model
    segCheckpoint: Path to Checkpoint of the used Segmentation Model
    segData: Path to file containing generated Segmentations (or dictionary). Can be used instead of Config and Checkpoint
    segDevice: Device used for generating the segmentations. Defaults to 'cuda'
    annfile: Path to annotation file specifng which samples should be used.
    dataClasses: Array of Class Prefixes that specify which sample classes should be used. If not specified everything will be generated.
    """
    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'annfile' in kwargs or 'imgRoot' in kwargs, 'Either annfile or imgRoot must be specified.'
        imgNames = get_samples(**kwargs) # Required annfile or (imgRoot) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    # For Seg: Here we need segConfig, segCheckpoint or segData, imgRoot, (segDevice), (dataClasses) and (annfile)
    segmentations, _, cams = prepareInput(prepImg=False, **kwargs)
    assert set(imgNames).issubset(set(cams.keys())), f'Given CAM Dictionary does not contain all imgNames as keys.'
    assert set(imgNames).issubset(set(segmentations.keys())), f'Given Segmentation Dictionary does not contain all imgNames as keys.'

    transformedSegmentations = {}
    cfg = get_pipeline_cfg(**kwargs)
    if cfg:
        pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
        print('Tranforming segmentation masks with the given pipeline.')
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    classes = load_classes(classes, **kwargs)

    #_, _, percentualSegmentedCAMActivations, percentualSegmentAreas =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations,percentualArea=True ,**kwargs)  # forceAll can be set in kwargs if desired

    _, _, percentualSegmentedCAMActivations, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=segmentations, classes=classes, percentualArea=True)

    # Pass fake segmentedActivations and totalCAM since i don't care about results.
    classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    if showNormalized:
        grid = fig.add_gridspec(ncols=1, nrows=2)
    else:
        grid = fig.add_gridspec(ncols=1, nrows=1)
    

    ax0 = fig.add_subplot(grid[0,0])
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
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercSegmentAreas, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMaskPercentual, addText=showPropPercent, hightlightDominant=False,
        textadjust_ypos=showPropPercent, format='.1%', textrotation=rotation)

    ax0.text(0.9,1.02, f'No.Samples:{len(imgNames)}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

    legendMap = {
        'b':'CAM Activations',
        'r':'Top CAM Activations',
        'g':'Proportional Segment Coverage'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax0.legend(handles=handles)

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')


    if showNormalized:
        import heapq
        ax1 = fig.add_subplot(grid[1,0])
        ax1.set_title('Normalized Area Activations')
        normalizedActivations = percentualSegmentedCAMActivations / percentualSegmentAreas

        dominantMaskNormalizedRaw = heapq.nlargest(3,normalizedActivations)
        dominantMaskNormalized = normalizedActivations >= np.min(dominantMaskNormalizedRaw)

        plot_bar(ax=ax1, x_ticks=classArray, data=normalizedActivations, dominantMask=dominantMaskNormalized, format='.1%')

    plt.show()

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(additional='ShowPropArea', **kwargs)

    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    saveFigure(savePath=os.path.join(RESULTS_PATH, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(RESULTS_PATH_ANN, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(RESULTS_PATH_DATACLASS, figure_name), kwargs['dataClasses'])