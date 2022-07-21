import matplotlib.pyplot as plt
import os
import numpy as np

from .utils.io import get_samples, saveFigure, get_save_figure_name, copyFile, writeArrayToFile
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.pipeline import get_pipeline_torchvision
from .utils.calculations import generate_stats
from .utils.plot import plot_bar
from .utils.preprocessing import load_classes, batch_statistics
from .utils.constants import RESULTS_PATH_ANN,RESULTS_PATH, RESULTS_PATH_DATACLASS

def new(classes=None, **kwargs):
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
    for name in imgNames:
        transformedSegmentations[name] = pipeline(segmentations[name]) if cfg else segmentations[name]

    classes = load_classes(classes, **kwargs)

    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, **kwargs)  # forceAll can be set in kwargs if desired

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

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(statType='Multiple', **kwargs)

    saveFigure(savePath=os.path.join(RESULTS_PATH, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(RESULTS_PATH_ANN, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(RESULTS_PATH_DATACLASS, figure_name), kwargs['dataClasses'])



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

    classes = load_classes(classes, **kwargs)

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

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(statType='Multiple', additional='ShowPropArea', **kwargs)

    saveFigure(savePath=os.path.join(RESULTS_PATH, figure_name), figure=fig)