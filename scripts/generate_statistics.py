import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np

from .utils.io import get_samples, saveFigure, get_save_figure_name, copyFile, writeArrayToFile
from .utils.prepareData import prepareInput, get_pipeline_cfg, prepare_generate_stats
from .utils.pipeline import get_pipeline_torchvision
from .utils.calculations import generate_stats, accumulate_statistics, get_area_normalized_stats
from .utils.plot import plot_bar, plot_errorbar
from .utils.preprocessing import load_classes
from .utils.constants import RESULTS_PATH_ANN,RESULTS_PATH, RESULTS_PATH_DATACLASS

def generate_statistic(classes=None, saveDir='', fileNamePrefix="" , **kwargs):
    """Generates a plot with average absolute and average relative CAM Activations.

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param saveDir: Directory Path where the resulting files will be saved to. If not specified defaults to './results/'
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
    imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

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
    ##### OPTION 1 FOR GENERATE_STATS_ABS #####
    plot_bar(ax=ax0, x_ticks=classArray, data=summarizedSegmentedCAMActivations, dominantMask=dominantMask, format='.1%', 
            textvalueModifier=numSamples/totalActivation)

    # So far OPTION 1 and OPTION 2 produce the SAME OUTPUT
    ##### OPTION 2 FOR GENERATE_STATS_ABS #####
    # plot_bar(ax=ax0, x_ticks=classArray, data=summarizedSegmentedCAMActivations, dominantMask=dominantMask, format='.1%')


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

    if saveDir:
        results_path = saveDir
        results_path_ann = os.path.join(saveDir, 'annfiles')
        results_path_dataclasses = os.path.join(saveDir, 'dataClasses')
    else:
        results_path = RESULTS_PATH
        results_path_ann = RESULTS_PATH_ANN
        results_path_dataclasses = RESULTS_PATH_DATACLASS

    saveFigure(savePath=os.path.join(results_path, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(results_path_ann, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(results_path_dataclasses, figure_name), kwargs['dataClasses'])




def generate_statistic_prop(classes=None, saveDir='', fileNamePrefix="", showPropPercent=False, **kwargs):
    """Generates a plot with average averaged relative CAM Activations and the corresponding area that each segment covered. 

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param showPropPercent: (default False) Determine if percentage labels will be shown on the proportional area bars as well.
    :param saveDir: Directory Path where the resulting files will be saved to. If not specified defaults to './results/'. Files will be saved 
                    under the given or default directory Path in a folder called 'statsProp'
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
    imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

    #_, _, percentualSegmentedCAMActivations, percentualSegmentAreas =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations,percentualArea=True ,**kwargs)  # forceAll can be set in kwargs if desired

    _, _, percentualSegmentedCAMActivations,_, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)

    classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=1)
    
    numSamples = len(imgNames)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
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

    plt.show()

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(additional='ShowPropArea', **kwargs)

    if saveDir:
        results_path = os.path.join(saveDir, 'statsProp')
        results_path_ann = os.path.join(saveDir, 'statsProp', 'annfiles')
        results_path_dataclasses = os.path.join(saveDir, 'statsProp', 'dataClasses')
    else:
        results_path = os.path.join(RESULTS_PATH, 'statsProp')
        results_path_ann = os.path.join(RESULTS_PATH, 'statsProp', 'annfiles')
        results_path_dataclasses = os.path.join(RESULTS_PATH, 'statsProp', 'dataClasses')

    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    saveFigure(savePath=os.path.join(results_path, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(results_path_ann, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(results_path_dataclasses, figure_name), kwargs['dataClasses'])

def generate_statistic_prop_normalized(classes=None, saveDir='',fileNamePrefix="", showPercent=False, **kwargs):
    """Generates a plot with average relative CAM Activations, the covered segment area as well as a normalized display 
    showing the CAM normliazed w.r.t the importance of the covered area of the segment.
    In a second plot the importance of CAM Activations w.r.t the covered segment area is shown.

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param showPercent: (default False) Determine if percentage labels will be shown on every bar.
    :param fileNamePrefix: Prefix that will be added in front of the filenames under which the results are saved.
    :param saveDir: Directory Path where the resulting files will be saved to. If not specified defaults to './results/'. Files will be saved 
                    under the given or default directory Path in a folder called 'normalized'

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

    imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

    _, _, percentualSegmentedCAMActivations,_, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)

    classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)

    relImportance, dominantMaskRelImportance, rescaledSummarizedPercActivions, dominantMaskRescaledActivations = get_area_normalized_stats(percentualActivations=summarizedPercSegmentedCAMActivations, percentualAreas=summarizedPercSegmentAreas)

    fig = plt.figure(figsize=(15,10),constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=2)

    numSamples = len(imgNames)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.set_title('Average relative CAM Activations')

    # Default width is 0.8 and since we are plotting three bars side by side avoiding overlap requires
    # reducing the width
    barwidth = 0.8 / 3

    bars = ax0.bar(np.arange(classArray.size), summarizedPercSegmentedCAMActivations, width=barwidth, label='CAM Activations', color='c')
    ax0.set_xticks([tick+barwidth for tick in range(classArray.size)], classArray)

    rotation = 90 if showPercent else 0

    # Format main Data with generated bar graph
    plot_bar(ax=ax0, bars=bars, dominantMask=dominantMaskPercentual, x_tick_labels=classArray, format='.1%', textadjust_ypos=showPercent,
        textrotation=rotation, hightlightColor='tab:blue')

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercSegmentAreas, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMaskPercentual, addText=showPercent, hightlightDominant=False,
        textadjust_ypos=showPercent, format='.1%', textrotation=rotation)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+2*barwidth, x_tick_labels=classArray, data=rescaledSummarizedPercActivions, barwidth=barwidth, barcolor='y',
        barlabel='Rescaled/Normalized Activation', dominantMask=dominantMaskRescaledActivations, addText=showPercent,
        textadjust_ypos=showPercent, format='.1%', textrotation=rotation, hightlightColor='tab:orange')

    ax0.text(0.9,1.02, f'No.Samples:{len(imgNames)}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

    legendMap = {
        'c':'CAM Activations',
        'tab:blue':'Top CAM Activations',
        'g':'Proportional Segment Coverage',
        'y':'Normalized Activation',
        'tab:orange':'Top Normalized Activations'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax0.legend(handles=handles)

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')


    ax1 = fig.add_subplot(grid[1,0])
    ax1.set_title('CAM Activation importance normalized by area')

    plot_bar(ax=ax1, x_ticks=classArray, data=relImportance, dominantMask=dominantMaskRelImportance, format='.1%')

    plt.show()

    figure_name, saveDataClasses, saveAnnfile = get_save_figure_name(additional='ShowPropArea_AreaNormalized', **kwargs)

    if saveDir:
        results_path = os.path.join(saveDir, 'normalized')
        results_path_ann = os.path.join(saveDir, 'normalized', 'annfiles')
        results_path_dataclasses = os.path.join(saveDir, 'normalized', 'dataClasses')
    else:
        results_path = os.path.join(RESULTS_PATH, 'normalized')
        results_path_ann = os.path.join(RESULTS_PATH, 'normalized', 'annfiles')
        results_path_dataclasses = os.path.join(RESULTS_PATH, 'normalized', 'dataClasses')

    if fileNamePrefix:
        figure_name = fileNamePrefix + "_" + figure_name

    saveFigure(savePath=os.path.join(results_path, figure_name), figure=fig)
    if saveAnnfile:
        copyFile(kwargs['annfile'], os.path.join(results_path_ann, figure_name))
    if saveDataClasses:
        writeArrayToFile(os.path.join(results_path_dataclasses, figure_name), kwargs['dataClasses'])

def generate_statistics_mean_variance(classes=None, saveDir='',fileNamePrefix="", usePercScale=False,  **kwargs):
    """Generates a plot showing the mean and variance of the activations within each segment category
    as well as in the totalCAM.

    :param classes: Classes that the segmentation model uses. If not specified it will be loaded from segConfig and segCheckpoint, defaults to None
    :param fileNamePrefix: Prefix that will be added in front of the filenames under which the results are saved.
    :param saveDir: Directory Path where the resulting files will be saved to. If not specified defaults to './results/'. Files will be saved 
                    under the given or default directory Path in a folder called 'normalized'
    :param usePercScale: Use Percentage Scale for first plot showing the mean and standard deviation of the CAM Activations per Segment. 
                    If False the absolut values will be used.

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
    imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

    totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations, segmentAreas, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)

    stats = generate_stats(classes=classes, segmentedActivations=segmentedCAMActivations, totalCAM=totalCAMActivations, percentualActivations=percentualSegmentedCAMActivations,
                          absoluteAreas=segmentAreas, percentualAreas=percentualSegmentAreas, get_std=True, get_total_mean=True, get_total_top_low_high=True)

    # Split the stats separatly to avoid mega crowding in one line
    classArray = stats[0]
    # sum, mean, std, lowestSampleIndices, lowestSamples, highestSampleIndices, highestSamples
    totalCAMStats = stats[1:8]
    summarizedSegmentCAMActivations = stats[8]
    #dominantMask = stats[9]
    stdSegmentCAMActivations = stats[10]
    summarizedPercSegmentCAMActivations = stats[11]
    #dominantMaskPerc = stats[12]
    stdPercSegmentCAMActivations = stats[13]
    summarizedSegmentAreas = stats[14]
    stdSegmentAreas = stats[15]
    summarizedPercSegmentAreas = stats[16]
    stdPercSegmentAreas = stats[17]

    fig = plt.figure(figsize=(15,20), constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=2)

    numSamples = len(imgNames)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

    if usePercScale:
        ax0.set_title('Percentual Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedPercSegmentCAMActivations, stdData=stdPercSegmentCAMActivations)
    else:
        ax0.set_title('Absolute Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedSegmentCAMActivations, stdData=stdSegmentCAMActivations)

    
    ax1 = fig.add_subplot(grid[1,0])
    ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)

    if usePercScale:
        ax1.set_title('Percentual Mean and Standard Deviation of Area of Segments')
        plot_errorbar(ax=ax1, x_ticks=classArray, meanData=summarizedPercSegmentAreas, stdData=stdPercSegmentAreas)
    else:
        ax1.set_title('Absolute Mean and Standard Deviation of Area of Segments')
        plot_errorbar(ax=ax1, x_ticks=classArray, meanData=summarizedSegmentAreas, stdData=stdSegmentAreas)


    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
    plt.show()