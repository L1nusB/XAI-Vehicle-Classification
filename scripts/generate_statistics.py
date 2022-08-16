import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np
import copy

from .utils.io import get_samples, saveFigure, get_save_figure_name, copyFile, writeArrayToFile, save_result_figure_data
from .utils.prepareData import prepareInput, get_pipeline_cfg, prepare_generate_stats
from .utils.pipeline import get_pipeline_torchvision
from .utils.calculations import generate_stats, accumulate_statistics, get_area_normalized_stats
from .utils.plot import plot_bar, plot_errorbar
from .utils.preprocessing import load_classes
from .utils.model import get_wrongly_classified

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

    save_result_figure_data(figure=fig, save_dir=saveDir, fileNamePrefix=fileNamePrefix, **kwargs)




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
        textrotation=rotation, keep_x_ticks=True)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercSegmentAreas, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMaskPercentual, addText=showPropPercent, hightlightDominant=False,
        textadjust_ypos=showPropPercent, format='.1%', textrotation=rotation, keep_x_ticks=True)

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

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='statsProp', fileNamePrefix=fileNamePrefix, **kwargs)

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
        textrotation=rotation, hightlightColor='tab:blue', keep_x_ticks=True)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercSegmentAreas, barwidth=barwidth, barcolor='g',
        barlabel='Proportional Segment Coverage', dominantMask=dominantMaskPercentual, addText=showPercent, hightlightDominant=False,
        textadjust_ypos=showPercent, format='.1%', textrotation=rotation, keep_x_ticks=True)

    # Plot proportion Data next to main Data
    plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+2*barwidth, x_tick_labels=classArray, data=rescaledSummarizedPercActivions, barwidth=barwidth, barcolor='y',
        barlabel='Rescaled/Normalized Activation', dominantMask=dominantMaskRescaledActivations, addText=showPercent,
        textadjust_ypos=showPercent, format='.1%', textrotation=rotation, hightlightColor='tab:orange', keep_x_ticks=True)

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

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='normalized', fileNamePrefix=fileNamePrefix, **kwargs)

def generate_statistics_mean_variance_total(classes=None, saveDir='',fileNamePrefix="", usePercScale=False,  **kwargs):
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

    fig = plt.figure(figsize=(15,25), constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=5)

    numSamples = len(imgNames)

    ax0 = fig.add_subplot(grid[0:2,0])
    ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

    if usePercScale:
        ax0.set_title('Percentual Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedPercSegmentCAMActivations, stdData=stdPercSegmentCAMActivations)
    else:
        ax0.set_title('Absolute Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedSegmentCAMActivations, stdData=stdSegmentCAMActivations)

    
    ax1 = fig.add_subplot(grid[2:4,0])
    ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)

    if usePercScale:
        ax1.set_title('Percentual Mean and Standard Deviation of Area of Segments')
        plot_errorbar(ax=ax1, x_ticks=classArray, meanData=summarizedPercSegmentAreas, stdData=stdPercSegmentAreas)
    else:
        ax1.set_title('Absolute Mean and Standard Deviation of Area of Segments')
        plot_errorbar(ax=ax1, x_ticks=classArray, meanData=summarizedSegmentAreas, stdData=stdSegmentAreas)


    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
        ax1.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')

    ax2 = fig.add_subplot(grid[4,0])
    ax2.set_title('Total CAM Stats')

    totalMean = totalCAMStats[1]
    totalStd = totalCAMStats[2]
    plot_bar(ax=ax2, x_ticks=[0], data=[totalMean], format='.2f', barcolor='tab:blue', baryerr=[totalStd], barcapsize=4, barwidth=0.2, addText=False, barecolor='r')
    ax2.text(0.3, totalMean, f'\u03BC={totalMean:.2f} \n \u03C3={totalStd:.2f}', ha='center', va='bottom')
    plot_bar(ax=ax2, x_ticks=[1,2,3], data=totalCAMStats[4], format='.2f', barcolor='tab:orange')
    plot_bar(ax=ax2, x_ticks=[4,5,6], x_tick_labels=['Mean + Variance'] + [imgNames[i] for i in totalCAMStats[3]] + [imgNames[i] for i in totalCAMStats[5]], data=totalCAMStats[6], format='.2f', barcolor='tab:green')

    legendMap = {
        'tab:blue':'Mean totalCAM',
        'tab:orange':'Lowest 3 totalCAMs',
        'tab:green':'Highest 3 totalCAMs',
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

    ax2.legend(handles=handles, bbox_to_anchor=(1,1.04), loc="lower right")
    plt.show()

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='meanStdTotal', fileNamePrefix=fileNamePrefix, **kwargs)

def generate_statistics_missclassified(imgRoot, annfile, method, camConfig, camCheckpoint, saveDir='', fileNamePrefix="", **kwargs):
    """
    Generates plots showing the activations for only the correctly classified samples for the given dataset,
    A plot showing the activations of the wrongly classified samples.
    """
    assert os.path.isdir(imgRoot), f'imgRoot does not lead to a directory {imgRoot}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        imgNames = get_samples(annfile=annfile, imgRoot=imgRoot,**kwargs)

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    annfileCorrect, annfileIncorrect =  get_wrongly_classified(imgRoot=imgRoot, annfile=annfile, imgNames=imgNames, 
                                                        camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)

    kwargsCorrected = copy.copy(kwargs)
    kwargsCorrected['camData'] = None # Set camData to none so that it must generate new cams

    imgNamesOriginal, transformedSegmentationsOriginal, camsOriginal, classes = prepare_generate_stats(
        imgRoot=imgRoot, annfile=annfile, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
    imgNamesCorrect, transformedSegmentationsCorrect, camsCorrect, _ = prepare_generate_stats(
        imgRoot=imgRoot, annfile=annfileCorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
    imgNamesIncorrect, transformedSegmentationsIncorrect, camsIncorrect, _ = prepare_generate_stats(
        imgRoot=imgRoot, annfile=annfileIncorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
    # Index here at the end because we get a list as return value
    camsCorrected = prepareInput(prepImg=False, prepSeg=False, prepCam=True, imgRoot=imgRoot, useAnnLabels=True,
                    annfile=annfileIncorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargsCorrected)[0]

    _, _, percentualCAMActivationsOriginal = accumulate_statistics(imgNames=imgNamesOriginal, classes=classes, cams=camsOriginal, segmentations=transformedSegmentationsOriginal)
    _, _, percentualCAMActivationsCorrect = accumulate_statistics(imgNames=imgNamesCorrect, classes=classes, cams=camsCorrect, segmentations=transformedSegmentationsCorrect)
    _, _, percentualCAMActivationsIncorrect = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsIncorrect, segmentations=transformedSegmentationsIncorrect)
    _, _, percentualCAMActivationsCorrected = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsCorrected, segmentations=transformedSegmentationsIncorrect)

    percentualCAMActivationsFixed = np.concatenate((percentualCAMActivationsCorrect, percentualCAMActivationsCorrected))

    classArray, summarizedPercCAMActivationsOriginal, _ = generate_stats(percentualActivations=percentualCAMActivationsOriginal, classes=classes)
    _, summarizedPercCAMActivationsCorrect, dominantMaskPercCorrect = generate_stats(percentualActivations=percentualCAMActivationsCorrect, classes=classes)
    _, summarizedPercCAMActivationsIncorrect, dominantMaskPercIncorret = generate_stats(percentualActivations=percentualCAMActivationsIncorrect, classes=classes)
    _, summarizedPercCAMActivationsCorrected, dominantMaskPercCorrected = generate_stats(percentualActivations=percentualCAMActivationsCorrected, classes=classes)
    _, summarizedPercCAMActivationsFixed, _ = generate_stats(percentualActivations=percentualCAMActivationsFixed, classes=classes)

    numSamplesOriginal = len(imgNamesOriginal)
    numSamplesCorrect = len(imgNamesCorrect)
    numSamplesIncorrect = len(imgNamesIncorrect)

    fig = plt.figure(figsize=(15,10),constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=2)

    axCorrect = fig.add_subplot(grid[0,0]) # Only correct
    axIncorrect = fig.add_subplot(grid[0,1]) # Only incorrect
    axCorrected = fig.add_subplot(grid[0,2]) # Only corrected
    axCompare = fig.add_subplot(grid[1,:]) # Compare original and fixed

    axCorrect.xaxis.set_label_position('top')
    axCorrect.set_xlabel(f'No.Samples:{numSamplesCorrect}')
    axCorrect.set_title('Correct samples average CAM Activations')

    plot_bar(ax=axCorrect, x_ticks=classArray, data=summarizedPercCAMActivationsCorrect, dominantMask=dominantMaskPercCorrect, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axIncorrect.xaxis.set_label_position('top')
    axIncorrect.set_xlabel(f'No.Samples:{numSamplesIncorrect}')
    axIncorrect.set_title('Wrongly classified samples average CAM Activations')

    plot_bar(ax=axIncorrect, x_ticks=classArray, data=summarizedPercCAMActivationsIncorrect, dominantMask=dominantMaskPercIncorret, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axCorrected.xaxis.set_label_position('top')
    axCorrected.set_xlabel(f'No.Samples:{numSamplesIncorrect}')
    axCorrected.set_title('Wrongly classified corrected average CAM Activations')

    plot_bar(ax=axCorrected, x_ticks=classArray, data=summarizedPercCAMActivationsCorrected, dominantMask=dominantMaskPercCorrected, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axCompare.text(0.9,1.02, f'No.Samples:{numSamplesOriginal}',horizontalalignment='center',verticalalignment='center',transform = axCompare.transAxes)
    axCompare.set_title('Original and fixed classification average CAM Activations')

    barwidth = 0.4
    bars = axCompare.bar(np.arange(classArray.size), summarizedPercCAMActivationsOriginal, width=barwidth)
    axCompare.set_xticks([tick+barwidth/2 for tick in range(classArray.size)], classArray)

    # Format orginal data
    plot_bar(ax=axCompare, bars=bars, x_tick_labels=classArray, format='.1%', textadjust_ypos=True,
        textrotation=90, keep_x_ticks=True, hightlightDominant=False)

    # Plot Fixed activation
    plot_bar(ax=axCompare, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercCAMActivationsFixed, 
        barwidth=barwidth, barcolor='g', addText=True, hightlightDominant=False,
        textadjust_ypos=True, format='.1%', textrotation=90, keep_x_ticks=True, increase_ylim_scale=1.2)

    legendMap = {
        'tab:blue':'CAM Activations',
        'tab:red':'Top CAM Activations'
    }
    legendMapCompare = {
        'tab:blue':'Original CAM Activations',
        'tab:green':'Fixed CAM Activations'
    }
    handles = [Patch(color=k, label=v) for k,v in legendMap.items()]
    handlesCompare = [Patch(color=k, label=v) for k,v in legendMapCompare.items()]

    axCorrect.legend(handles=handles)
    axIncorrect.legend(handles=handles)
    axCorrected.legend(handles=handles)
    axCompare.legend(handles=handlesCompare)

    plt.show()

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='wronglyClassifications', fileNamePrefix=fileNamePrefix, **kwargs)