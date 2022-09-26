import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import numpy as np
import copy
from datetime import date

from .utils.io import get_samples, save_result_figure_data, save_excel_auto_name, load_results_excel
from .utils.prepareData import prepareInput, prepare_generate_stats
from .utils.calculations import generate_stats, accumulate_statistics, get_area_normalized_stats, get_top_k, accumulate_statistics_together
from .utils.plot import plot_bar, plot_errorbar
from .utils.model import get_wrongly_classified
from .utils.preprocessing import load_classes

from .utils.constants import EXCELCOLNAMESSTANDARD, EXCELCOLNAMESPROPORTIONAL, EXCELCOLNAMESNORMALIZED, EXCELCOLNAMESMEANSTDTOTAL, EXCELCOLNAMESMISSCLASSIFIED

def generate_statistic(classes=None, saveDir='', fileNamePrefix="" , results_file='', columnMap=EXCELCOLNAMESSTANDARD, filename='', saveFigureFormat='.jpg', sharedStats=None, numSamples=0, **kwargs):
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
    if results_file:
        print(f'Using given results from file {results_file}')
        assert set(EXCELCOLNAMESSTANDARD.keys()).issubset(set(columnMap.keys())), f'Not all required keys in columnMap. Required are: {",".join(list(EXCELCOLNAMESSTANDARD.keys()))}'
        classArray, loadedResults = load_results_excel(results_file, columnMap)
        summarizedSegmentedCAMActivations = loadedResults['summarizedSegmentedCAMActivations']
        summarizedPercSegmentedCAMActivations = loadedResults['summarizedPercSegmentedCAMActivations']
        totalActivation = loadedResults['totalActivation'][0] # Since totalActivation is only a single value and only the first is relevant.
        numSamples = loadedResults['numSamples'][0]

        dominantMask = get_top_k(summarizedSegmentedCAMActivations)
        dominantMaskPercentual = get_top_k(summarizedPercSegmentedCAMActivations)

    else:
        if sharedStats is not None:
            totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations, _, percentualSegmentAreas = sharedStats
        else:
            imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

            #totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, **kwargs)

            totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations = accumulate_statistics(imgNames=imgNames, classes=classes, cams=cams, segmentations=transformedSegmentations)
            numSamples = len(imgNames)

        classArray, totalActivation, summarizedSegmentedCAMActivations, dominantMask, summarizedPercSegmentedCAMActivations, dominantMaskPercentual = generate_stats(
            segmentedActivations=segmentedCAMActivations, percentualActivations=percentualSegmentedCAMActivations, totalCAM=totalCAMActivations, classes=classes)



    x_label_text = f'No.Samples:{numSamples}'

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


    #ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel(x_label_text)
    ax0.set_title('Average absolut CAM Activations')

    # Plot percentualSegmentedCAMActivations aka the relative of the absolute CAM Activations
    plot_bar(ax=ax1, x_ticks=classArray, data=summarizedPercSegmentedCAMActivations, dominantMask=dominantMaskPercentual, format='.1%')
    #ax1.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel(x_label_text)
    ax1.set_title('Average relative CAM Activations')

    if 'dataClasses' in kwargs:
        ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
        ax1.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')
    plt.show()

    # Here one has to ensure filename is set since we can not guaranteed infer the name from other parameters.
    if results_file and filename == '':
        warnings.simplefilter('always')
        warnings.warn('No filename is set. Using default: results')
        filename = 'results'

    save_result_figure_data(figure=fig, save_dir=saveDir, fileNamePrefix=fileNamePrefix, fileExtension=saveFigureFormat, fileName=filename, **kwargs)
    saveDic = {
        'RawActivations':summarizedSegmentedCAMActivations,
        'PercActivations' : summarizedPercSegmentedCAMActivations,
        'totalActivation' : [totalActivation],
        'numSamples': [numSamples],
    }
    save_excel_auto_name(saveDic, save_dir=saveDir, fileNamePrefix=fileNamePrefix, segments=classArray, fileName=filename, **kwargs)



def generate_statistic_prop(classes=None, saveDir='', fileNamePrefix="", showPropPercent=False, results_file='', columnMap=EXCELCOLNAMESPROPORTIONAL,
                            saveFigureFormat='.jpg', filename='', sharedStats=None, numSamples=0,**kwargs):
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
    if results_file:
        print(f'Using given results from file {results_file}')
        assert set(EXCELCOLNAMESPROPORTIONAL.keys()).issubset(set(columnMap.keys())), f'Not all required keys in columnMap. Required are: {",".join(list(EXCELCOLNAMESPROPORTIONAL.keys()))}'
        classArray, loadedResults = load_results_excel(results_file, columnMap)
        summarizedPercSegmentedCAMActivations = loadedResults['PercActivations']
        summarizedPercSegmentAreas = loadedResults['PercSegmentAreas']

        dominantMaskPercentual = get_top_k(summarizedPercSegmentedCAMActivations)

        x_label_text = f'No. samples:unknown (Data from file)'
    else:
        if sharedStats is not None:
            _, _, percentualSegmentedCAMActivations,_, percentualSegmentAreas = sharedStats
        else:
            imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

            _, _, percentualSegmentedCAMActivations,_, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)
            numSamples = len(imgNames)
        #_, _, percentualSegmentedCAMActivations, percentualSegmentAreas =  batch_statistics(classes=classes, imgNames=imgNames, cams=cams, segmentations=transformedSegmentations,percentualArea=True ,**kwargs)  # forceAll can be set in kwargs if desired


        classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)
        

        x_label_text = f'No.Samples:{numSamples}'

    fig = plt.figure(figsize=(15,5),constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=1)

    ax0 = fig.add_subplot(grid[0,0])
    #ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel(x_label_text)
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

    # Here one has to ensure filename is set since we can not guaranteed infer the name from other parameters.
    if results_file and filename == '':
        warnings.simplefilter('always')
        warnings.warn('No filename is set. Using default: resultsProportional')
        filename = 'resultsProportional'

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='statsProp', fileNamePrefix=fileNamePrefix, fileExtension=saveFigureFormat, fileName=filename, **kwargs)
    saveDic = {
        'PercActivations' : summarizedPercSegmentedCAMActivations,
        'PercSegmentAreas' : summarizedPercSegmentAreas
    }
    save_excel_auto_name(saveDic, fileNamePrefix=fileNamePrefix + 'prop', save_dir=saveDir, path_intermediate='statsProp', segments=classArray, fileName=filename, **kwargs)

def generate_statistic_prop_normalized(classes=None, saveDir='',fileNamePrefix="", showPercent=False, results_file='', columnMap=EXCELCOLNAMESNORMALIZED,
                                        saveFigureFormat='.jpg', filename='', sharedStats=None, numSamples=0,**kwargs):
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
    if results_file:
        print(f'Using given results from file {results_file}')
        assert set(EXCELCOLNAMESNORMALIZED.keys()).issubset(set(columnMap.keys())), f'Not all required keys in columnMap. Required are: {",".join(list(EXCELCOLNAMESNORMALIZED.keys()))}'
        classArray, loadedResults = load_results_excel(results_file, columnMap)
        summarizedPercSegmentedCAMActivations = loadedResults['summarizedPercSegmentedCAMActivations']
        summarizedPercSegmentAreas = loadedResults['summarizedPercSegmentAreas']
        relImportance = loadedResults['relImportance']
        rescaledSummarizedPercActivions = loadedResults['rescaledSummarizedPercActivions']

        dominantMaskPercentual = get_top_k(summarizedPercSegmentedCAMActivations)
        dominantMaskRelImportance = get_top_k(relImportance)
        dominantMaskRescaledActivations = get_top_k(rescaledSummarizedPercActivions)

        x_label_text = f'No. samples:unknown (Data from file)'
    else:
        if sharedStats is not None:
            _, _, percentualSegmentedCAMActivations, _, percentualSegmentAreas = sharedStats
        else:

            imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

            _, _, percentualSegmentedCAMActivations,_, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)

            numSamples = len(imgNames)

        classArray, summarizedPercSegmentedCAMActivations, dominantMaskPercentual, summarizedPercSegmentAreas = generate_stats(classes=classes, percentualActivations=percentualSegmentedCAMActivations,percentualAreas=percentualSegmentAreas)

        relImportance, dominantMaskRelImportance, rescaledSummarizedPercActivions, dominantMaskRescaledActivations = get_area_normalized_stats(percentualActivations=summarizedPercSegmentedCAMActivations, percentualAreas=summarizedPercSegmentAreas)


        x_label_text = f'No.Samples:{numSamples}'

    fig = plt.figure(figsize=(15,10),constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=2)

    ax0 = fig.add_subplot(grid[0,0])
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel(x_label_text)
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

    # Here one has to ensure filename is set since we can not guaranteed infer the name from other parameters.
    if results_file and filename == '':
        warnings.simplefilter('always')
        warnings.warn('No filename is set. Using default: resultsNormalized')
        filename = 'resultsNormalized'

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='normalized', fileNamePrefix=fileNamePrefix, fileExtension=saveFigureFormat, fileName=filename, **kwargs)
    saveDic = {
        'PercActivations' : summarizedPercSegmentedCAMActivations,
        'PercSegmentAreas' : summarizedPercSegmentAreas,
        'RelativeCAMImportance':relImportance,
        'PercActivationsRescaled':rescaledSummarizedPercActivions
    }
    save_excel_auto_name(saveDic, fileNamePrefix=fileNamePrefix + 'normalized', save_dir=saveDir, path_intermediate='normalized', segments=classArray, fileName=filename, **kwargs)
    

def generate_statistics_mean_variance_total(classes=None, saveDir='',fileNamePrefix="", usePercScale=False, results_file='', columnMap=EXCELCOLNAMESMEANSTDTOTAL,
                                            saveFigureFormat='.jpg', filename='', sharedStats=None, numSamples=0, pregenImgNames=None, **kwargs):
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
    if results_file:
        print(f'Using given results from file {results_file}')
        assert set(EXCELCOLNAMESMEANSTDTOTAL.keys()).issubset(set(columnMap.keys())), f'Not all required keys in columnMap. Required are: {",".join(list(EXCELCOLNAMESMEANSTDTOTAL.keys()))}'
        classArray, loadedResults = load_results_excel(results_file, columnMap)
        summarizedPercSegmentCAMActivations = loadedResults['PercActivations']
        stdPercSegmentCAMActivations = loadedResults['PercActivationsStd']
        summarizedSegmentCAMActivations = loadedResults['RawActivations']
        stdSegmentCAMActivations = loadedResults['RawActivationsStd']
        summarizedPercSegmentAreas = loadedResults['PercSegmentAreas']
        stdPercSegmentAreas = loadedResults['PercSegmentAreasStd']
        summarizedSegmentAreas = loadedResults['RawSegmentAreas']
        stdSegmentAreas = loadedResults['RawSegmentAreasStd']
        totalMean = loadedResults['totalMean'][0]
        totalStd = loadedResults['totalStd'][0]

        x_label_text = f'No. samples:unknown (Data from file)'
    else:
        if sharedStats is not None:
            totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations, segmentAreas, percentualSegmentAreas = sharedStats
            assert pregenImgNames is not None
            imgNames = pregenImgNames
        else:
            imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, **kwargs)

            totalCAMActivations, segmentedCAMActivations, percentualSegmentedCAMActivations, segmentAreas, percentualSegmentAreas = accumulate_statistics(imgNames=imgNames, cams=cams, segmentations=transformedSegmentations, classes=classes, percentualArea=True)

            numSamples = len(imgNames)
            
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
        

        x_label_text = f'No.Samples:{numSamples}'

    fig = plt.figure(figsize=(15,25), constrained_layout=True)
    grid = fig.add_gridspec(ncols=1, nrows=5)


    ax0 = fig.add_subplot(grid[0:2,0])
    ax0.xaxis.set_label_position('top')
    ax0.set_xlabel(x_label_text)

    if usePercScale:
        ax0.set_title('Percentual Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedPercSegmentCAMActivations, stdData=stdPercSegmentCAMActivations)
    else:
        ax0.set_title('Absolute Mean and Standard Deviation of Each Segment Category')
        plot_errorbar(ax=ax0, x_ticks=classArray, meanData=summarizedSegmentCAMActivations, stdData=stdSegmentCAMActivations)

    
    ax1 = fig.add_subplot(grid[2:4,0])
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel(x_label_text)

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

    # Here one has to ensure filename is set since we can not guaranteed infer the name from other parameters.
    if results_file and filename == '':
        warnings.simplefilter('always')
        warnings.warn('No filename is set. Using default: resultsMeanStdTotal')
        filename = 'resultsMeanStdTotal'

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='meanStdTotal', fileNamePrefix=fileNamePrefix, fileExtension=saveFigureFormat, fileName=filename, **kwargs)
    saveDic = {
        'PercActivations' : summarizedPercSegmentCAMActivations,
        'PercActivationsStd':stdPercSegmentCAMActivations,
        'RawActivations':summarizedSegmentCAMActivations,
        'RawActivationsStd':stdSegmentCAMActivations,
        'PercSegmentAreas' : summarizedPercSegmentAreas,
        'PercSegmentAreasStd':stdPercSegmentAreas,
        'RawSegmentAreas':summarizedSegmentAreas,
        'RawSegmentAreasStd':stdSegmentAreas,
        'totalMean':[totalMean],
        'totalStd':[totalStd]
    }
    save_excel_auto_name(saveDic, fileNamePrefix=fileNamePrefix + 'meanStdTotal', save_dir=saveDir, path_intermediate='meanStdTotal', segments=classArray, fileName=filename, **kwargs)

def generate_statistics_missclassified(imgRoot="", annfile="", method="gradcam", camConfig="", camCheckpoint="", saveDir='', fileNamePrefix="", classes=None,
                                        annfileCorrect="", annfileIncorrect="", results_file='', columnMap=EXCELCOLNAMESMISSCLASSIFIED, filename='',
                                        saveFigureFormat='.jpg',sharedStats=None, numSamples=0,preGenAllImgNames=None, preGenAllTransformedSegs=None,**kwargs):
    """
    Generates plots showing the activations for only the correctly classified samples for the given dataset,
    A plot showing the activations of the wrongly classified samples.
    """
    if results_file:
        print(f'Using given results from file {results_file}')
        assert set(EXCELCOLNAMESMISSCLASSIFIED.keys()).issubset(set(columnMap.keys())), f'Not all required keys in columnMap. Required are: {",".join(list(EXCELCOLNAMESMISSCLASSIFIED.keys()))}'
        classArray, loadedResults = load_results_excel(results_file, columnMap)
        summarizedPercCAMActivationsOriginal = loadedResults['summarizedPercCAMActivationsOriginal']
        summarizedPercCAMActivationsCorrect = loadedResults['summarizedPercCAMActivationsCorrect']
        summarizedPercCAMActivationsIncorrect = loadedResults['summarizedPercCAMActivationsIncorrect']
        summarizedPercCAMActivationsCorrected = loadedResults['summarizedPercCAMActivationsCorrected']
        summarizedPercCAMActivationsFixed = loadedResults['summarizedPercCAMActivationsFixed']

        dominantMaskPercCorrect = get_top_k(summarizedPercCAMActivationsCorrect)
        dominantMaskPercIncorret = get_top_k(summarizedPercCAMActivationsIncorrect)
        dominantMaskPercCorrected = get_top_k(summarizedPercCAMActivationsCorrected)

        x_label_text_original = f'No. samples:unknown (Data from file)'
        x_label_text_correct = f'No. samples:unknown (Data from file)'
        x_label_text_incorrect = f'No. samples:unknown (Data from file)'
    else:
        assert os.path.isdir(imgRoot), f'imgRoot does not lead to a directory {imgRoot}'
        assert os.path.isfile(annfile), f'No such file {annfile}'
        assert os.path.isfile(camConfig), f'No such file {camConfig}'
        assert os.path.isfile(camCheckpoint), f'No such file {camCheckpoint}'

        if 'imgNames' in kwargs:
            imgNames = kwargs['imgNames']
        else:
            imgNames = get_samples(annfile=annfile, imgRoot=imgRoot,**kwargs)

        if len(imgNames) == 0:
            raise ValueError('Given parameters do not yield any images.')

        # Only generate files if we don't have path already specified.
        if annfileCorrect == "" or annfileIncorrect == "":
            annfileCorrect, annfileIncorrect =  get_wrongly_classified(imgRoot=imgRoot, annfile=annfile, imgNames=imgNames, 
                                                                camConfig=camConfig, camCheckpoint=camCheckpoint, saveDir=saveDir, **kwargs)
        else:
            print(f'Using provided annfile correct:{annfileCorrect}, incorrect: {annfileIncorrect}')

        kwargsCorrected = copy.copy(kwargs)
        kwargsCorrected['camData'] = None # Set camData to none so that it must generate new cams

        if sharedStats is not None:
            numSamplesOriginal = numSamples
            _, _, percentualCAMActivationsOriginal, _, _ = sharedStats
            imgNamesOriginal = preGenAllImgNames
            transformedSegmentationsOriginal = preGenAllTransformedSegs
        else:
            imgNamesOriginal, transformedSegmentationsOriginal, camsOriginal, classes = prepare_generate_stats(
                classes=classes, imgRoot=imgRoot, annfile=annfile, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
            numSamplesOriginal = len(imgNamesOriginal)
            _, _, percentualCAMActivationsOriginal = accumulate_statistics(imgNames=imgNamesOriginal, classes=classes, cams=camsOriginal, segmentations=transformedSegmentationsOriginal)


        # imgNamesCorrect, transformedSegmentationsCorrect, camsCorrect, _ = prepare_generate_stats(
        #     classes=classes, imgRoot=imgRoot, annfile=annfileCorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
        # imgNamesIncorrect, transformedSegmentationsIncorrect, camsIncorrect, _ = prepare_generate_stats(
        #     classes=classes, imgRoot=imgRoot, annfile=annfileIncorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
        imgNamesCorrect, _, camsCorrect, _ = prepare_generate_stats(
            classes=classes, imgRoot=imgRoot, annfile=annfileCorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
        imgNamesIncorrect, _, camsIncorrect, _ = prepare_generate_stats(
            classes=classes, imgRoot=imgRoot, annfile=annfileIncorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargs)
        # Index here at the end because we get a list as return value
        camsCorrected = prepareInput(prepImg=False, prepSeg=False, prepCam=True, imgRoot=imgRoot, useAnnLabels=True,
                        annfile=annfileIncorrect, method=method, camConfig=camConfig, camCheckpoint=camCheckpoint, **kwargsCorrected)[0]


        # _, _, percentualCAMActivationsCorrect = accumulate_statistics(imgNames=imgNamesCorrect, classes=classes, cams=camsCorrect, segmentations=transformedSegmentationsCorrect)
        # _, _, percentualCAMActivationsIncorrect = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsIncorrect, segmentations=transformedSegmentationsIncorrect)
        # _, _, percentualCAMActivationsCorrected = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsCorrected, segmentations=transformedSegmentationsIncorrect)

        # _, _, percentualCAMActivationsCorrect = accumulate_statistics(imgNames=imgNamesCorrect, classes=classes, cams=camsCorrect, segmentations=transformedSegmentationsOriginal)
        # _, _, percentualCAMActivationsIncorrect = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsIncorrect, segmentations=transformedSegmentationsOriginal)
        # _, _, percentualCAMActivationsCorrected = accumulate_statistics(imgNames=imgNamesIncorrect, classes=classes, cams=camsCorrected, segmentations=transformedSegmentationsOriginal)

        collectedResults = accumulate_statistics_together(allImgNames=imgNamesOriginal, imgNamesList=[imgNamesCorrect,imgNamesIncorrect,imgNamesIncorrect], 
                                                            camsList=[camsCorrect,camsIncorrect,camsCorrected], segmentations=transformedSegmentationsOriginal, classes=classes)
  
        _, _, percentualCAMActivationsCorrect = collectedResults[0]   
        _, _, percentualCAMActivationsIncorrect = collectedResults[1]    
        _, _, percentualCAMActivationsCorrected = collectedResults[2]   

        percentualCAMActivationsFixed = np.concatenate((percentualCAMActivationsCorrect, percentualCAMActivationsCorrected))

        classArray, summarizedPercCAMActivationsOriginal, _ = generate_stats(percentualActivations=percentualCAMActivationsOriginal, classes=classes)
        _, summarizedPercCAMActivationsCorrect, dominantMaskPercCorrect = generate_stats(percentualActivations=percentualCAMActivationsCorrect, classes=classes)
        _, summarizedPercCAMActivationsIncorrect, dominantMaskPercIncorret = generate_stats(percentualActivations=percentualCAMActivationsIncorrect, classes=classes)
        _, summarizedPercCAMActivationsCorrected, dominantMaskPercCorrected = generate_stats(percentualActivations=percentualCAMActivationsCorrected, classes=classes)
        _, summarizedPercCAMActivationsFixed, _ = generate_stats(percentualActivations=percentualCAMActivationsFixed, classes=classes)

        numSamplesCorrect = len(imgNamesCorrect)
        numSamplesIncorrect = len(imgNamesIncorrect)

        x_label_text_original = f'No. samples:{numSamplesOriginal}'
        x_label_text_correct = f'No. samples:{numSamplesCorrect}'
        x_label_text_incorrect = f'No. samples:{numSamplesIncorrect}'

    fig = plt.figure(figsize=(15,10),constrained_layout=True)
    grid = fig.add_gridspec(ncols=3, nrows=2)

    axCorrect = fig.add_subplot(grid[0,0]) # Only correct
    axIncorrect = fig.add_subplot(grid[0,1]) # Only incorrect
    axCorrected = fig.add_subplot(grid[0,2]) # Only corrected
    axCompare = fig.add_subplot(grid[1,:]) # Compare original and fixed

    axCorrect.xaxis.set_label_position('top')
    axCorrect.set_xlabel(x_label_text_correct)
    axCorrect.set_title('Correct samples average CAM Activations')

    plot_bar(ax=axCorrect, x_ticks=classArray, data=summarizedPercCAMActivationsCorrect, dominantMask=dominantMaskPercCorrect, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axIncorrect.xaxis.set_label_position('top')
    axIncorrect.set_xlabel(x_label_text_incorrect)
    axIncorrect.set_title('Wrongly classified samples average CAM Activations')

    plot_bar(ax=axIncorrect, x_ticks=classArray, data=summarizedPercCAMActivationsIncorrect, dominantMask=dominantMaskPercIncorret, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axCorrected.xaxis.set_label_position('top')
    axCorrected.set_xlabel(x_label_text_incorrect)
    axCorrected.set_title('Wrongly classified corrected average CAM Activations')

    plot_bar(ax=axCorrected, x_ticks=classArray, data=summarizedPercCAMActivationsCorrected, dominantMask=dominantMaskPercCorrected, 
            textadjust_ypos=True, format='.1%', textrotation=90, increase_ylim_scale=1.2)

    axCompare.xaxis.set_label_position('top')
    axCompare.set_xlabel(x_label_text_original)
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
    
    # Here one has to ensure filename is set since we can not guaranteed infer the name from other parameters.
    if results_file and filename == '':
        warnings.simplefilter('always')
        warnings.warn('No filename is set. Using default: resultsMissclassified')
        filename = 'resultsMissclassified'

    save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='wronglyClassifications', fileExtension=saveFigureFormat, fileNamePrefix=fileNamePrefix, fileName=filename, **kwargs)
    saveDic = {
        'PercActivationsOriginal':summarizedPercCAMActivationsOriginal,
        'PercActivationsCorrect' : summarizedPercCAMActivationsCorrect,
        'PercActivationsIncorrect' : summarizedPercCAMActivationsIncorrect,
        'PercActivationsCorrected' : summarizedPercCAMActivationsCorrected,
        'PercActivationsFixed' : summarizedPercCAMActivationsFixed,
    }
    save_excel_auto_name(saveDic, fileNamePrefix=fileNamePrefix + 'wronglyClassified', save_dir=saveDir, path_intermediate='wronglyClassifications', segments=classArray, 
                        camConfig=camConfig, camCheckpoint=camCheckpoint, annfile=annfile, fileName=filename, **kwargs)


def generate_statistic_collection(imgRoot, classifierConfig, classifierCheckpoint, camData,
                                segConfig, segCheckpoint, segData, saveDir, annfile, method, vitLike,
                                modelName, prefix='Full', segModelName='ocrnet_hr48_carparts_noflip',
                                useGPU=True, **kwargs):
    """
    This function generates a collection of all statistics for the given model and dataset.
    All required data must already be generated and given i.e. cams and segmentations.

    Generated statistics are:
        generate_statistics
        generate_statistic_prop
        generate_statistic_prop_normalized
        generate_statistics_mean_variance_total (without percScale)
        generate_statistics_mean_variance_total (with percScale)
        generate_statistics_missclassified


    kwargs:
        annfile
        dataClasses
    """
    print(f'Creating all statistics and saving into {saveDir}')
    if useGPU:
        kwargs['camDevice']='cuda'
    # Load classes one time and pass it to functions in order to not always initialze segmentation model
    classes = load_classes(segConfig=segConfig, segCheckpoint=segCheckpoint)

    # Prefix_method_Model_Dataset_SegsConfig_date
    baseName = f'{prefix}_{method}_{modelName}_{segModelName}'

    # From kwargs possibly relevant here:
    # annfileCorrect, annfileIncorrect (generate_statistics_missclassified)
    # dataClasses (get_samples)
    # pipelineCfg (get_pipeline_cfg)
    # segDevice, dataClasses (prepareSegmentation)
    # camDevice, dataClasses (prepareCams)

    imgNames, transformedSegmentations, cams, classes = prepare_generate_stats(classes=classes, imgRoot=imgRoot, segData=segData, camData=camData, 
                                                                                segCheckpoint=segCheckpoint, segConfig=segConfig, method=method, annfile=annfile, 
                                                                                vitLike=vitLike, useGPU=useGPU, **kwargs)
    numSamples = len(imgNames)
    sharedStats = accumulate_statistics(imgNames=imgNames, classes=classes, cams=cams, segmentations=transformedSegmentations, percentualArea=True)

    fileName = baseName + "_" + date.today().strftime("%d_%m_%Y")
    generate_statistic(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                        camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                        segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                        method=method, vitLike=vitLike, filename=fileName,
                        saveAdditional=False, saveFigureFormat='.pdf',
                        numSamples=numSamples, sharedStats=sharedStats, **kwargs)
    fileName = baseName + "_ShowPropArea_" + date.today().strftime("%d_%m_%Y")
    generate_statistic_prop(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                            camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                            segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                            method=method, vitLike=vitLike, showPropPercent=True, filename=fileName,
                            saveAdditional=False, saveFigureFormat='.pdf', 
                            numSamples=numSamples, sharedStats=sharedStats, **kwargs)
    fileName = baseName + "_normalized_PropArea_" + date.today().strftime("%d_%m_%Y")
    generate_statistic_prop_normalized(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                                        camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                                        segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                                        method=method, vitLike=vitLike, showPercent=True, filename=fileName,
                                        saveAdditional=False, saveFigureFormat='.pdf', 
                                        numSamples=numSamples, sharedStats=sharedStats, **kwargs)
    fileName = baseName + "_Abs_Mean_Std_Total_" + date.today().strftime("%d_%m_%Y")
    generate_statistics_mean_variance_total(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                                            camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                                            segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                                            method=method, vitLike=vitLike, filename=fileName,
                                            saveAdditional=False, saveFigureFormat='.pdf', 
                                            numSamples=numSamples, sharedStats=sharedStats, pregenImgNames=imgNames, **kwargs)
    fileName = baseName + "_Perc_Mean_Std_Total_" + date.today().strftime("%d_%m_%Y")
    generate_statistics_mean_variance_total(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                                            camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                                            segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                                            method=method, vitLike=vitLike, usePercScale=True, filename=fileName,
                                            saveAdditional=False, saveFigureFormat='.pdf', 
                                            numSamples=numSamples, sharedStats=sharedStats, pregenImgNames=imgNames, **kwargs)
    fileName = baseName + "_wrongClassified_" + date.today().strftime("%d_%m_%Y")
    generate_statistics_missclassified(classes=classes, saveDir=saveDir, imgRoot=imgRoot, annfile=annfile,
                                        camConfig=classifierConfig, camCheckpoint=classifierCheckpoint, camData=camData,
                                        segConfig=segConfig, segCheckpoint=segCheckpoint, segData=segData, 
                                        method=method, vitLike=vitLike, filename=fileName,
                                        saveAdditional=False, saveFigureFormat='.pdf', 
                                        numSamples=numSamples, sharedStats=sharedStats, preGenAllImgNames=imgNames, 
                                        preGenAllTransformedSegs=transformedSegmentations, **kwargs)
