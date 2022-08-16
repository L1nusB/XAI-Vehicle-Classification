import warnings
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .generate_cams import generate_cam_overlay
from .utils.io import get_samples, savePIL, save_result_figure_data
from .utils.prepareData import prepareInput, get_pipeline_cfg
from .utils.imageProcessing import add_text, concatenate_images, convert_numpy_to_PIL
from .utils.pipeline import get_pipeline_torchvision
from .utils.preprocessing import load_classes
from .utils.calculations import accumulate_statistics, generate_stats
from .utils.plot import plot_bar

VISUALIZE_TYPES=['none','heatmap','overlay', 'both']

def visualize_cpu_vs_gpu(saveDir='', visType='overlay', fileNamePrefix="", plot=False,**kwargs):
    """Create images showing the CAM results for CPU and GPU comparing them.
    The type can be the raw CAM Heatmap or the CAM Overlay over the original image

    :param saveDir: Directory where the images will be saved to. It will be appended by '/imgs/'. Plot will be saved into saveDir.
    :type saveDir: str
    :param visType: Type of the visualizations images. Must be included in VISUALIZE_TYPES (Currently ['heatmap','overlay', 'both'])
    :type visType: str
    :param plot: Create a plot that compares the cam Activations between CPU and GPU and the totalCAMs (default False)
    :type plot: bool
    :param fileNamePrefix: Prefix that will be added to a plot and corresponding files if specified.
    :type fileNamePrefix: str

    The kwargs for the CAMs are specified by prefixing them with 'gpu' or 'cpu' respectively
    Relevant kwargs are:
    imgRoot: Path to root folder where images/samples lie
    (cpu/gpu)camConfig: Path to config of the used CAM Model
    (cpu/gpu)camCheckpoint: Path to Checkpoint of the used CAM Model
    (cpu/gpu)camData: Path to file containing generated CAMs (or dictionary). Can be used instead of Config and Checkpoint
    method: Method used for generating the CAMs. Defaults to 'gradcam'
    annfile: Path to annotation file specifng which samples should be used.
    dataClasses: Array of Class Prefixes that specify which sample classes should be used. If not specified everything will be generated.
    """
    assert visType in VISUALIZE_TYPES, f'Specified type {visType} not the available types: {",".join(VISUALIZE_TYPES)}'

    gpuKwargs = {key[len('gpu'):]:value for key, value in kwargs.items() if key.lower().startswith('gpu')}
    cpuKwargs = {key[len('cpu'):]:value for key, value in kwargs.items() if key.lower().startswith('cpu')}

    assert os.path.isdir(kwargs['imgRoot']), f'imgRoot does not lead to a directory {kwargs["imgRoot"]}'

    if 'imgNames' in kwargs:
        imgNames = kwargs['imgNames']
    else:
        assert 'imgRoot' in kwargs, 'imgRoot must be specified.'
        imgNames = get_samples(**kwargs) # Required annfile or (imgRoot) in kwargs

    if len(imgNames) == 0:
        raise ValueError('Given parameters do not yield any images.')

    print(f'Generate results for type {visType}')

    # For CAM: Here we need camConfig, camCheckpoint or camData, imgRoot, (camDevice), (method), (dataClasses) and (annfile)
    camsGPU = prepareInput(prepImg=False, prepSeg=False, prepCam=True, **kwargs, **gpuKwargs)[0] # Need to index here since we return a list
    camsCPU = prepareInput(prepImg=False, prepSeg=False, prepCam=True, **kwargs, **cpuKwargs)[0] # Need to index here since we return a list
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given GPU CAM Dictionary does not contain all imgNames as keys.'
    assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(camsGPU.keys()))) or set(imgNames).issubset(set(camsGPU.files)), f'Given CPU CAM Dictionary does not contain all imgNames as keys.'

    cfg = get_pipeline_cfg(**kwargs)

    if visType != VISUALIZE_TYPES[0]:
        # Create and save visualization images
        if visType == VISUALIZE_TYPES[2] or visType == VISUALIZE_TYPES[3]:
            # Only load sourceImgs if we need them for overlays
            sourceImgs = prepareInput(prepImg=True, prepSeg=False, prepCam=False, **kwargs)[0] # Need to index here since we return a list
            transformedSourceImgs = {}
            if cfg:
                pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=False, workPIL=True)
                print('Tranforming sourceImages with the given pipeline.')
            for name in imgNames:
                transformedSourceImgs[name] = pipeline(sourceImgs[name]) if cfg else sourceImgs[name]

        saveDirImg = os.path.join(saveDir, 'imgs')
        print(f'Saving images into directory {saveDirImg}')
        for name in imgNames:
            if visType == VISUALIZE_TYPES[1] or visType == VISUALIZE_TYPES[3]:
                # Use CAM Heatmaps
                cpuImg = convert_numpy_to_PIL(camsCPU[name])
                gpuImg = convert_numpy_to_PIL(camsGPU[name])
            elif visType == VISUALIZE_TYPES[2]:    
                cpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsCPU[name]))
                gpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsGPU[name]))
            else:
                raise ValueError(f'Given type {visType} is unknown. Please use one of {",".join(VISUALIZE_TYPES)}')
            add_text(cpuImg, 'CPU')
            add_text(gpuImg, 'GPU')
            combinedImg = concatenate_images(cpuImg, gpuImg)
            if visType == VISUALIZE_TYPES[3]:
                cpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsCPU[name]))
                gpuImg = convert_numpy_to_PIL(generate_cam_overlay(transformedSourceImgs[name], camsGPU[name]))
                combinedImgOverlay = concatenate_images(cpuImg, gpuImg)
                combinedImg = concatenate_images(combinedImg, combinedImgOverlay, direction='vertical')
            nameParts = name.split('.')
            if len(nameParts) > 1:
                fName = ".".join(nameParts[:-1])
                fExtension = "." + nameParts[-1]
            else:
                fName = name
                fExtension = ""
            savePIL(combinedImg, fileName=fName + "_" + visType + fExtension, dir=saveDirImg, logSave=False)

        
    if plot:
        # Create plot
        segmentationsGPU, _ = prepareInput(prepImg=False, prepSeg=True, prepCam=False, **kwargs, **gpuKwargs)
        segmentationsCPU, _ = prepareInput(prepImg=False, prepSeg=True, prepCam=False, **kwargs, **cpuKwargs)
        assert (isinstance(camsGPU, dict) and set(imgNames).issubset(set(segmentationsGPU.keys()))) or set(imgNames).issubset(set(segmentationsGPU.files)), f'Given GPU Segmentation Dictionary does not contain all imgNames as keys.'
        assert (isinstance(camsCPU, dict) and set(imgNames).issubset(set(segmentationsCPU.keys()))) or set(imgNames).issubset(set(segmentationsCPU.files)), f'Given CPU Segmentation Dictionary does not contain all imgNames as keys.'

        transformedSegmentationsGPU = {}
        transformedSegmentationsCPU = {}

        if cfg:
            pipeline = get_pipeline_torchvision(cfg.data.test.pipeline, scaleToInt=True, workPIL=True)
            print('Tranforming segmentation masks with the given pipeline.')
        for name in imgNames:
            transformedSegmentationsGPU[name] = pipeline(segmentationsGPU[name]) if cfg else segmentationsGPU[name]
            transformedSegmentationsCPU[name] = pipeline(segmentationsCPU[name]) if cfg else segmentationsCPU[name]
        
        classes = load_classes(**kwargs)

        totalCAMActivationsGPU, _, percentualSegmentedCAMActivationsGPU = accumulate_statistics(imgNames=imgNames, classes=classes, cams=camsGPU, segmentations=transformedSegmentationsGPU)
        totalCAMActivationsCPU, _, percentualSegmentedCAMActivationsCPU = accumulate_statistics(imgNames=imgNames, classes=classes, cams=camsCPU, segmentations=transformedSegmentationsCPU)

        classArray, totalActivationMeanGPU, summarizedPercSegmentedCAMActivationsGPU, _ = generate_stats(classes=classes, totalCAM=totalCAMActivationsGPU, percentualActivations=percentualSegmentedCAMActivationsGPU, get_total_mean=True, get_total_sum=False)
        _, totalActivationMeanCPU, summarizedPercSegmentedCAMActivationsCPU, _ = generate_stats(classes=classes, totalCAM=totalCAMActivationsCPU, percentualActivations=percentualSegmentedCAMActivationsCPU, get_total_mean=True, get_total_sum=False)

        fig = plt.figure(figsize=(15,5), constrained_layout=True)
        grid = fig.add_gridspec(ncols=4, nrows=1)

        numSamples = len(imgNames)

        ax0 = fig.add_subplot(grid[0,0:3])
        ax0.text(0.9,1.02, f'No.Samples:{numSamples}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)
        ax0.set_title('Average relative CAM Activations')

        # Default width is 0.8 and since we are plotting two bars side by side avoiding overlap requires
        # reducing the width
        barwidth = 0.4

        bars = ax0.bar(np.arange(classArray.size), summarizedPercSegmentedCAMActivationsGPU, width=barwidth)
        ax0.set_xticks([tick+barwidth/2 for tick in range(classArray.size)], classArray)

        plot_bar(ax=ax0, bars=bars, x_tick_labels=classArray, data=summarizedPercSegmentedCAMActivationsGPU, barwidth=barwidth, hightlightDominant=False,
                                textadjust_ypos=True, format='.1%', textrotation=90, keep_x_ticks=True)

        plot_bar(ax=ax0, x_ticks=np.arange(classArray.size)+barwidth, x_tick_labels=classArray, data=summarizedPercSegmentedCAMActivationsCPU, barwidth=barwidth, 
                        barcolor='g', hightlightDominant=False, textadjust_ypos=True, format='.1%', textrotation=90, keep_x_ticks=True, increase_ylim_scale=1.3)

        ax0.text(0.9,1.02, f'No.Samples:{len(imgNames)}',horizontalalignment='center',verticalalignment='center',transform = ax0.transAxes)

        legendMap = {
            'tab:blue':'CAM Activations GPU',
            'tab:green':'Proportional Segment Coverage CPU'
        }
        handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

        ax0.legend(handles=handles)

        if 'dataClasses' in kwargs:
            ax0.set_xlabel(','.join(kwargs['dataClasses']), fontsize='x-large')

        ax1 = fig.add_subplot(grid[0,3])
        ax1.set_title('Mean totalCAM Activation')

        bars = ax1.bar([0], [totalActivationMeanGPU], width=barwidth)
        ax1.set_xticks([barwidth/2], [classArray[0]])
        plot_bar(ax=ax1, bars=bars, x_tick_labels=['Mean totalCAM Activation'], barwidth=barwidth, dominantMask=[True], hightlightColor='tab:blue',
                                textadjust_ypos=True, format='.2f', textrotation=90, keep_x_ticks=True)
        plot_bar(ax=ax1, x_ticks=[barwidth],  x_tick_labels=['Mean totalCAM Activation'], data=[totalActivationMeanCPU], dominantMask=[True], hightlightColor='tab:green', 
                                textadjust_ypos=True, barcolor='g', format='.2f', textrotation=90, barwidth=barwidth, keep_x_ticks=True, increase_ylim_scale=1.3)

        ax1.text(0.9,1.02, f'No.Samples:{len(imgNames)}',horizontalalignment='center',verticalalignment='center',transform = ax1.transAxes)

        legendMap = {
            'tab:blue':'totalCAMs GPU',
            'tab:green':'totalCAMs CPU'
        }
        handles = [Patch(color=k, label=v) for k,v in legendMap.items()]

        ax1.legend(handles=handles)
        
        plt.show()

        save_result_figure_data(figure=fig, save_dir=saveDir, path_intermediate='GPUvsCPU', fileNamePrefix=fileNamePrefix, **kwargs, **cpuKwargs)