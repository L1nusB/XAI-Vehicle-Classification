# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import mmcv
from torch.utils.data import DataLoader
from mmcv import Config
import warnings

from mmseg.apis import inference_segmentor, init_segmentor

#from . import utils
from . import transformations
from .ImageDataset import ImageDataset

PALETTES={
    'Comp_Original_Ocrnet_Carparts_Noflip':
    [[102, 179,  92],
       [ 14, 106,  71],
       [188,  20, 102],
       [121, 210, 214],
       [ 74, 202,  87],
       [116,  99, 103],
       [151, 130, 149],
       [ 52,   1,  87],
       [235, 157,  37],
       [129, 191, 187],
       [ 20, 160, 203],
       [ 57,  21, 252],
       [235,  88,  48],
       [218,  58, 254],
       [169, 219, 187],
       [207,  14, 189],
       [189, 174, 189],
       [ 50, 107,  54]]
}

TYPES=[
    'masks',
    'images',
]

def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('img',type=str, help='Path to image/root folder.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', 
        default='cuda:0', 
        help='Device used for inference'
    )
    parser.add_argument(
        '--types', '-t',
        nargs='+',
        help=f'What is generated by the script. Supports {",".join(TYPES)}'
    )
    parser.add_argument(
        '--save', '-s',
        type=str,
        nargs='?',
        const='./output',
        help='Save generated masks/images.'
    )
    parser.add_argument(
        '--palette',
        help='Name of Color palette used for segmentation map See static definitions above.'
    )
    parser.add_argument(
        '--ann-file',
        help='Path to a txt file specifying a set of images for which'
        ' results arae generated.'
    )
    parser.add_argument(
        '--classes','-c',
        default=[],
        type=str,
        nargs='+',
        help='Specifies of classes for which results are generated.'
    )
    parser.add_argument(
        '--pipeline', '-p',
        type=Path,
        help='Path to config File from which a pipeline will be extracated based on'
        ' .data.test.pipeline'
    )
    parser.add_argument(
        '--pipelineScale',
        type=bool,
        default=False,
        help='Use the scaleToInt option for the pipeline. Only relevant if -p is specified.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='NOT SUPPORTED (YET) Batch size used in dataloader. Only applied when -p is specified.'
    )
    parser.add_argument(
        '--consolidate-out',
        type=bool,
        default=False,
        help='WILL CAUSE CRASHES WHEN RESULT IS TOO LARGE. Tries to consolidate the output files '
        ' into one.'
    )
    args = parser.parse_args(args)
    for type in args.types:
        if not type in TYPES:
            raise ValueError(f'Invalid Type specified:{type},'
                         f' supports {", ".join(TYPES)}.')
    return args

def generateImage(model,
                    img,
                    result,
                    palette=None,
                    opacity=0.5):
    """Visualize the segmentation results on the image.
    See @mmseg.apis inference show_result_pyplot

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    return img


def saveResults(savePath, results, defaultName='generated_segmentations.npz'):
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

def saveImages(savePath, images):
    print(f'Saving result images in: {savePath}')
    Path(os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    base = os.path.dirname(savePath)
    if not os.path.isdir(savePath):
        print(f'Output path is not a directory. Using base directory: {os.path.dirname(savePath)}.')
    outPath = os.path.join(base, 'images')
    Path(os.path.dirname(outPath)).mkdir(parents=True, exist_ok=True)
    print(f'Saving images into folder: {outPath}')
    for name,img in images.items():
        mmcv.imwrite(img, os.path.join(outPath, name))

def main(args):
    args = parse_args(args)

    results = {}
    images = {}

    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    if args.palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        # random palette
        palette = np.random.randint(0, 255, size=(len(model.CLASSES), 3))
        np.random.set_state(state)
    else:
        assert args.palette in PALETTES.keys(),f'Palette {args.palette} not defined. Remove parameter to generate a random one.'
        palette = PALETTES[args.palette]

    pipeline = None
    batch_size=1
    if args.batch_size != 1:
        warnings.warn('Batch Size Argument is currently not supported. Size of 1 will be used.')
    
    if args.pipeline:
        cfg = Config.fromfile(args.pipeline)
        pipeline = transformations.get_pipeline_from_config_pipeline(cfg.data.test.pipeline, scaleToInt=args.pipelineScale)

    if os.path.isfile(args.img):
        if args.ann_file:
            raise ValueError(f'img Parameter does not specify a directory {args.img}')
        print(f'Generate Results for file: {args.img}')
        imgData = ImageDataset(os.path.dirname(args.img), imgNames=[os.path.basename(args.img)], pipeline=pipeline)
    else:
        assert os.path.isdir(args.img), f'Provided path is no file or directory: {args.img}'
        imgData = ImageDataset(args.img, annfile=args.ann_file, classes=args.classes, pipeline=pipeline)

    totalFiles = len(imgData)
    saveGranularity = 5000 / batch_size
    saveIndex = 0
    imgLoader = DataLoader(imgData, batch_size=batch_size)

    if args.save:
        if os.path.isdir(args.save):
            saveFolder = args.save
            savePrefix = 'generated_segmentation'
        else:
            saveFolder = os.path.dirname(args.save)
            savePrefix = os.path.basename(args.save) if os.path.basename(args.save) else 'generated_segmentation'

    with tqdm(total = totalFiles) as pbar:
        for index, item in enumerate(imgLoader):
            # Save after saveGranuality steps to avoid crashing
            if args.save and index % saveGranularity == 0 and index > 0:
                savePath = os.path.join(saveFolder,  savePrefix + str(saveIndex) + '.npz')
                print(f'Saving intermediate file after {index*batch_size} samples at {savePath}')
                if TYPES[0] in args.types:
                    saveResults(savePath, results)
                # Save image files
                if TYPES[1] in args.types:
                    saveImages(args.save, images)
                saveIndex += 1
                results = {}
                images = {}
            result = inference_segmentor(model, item['img'][0].numpy())
            if TYPES[0] in args.types:
                # Use Index here because return value is list(tensor) even though the result is always only one array.
                results[item['name'][0]] = result[0]
            if TYPES[1] in args.types:
                images[item['name'][0]] = generateImage(model, item['img'][0].numpy(), result, palette=palette)
            if pbar.n + batch_size <= pbar.total:
                pbar.set_description(f'Results generated:{index*batch_size+batch_size}/{totalFiles}')
                pbar.update(batch_size)
            else:
                pbar.set_description(f'Results generated:{index*batch_size+1}/{totalFiles}')
                pbar.update(1)

    if args.save:
        savePath = os.path.join(saveFolder,  savePrefix + str(saveIndex) + '.npz')
        print(f'Saving intermediate file at {savePath}')
        # Save masks array
        if TYPES[0] in args.types:
            saveResults(savePath, results)
        # Save image files
        if TYPES[1] in args.types:
            saveImages(args.save, images)
        saveIndex+=1

        if args.consolidate_out:
            warnings.warn('Using Consolidate Out will cause crashed if total file Size exceeds RAM Threshold.')

            dic = {}
            #Collect and combine temporary files.
            for i in range(saveIndex):
                with np.load('temp' + str(i) + '.npz') as temp:
                    dic.update(dict(temp))
            
            #Save final File
            if TYPES[0] in args.types:
                saveResults(args.save, dic)
            for i in range(saveIndex):
                os.remove('temp' + str(i) + '.npz')

    out = []

    if len(results) > 0:
        out.append(results)
    if len(images) > 0:
        out.append(images)

    out.append(palette)

    return out

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])