# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
import mmcv

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

from . import utils

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
        help='Color palette used for segmentation map'
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


def saveResults(args, results):
    print(f'Saving results in: {args.save}')
    Path(os.path.dirname(args.save)).mkdir(parents=True, exist_ok=True)
    if not os.path.basename(args.save):
        print(f'No filename specified. Generating file "generated_segmentations.npz" in directory {args.save}')
        path = os.path.join(args.save,"generated_segmentations.npz")
    else: 
        if os.path.basename(args.save)[-4:] == ".npz":
            path = args.save
        else:
            path = os.path.join(os.path.dirname(args.save), os.path.basename(args.save)+".npz")
    
    np.savez(path,**results)

def saveImages(args, images):
    print(f'Saving result images in: {args.save}')
    Path(os.path.dirname(args.save)).mkdir(parents=True, exist_ok=True)
    base = os.path.dirname(args.save)
    if not os.path.isdir(args.save):
        print(f'Output path is not a directory. Using base directory: {os.path.dirname(args.save)}.')
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

    state = np.random.get_state()
    np.random.seed(42)
    # random palette
    palette = np.random.randint(0, 255, size=(len(model.CLASSES), 3))
    np.random.set_state(state)

    if os.path.isfile(args.img):
        if args.ann_file:
            raise ValueError(f'img Parameter does not specify a directory {args.img}')
        print(f'Generate Results for file: {args.img}')
        imgList = [args.img]
    else:
        assert os.path.isdir(args.img), f'Provided path is no file or directory: {args.img}'
        imgList = utils.getImageList(args.img, args.ann_file, args.classes)
    
    totalFiles = len(imgList)
    with tqdm(total = totalFiles) as pbar:
        for index, img in enumerate(imgList):
            assert os.path.isfile(img), f'Provided Path is no image file: {img}'
            result = inference_segmentor(model, img)
            if TYPES[0] in args.types:
                # Use Index here because return value is list(tensor) even though the result is always only one array.
                results[os.path.basename(img)] = result[0]
            if TYPES[1] in args.types:
                images[os.path.basename(img)] = generateImage(model, img, result, palette=palette)
            pbar.set_description(f'Results generated:{index+1}/{totalFiles}')
            pbar.update(1)

    if args.save:
        # Save masks array
        if TYPES[0] in args.types:
            saveResults(args, results)
        # Save image files
        if TYPES[1] in args.types:
            saveImages(args, images)

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