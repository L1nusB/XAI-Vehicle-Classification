import warnings
import numpy as np
import torch
import argparse
import os.path as osp
import shutil


import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmseg.apis import init_segmentor, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset

from .utils.pipeline import get_pipeline_pre_post
from .utils.constants import PALETTES, TYPES
from .utils.io import generate_split_file, get_dir_and_file_path
from .utils.model import wrap_model
from .utils.preprocessing import load_classes

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('imgDir',type=str, help='Path to image folder.')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--root',
        type=str, 
        default='./',
        help='Path to image/root folder.')
    parser.add_argument('--batch-size',
    type=int,
    default=1,
    help='Number of samples each batch is processing. Only appliable if pipeline is given '
    'in order to ensure all samples have same dimensions')
    parser.add_argument('--worker-size',
    type=int,
    help='Number of workers per GPU.')
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
        const='./output/',
        help='Save generated masks/images.'
    )
    parser.add_argument(
        '--results','-r',
        type=bool,
        nargs='?',
        const=True,
        help='Return the results.'
    )
    parser.add_argument(
        '--palette',
        help='Name of Color palette used for segmentation map See static definitions above.'
    )
    parser.add_argument(
        '--ann-file',
        help='Path to a txt file specifying a set of images for which'
        ' results are generated.'
    )
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--classes','-c',
        default=[],
        type=str,
        nargs='+',
        help='Specifies of classes for which results are generated.'
    )
    parser.add_argument(
        '--pipeline', '-p',
        nargs='+',
        default=[],
        type=str,
        help='Specify whether to apply a pipeline transformation before calculating '
        'the segmentations or afterwards. The second part specifies what pipeline is applied '
        'either by passing a configPath or a predefined Pipeline Mapping. Additionally it can be '
        'specified if a given pipeline will be compared to a translation table for the components.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=.7,
        help='Threshold value which pixels will be segmented into an additional '
        'background category.'
    )
    parser.add_argument(
        '--use-threshold',
        default=False,
        type=bool,
        help='Use a different inference method for single_gpu_test that takes distribution of input '
        'and assigns values below the threshold to a background class. Background class will be added '
        'based on --background to the segmentation classes.'
    )
    parser.add_argument(
        '--background',
        type=str,
        default='background',
        help='Name of the background category that uncertain areas will be assigned to. '
        'If this category is not in model.CLASSES it will be added.'
    )
    args = parser.parse_args(args)
    if args.types is None:
        args.types = ['masks']
    for type in args.types:
        if not type in TYPES:
            raise ValueError(f'Invalid Type specified:{type},'
                         f' supports {", ".join(TYPES)}.')
    return args

def set_dataset_fields(cfg, args, classes, palette):
    cfg.type = "GenerationDataset"   # Set type of the Dataset --> Needs to match the custom Datset type in mmseg.datasets
    cfg.img_dir = args.imgDir # Path to the Data that should be converted --> somewhere/data/val
    cfg.data_root = args.root # Path to root folder. Default is ./
    cfg.ann_dir = None # Reset ann_dir so it does try to look for something that does not exist. (Not really necessary)
    cfg.split = osp.abspath(args.ann_file) if (args.ann_file and osp.isfile(args.ann_file)) else None # Path to the Ann-file that will be used to determine the relevant files. (Like annfile in mmclas)
    cfg.classes = classes    # Set custom Classes from Config since i can not encode it into the Dataset
    cfg.palette = palette # Again set custom Palette based on palettes variable.
    return cfg

def prepare_data_cfg(cfg, args, work_dir, classes=[], filename='resultsSeg'):
    """
    Generate Split files for dataset generation and set it in the cfg.
    """
    fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if args.ann_file and osp.isfile(args.ann_file):
        generate_split_file(mmcv.list_from_file(args.ann_file, file_client_args=dict(backend='disk')), work_dir, classes, fileprefix=filename)
    else:
        generate_split_file(fc.list_dir_or_file(dir_path=osp.join(args.root, args.imgDir), list_dir=False, recursive=True), work_dir, classes, fileprefix=filename)
    cfg.split = osp.abspath(osp.join(work_dir, f'{filename}.txt'))
    return cfg

def main(args):
    args = parse_args(args)
    cfg = mmcv.Config.fromfile(args.config)

    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [0]

    # Save folder for image saving if specified. (Outside since it is referenced later
    # regardless of args.save)
    img_dir = None
    work_dir = './'
    if args.save:
        # results_file_prefix will not have a file extension!
        work_dir, result_file_prefix = get_dir_and_file_path(args.save, defaultName='resultsSeg', removeFileExtensions=True)
        # If images specified create folder for images
        if TYPES[0] in args.types:
            print(f'Saving resulting segmentation masks in {work_dir}')
        if TYPES[1] in args.types:
            img_dir = osp.join(work_dir, 'images')
            print(f'Saving result segmentation images in {img_dir}')
            # Only use one mkdir here since it is recursive and generates work_dir
            mmcv.mkdir_or_exist(osp.abspath(img_dir))
        else:
            mmcv.mkdir_or_exist(osp.abspath(work_dir))
    elif args.results and TYPES[1] in args.types:   
        # Temporary create images as output to be read afterwards. Will be deleted after completion.
        img_dir = osp.join(work_dir, 'images')
        mmcv.mkdir_or_exist(osp.abspath(img_dir))

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.use_threshold and args.background not in model.CLASSES:
        model.CLASSES = model.CLASSES + (args.background,) # Add background class so show_results throws no error if specified (only if not present yes)


    assert 'CLASSES' in checkpoint.get('meta', {}), f'No CLASSES specified in the checkpoint of the model.'
    classes = checkpoint['meta']['CLASSES']
    # Add Background class
    classes = load_classes(classes, addBackground=args.use_threshold, backgroundCls=args.background)

    if args.palette is None:
        if 'PALETTE' in checkpoint.get('meta', {}):
            palette = checkpoint['meta']['PALETTE']
        else:
            print("No Palette specified or found in checkpoint. Generating random one.")
            state = np.random.get_state()
            np.random.seed(42)
            # random palette
            palette = np.random.randint(0, 255, size=(len(model.CLASSES), 3))
            np.random.set_state(state)
    else:
        assert args.palette in PALETTES.keys(),f"Palette {args.palette} not defined. "\
            "Remove parameter use the Palette from the checkpoint or generate a random one."
        palette = PALETTES[args.palette]

    cfg.data.test = set_dataset_fields(cfg.data.test, args, classes, palette)

    prePipeline, postPipeline = get_pipeline_pre_post(args)
    if prePipeline:
        print('Adding Pipeline steps into preprocessing.')
        for step in cfg.data.test.pipeline:
            if step.type=='MultiScaleFlipAug':
                step.transforms = prePipeline + step.transforms

    cfg.data.test = prepare_data_cfg(cfg.data.test, args, work_dir, args.classes, filename=result_file_prefix)

    if args.worker_size:
        workers_per_gpu = args.worker_size
    else:
        workers_per_gpu = cfg.data.workers_per_gpu

    if args.batch_size != 1:
        assert args.pipeline, 'Batch Size can ONLY be used if pipeline is given. See --batch-size'
    torch.cuda.empty_cache()

    #Wrap Model into a Wrapper Class that can dynamically use Threshold or standard behaviour if desired
    model = wrap_model(model, args.use_threshold, args.threshold, model.CLASSES.index(args.background))

    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    print('Calculating segmentation results.')

    dataset = build_dataset(cfg.data.test)

    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=workers_per_gpu,
            shuffle=False,
            dist=False
    )

    if args.use_threshold:
            assert args.background in dataset.CLASSES, f'Category {args.background} not in dataset.CLASSES.'

    results = single_gpu_test(
        model=model,
        data_loader=data_loader,
        out_dir=img_dir
    )

    if postPipeline:
        transformedResults = []
        for result in results:
            transformedResult = postPipeline(result)
            transformedResults.append(transformedResult)
        results = transformedResults

    if args.save or args.results:
        filenames = [i['filename'].strip() for i in dataset.img_infos]

    if args.save:
        print(f'\nSaving results at ' + osp.join(work_dir, result_file_prefix + ".npz"))
        np.savez(osp.join(work_dir, result_file_prefix + ".npz"), **dict(zip(filenames, results)))

    if args.results:
        if TYPES[1] in args.types:
            imgs = {}
            for result in filenames:
                imgs[result] = mmcv.imread(osp.join(img_dir, result))
            if not args.save:
                shutil.rmtree(img_dir)
            return dict(zip(filenames, results)),imgs
        else:
           return dict(zip(filenames, results)) 

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])