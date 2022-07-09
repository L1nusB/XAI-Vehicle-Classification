import warnings
import numpy as np
import torch
import argparse
import os.path as osp
import shutil


import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmseg.apis import init_segmentor
from mmseg.datasets import build_dataloader, build_dataset

from .utils.pipeline import get_pipeline_pre_post
from .utils.constants import PALETTES, TYPES
from .utils.io import generate_split_files, get_dir_and_file_path, get_sample_count
from .utils.model import single_gpu_test_thresh

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
        help='Return the results from ONLY the last batch.'
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
        '--consolidate-out',
        nargs='?',
        type=bool,
        const=True,
        help='WILL CAUSE CRASHES WHEN RESULT IS TOO LARGE. Tries to consolidate the output files '
        ' into one.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=.7,
        help='Threshold value which pixels will be segmented into an additional '
        'background category.'
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

def add_background_class(classes, background='background'):
    if background in classes:
        return classes
    classes = classes + (background,)
    return classes


def set_dataset_fields(cfg, args,  classes, palette):
    cfg.type = "GenerationDataset"   # Set type of the Dataset --> Needs to match the custom Datset type in mmseg.datasets
    cfg.img_dir = args.imgDir # Path to the Data that should be converted --> somewhere/data/val
    cfg.data_root = args.root # Path to root folder. Default is ./
    cfg.ann_dir = None # Reset ann_dir so it does try to look for something that does not exist. (Not really necessary)
    cfg.split = osp.abspath(args.ann_file) if args.ann_file else None # Path to the Ann-file that will be used to determine the relevant files. (Like annfile in mmclas)
    cfg.classes = classes    # Set custom Classes from Config since i can not encode it into the Dataset
    cfg.palette = palette # Again set custom Palette based on palettes variable.
    return cfg

def batch_data(cfg, args, work_dir, classes=[], batch_size=5000):
    import copy
    fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    sample_count = get_sample_count(args, fc, classes)
    batch_count = sample_count//batch_size
    if batch_count*batch_size != sample_count:
        batch_count += 1 # Add one if not even division
    subset_cfgs = [None] * batch_count
    if args.ann_file:
        generate_split_files(mmcv.list_from_file(args.ann_file, file_client_args=dict(backend='disk')), batch_count, batch_size, work_dir, classes)
    else:
        generate_split_files(fc.list_dir_or_file(dir_path=osp.join(args.root, args.imgDir), list_dir=False, recursive=True), batch_count, batch_size, work_dir, classes)

    if batch_size == -1:
        subset_cfgs = copy.copy(cfg)
        subset_cfgs.split = osp.abspath(osp.join(work_dir, f'split_{0}.txt'))  # Set split from generated Files
        return [subset_cfgs]
    for i in range(batch_count):
        subset_cfgs[i] = copy.copy(cfg)
        subset_cfgs[i].split = osp.abspath(osp.join(work_dir, f'split_{i}.txt'))  # Set split from generated Files
    return subset_cfgs

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
        work_dir, result_file = get_dir_and_file_path(args.save)
        # If images specified create folder for images
        if TYPES[0] in args.types:
            print(f'Saving resulting masks in {work_dir}')
        if TYPES[1] in args.types:
            img_dir = osp.join(work_dir, 'images')
            print(f'Saving result images in {img_dir}')
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

    model.CLASSES = model.CLASSES + (args.background,)  # Add background class so show_results throws no error

    assert 'CLASSES' in checkpoint.get('meta', {}), f'No CLASSES specified in the checkpoint of the model.'
    classes = checkpoint['meta']['CLASSES']
    # Add Background class
    classes = add_background_class(classes, background=args.background)



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

    if args.consolidate_out:
        warnings.warn('Using Consolidate-out will cause crashes if the amount of samples exceeds memory capacity.')
        subset_cfgs = batch_data(cfg.data.test,args, work_dir, classes=args.classes, batch_size=-1) # Using batch_size -1 causes no limit
    else:
        subset_cfgs = batch_data(cfg.data.test,args, work_dir, classes=args.classes)

    if args.worker_size:
        workers_per_gpu = args.worker_size
    else:
        workers_per_gpu = cfg.data.workers_per_gpu

    if args.batch_size != 1:
        assert args.pipeline, 'Batch Size can ONLY be used if pipeline is given. See --batch-size'
    torch.cuda.empty_cache()


    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    for index, subset in enumerate(subset_cfgs):
        print(f'Calculating results for batch {index}')

        dataset = build_dataset(subset)
        assert args.background in dataset.CLASSES, f'Category {args.background} not in dataset.CLASSES.'
        backgroundIndex = dataset.CLASSES.index(args.background)   # Get numerical index

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=workers_per_gpu,
            shuffle=False,
            dist=False
        )

        results = single_gpu_test_thresh(
            model=model,
            data_loader=data_loader,
            out_dir=img_dir,
            threshold=args.threshold,
            backgroundIndex=backgroundIndex
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
            print(f'\nSaving results for batch {index} at ' + osp.join(work_dir, result_file.split('.')[0] + '_' + str(index) + ".npz"))
            np.savez(osp.join(work_dir, result_file.split('.')[0] + '_' + str(index) + ".npz"), **dict(zip(filenames, results)))

    if args.results:
        imgs = {}
        for result in filenames:
            imgs[result] = mmcv.imread(osp.join(img_dir, result))
        if not args.save:
            shutil.rmtree(img_dir)
        return dict(zip(filenames, results)),imgs

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])