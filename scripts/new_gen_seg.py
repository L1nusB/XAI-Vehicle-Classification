from email.generator import Generator
from fileinput import filename
from typing import Iterable
import numpy as np
import torch
import argparse
from pathlib import Path
import os.path as osp

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from mmseg.apis import single_gpu_test, init_segmentor
from mmseg.datasets import build_dataloader, build_dataset

from mmseg.datasets.pipelines.transforms import ResizeCls


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
        type=Path,
        help='Path to config File from which a pipeline will be extracated based on'
        ' .data.test.pipeline'
    )
    parser.add_argument(
        '--consolidate-out',
        type=bool,
        default=False,
        help='WILL CAUSE CRASHES WHEN RESULT IS TOO LARGE. Tries to consolidate the output files '
        ' into one.'
    )
    args = parser.parse_args(args)
    if args.types is None:
        args.types = ['masks']
    for type in args.types:
        if not type in TYPES:
            raise ValueError(f'Invalid Type specified:{type},'
                         f' supports {", ".join(TYPES)}.')
    return args

def get_dir_and_file_path(path, defaultName='results.npz', defaultDir='./output/'):
    directory = defaultDir
    fileName = defaultName
    if osp.isdir(path):
        # Path is a directory
        # Just use default Name
        directory = path
    elif osp.dirname(path):
        # Base Directory Specified
        # Override default Dir
        directory = osp.dirname(path)
    # No else since otherwise default Dir is used

    if osp.basename(path):
        fileName = osp.basename(path)
        if osp.basename(path)[:-4] != '.npz':
            # Change file extension to .npz
            fileName = fileName + ".npz"
    # Again no else needed since default is used otherwise
    return directory, fileName

def set_dataset_fields(cfg, args,  classes, palette):
    cfg.type = "GenerationDataset"   # Set type of the Dataset --> Needs to match the custom Datset type in mmseg.datasets
    cfg.img_dir = args.imgDir # Path to the Data that should be converted --> somewhere/data/val
    cfg.data_root = args.root # Path to root folder. Default is ./
    cfg.ann_dir = None # Reset ann_dir so it does try to look for something that does not exist. (Not really necessary)
    cfg.split = osp.abspath(args.ann_file) if args.ann_file else None # Path to the Ann-file that will be used to determine the relevant files. (Like annfile in mmclas)
    cfg.classes = classes    # Set custom Classes from Config since i can not encode it into the Dataset
    cfg.palette = palette # Again set custom Palette based on palettes variable.
    return cfg

def get_sample_count(args, fc=None, classes=[]):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if args.ann_file:
        if len(classes)>0:
            sample_size = sum(1 for i in mmcv.list_from_file(args.ann_file, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in classes))
        else:
            sample_size = sum(1 for _ in mmcv.list_from_file(args.ann_file, file_client_args=dict(backend='disk')))
    else:
        if classes:
            sample_size = sum(1 for i in fc.list_dir_or_file(dir_path=osp.join(args.root, args.imgDir), list_dir=False, recursive=True)if any(i.startswith(c) for c in classes))
        else:
            sample_size = sum(1 for _ in fc.list_dir_or_file(dir_path=osp.join(args.root, args.imgDir), list_dir=False, recursive=True))
    return sample_size

def generate_split_files(sample_iterator, batch_count, batch_size, work_dir, classes=[]):
    sample_list = list(sample_iterator)
    if len(classes)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in classes)]
    for i in range(batch_count):
        with open(osp.join(work_dir, f'split{i}.txt'),'w') as f:
            f.write('\n'.join(sample_list[i*batch_size:(i+1)*batch_size]))

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

    for i in range(batch_count):
        subset_cfgs[i] = copy.copy(cfg)
        subset_cfgs[i].split = osp.abspath(osp.join(work_dir, f'split{i}.txt'))  # Set split from generated Files
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

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    assert 'CLASSES' in checkpoint.get('meta', {}), f'No CLASSES specified in the checkpoint of the model.'
    classes = checkpoint['meta']['CLASSES']

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


    # Insert a Resize Step
    # IF THIS DOES NOT WORK YOU NEED TO COPY
    # THE RESIZE FROM MMCLS INTO mmseg.datasets.pipelines.transforms 
    # AS RESIZECLS CLASS
    # for step in cfg.data.test.pipeline:
    #     if step.type=='MultiScaleFlipAug':
    #         step.transforms.insert(0,ResizeCls(size=(224,224)))


    # If I Want to use a pipeline beforehand (apply model to rescaled images etc.)
    # I need to modify cfg.data.test.pipeline here so the dataset gets built with the
    # correct pipeline.

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

        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=args.batch_size,
            workers_per_gpu=workers_per_gpu,
            shuffle=False,
            dist=False
        )

        # All other params can be omitted 
        # (maybe besides out_dir which could be used for saving images?)
        results = single_gpu_test(
            model=model,
            data_loader=data_loader,
            out_dir=img_dir
        )

        if args.save:
            print(f'\nSaving results for batch {index} at ' + osp.join(work_dir, result_file.split('.')[0] + str(index) + ".npz"))
            filenames = [i['filename'].strip() for i in dataset.img_infos]
            np.savez(osp.join(work_dir, result_file.split('.')[0] + str(index) + ".npz"), **dict(zip(filenames, results)))

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])