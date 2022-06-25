import argparse
import os
from mmcv import Config, DictAction
from mmcls.apis import init_model
from pathlib import Path
import numpy as np
from tqdm import tqdm

from .vis_cam_custom import getCAM, getCAM_Multiple, get_default_traget_layers, get_layer, build_reshape_transform

try:
    from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM,
                                  GradCAMPlusPlus, LayerCAM, XGradCAM)
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

# Supported grad-cam type map
METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigencam': EigenCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
}

def parse_args(args):
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('img', help='Path to Image file or folder of image files')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--save-path',
        help='The path to save visualize cam image.')
    parser.add_argument(
        '--target-layers',
        default=[],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the norm layer in the last block. Backbones '
        'implemented by users are recommended to manually specify'
        ' target layers in commmad statement.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers')
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=[],
        nargs='+',
        type=int,
        help='The target category to get CAM, default to use result '
        'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Wether to use test time augmentation, default not to use')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--vit-like',
        action='store_true',
        help='Whether the network is a ViT-like network.')
    parser.add_argument(
        '--num-extra-tokens',
        type=int,
        help='The number of extra tokens in ViT-like backbones. Defaults to'
        ' use num_extra_tokens of the backbone.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--ann-file',
        help='Path to a txt file specifying a set of images for which'
        ' the CAM is generated. Requires \{img\} to be a directory'
        ' which is used as the base for the images.'
    )
    parser.add_argument(
        '--use-ann-labels',
        default=False,
        action='store_true',
        help='Determine whether to use the target category specified'
        ' by the annotation file.'
    )
    parser.add_argument(
        '--classes',
        default=[],
        type=str,
        nargs='+',
        help='Allows specification of classes for which CAMs are to be generated.'
    )
    args = parser.parse_args(args)
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args

def printProgess(iteration, total, prefix='', suffix='', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def getFileCount(args):
    folder = args.img
    pathiter = (os.path.join(root, filename)
        for root, _, filenames in os.walk(folder)
        for filename in filenames
    )
    return len(list(pathiter))

def getCAMConfig(args):
    cfg = Config.fromfile(args.config)
    model = init_model(cfg, args.checkpoint, device=args.device)
    use_cuda = ('cuda' in args.device)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.target_layers:
        target_layers = [
            get_layer(layer, model) for layer in args.target_layers
        ]
    else:
        target_layers = get_default_traget_layers(model, args)
    reshape_transform = build_reshape_transform(model, args)

    return cfg,model,use_cuda,target_layers,reshape_transform

def generateDir(args):
    assert os.path.isdir(args.img)
    if args.classes and ('None' not in args.classes):
        print("Generating CAMs for all files in folder:" + args.img + " matching any of class: " + ",".join(args.classes))
    else:
        print("Generating CAMs for all files in folder:" + args.img)
    cams = {}
    cfg,model,use_cuda,target_layers,reshape_transform = getCAMConfig(args)
    if args.classes and ('None' not in args.classes):
        pathiter = [os.path.join(args.img,f) for f in os.listdir(args.img) if any(f.startswith(s) for s in args.classes) and os.path.isfile(os.path.join(args.img,f))]
        totalFiles = len(pathiter)
    else:
        pathiter = (os.path.join(root, filename)
            for root, _, filenames in os.walk(args.img)
            for filename in filenames
        )
        totalFiles = getFileCount(args)
    with tqdm(total=totalFiles) as pbar:
        for index,path in enumerate(pathiter):
            cam = getCAM_Multiple(path, cfg.data.test.pipeline, args.method, model, target_layers, use_cuda, reshape_transform, args.eigen_smooth, args.aug_smooth, args.target_category)
            cams[os.path.basename(path)] = cam.squeeze()
            pbar.set_description(f'CAMs generated:{index+1}/{totalFiles}')
            pbar.update(1)
            #printProgess(iteration=index+1, total=totalFiles, prefix=f'CAMs generated:{index+1}/{totalFiles}')
            #print(f'CAMs generated:{index+1}/{totalFiles}')
    return cams

def generateDefined(args):
    if args.classes and ('None' not in args.classes):
        print("Generating CAMs for all files specified by file: "+args.ann_file+ " matching any of class: " + ",".join(args.classes))
    else:
        print("Generating CAMs for all files specified by file: " + args.ann_file)
    if args.use_ann_labels:
        print("Use target categories provided by annotation file.")
    cfg,model,use_cuda,target_layers,reshape_transform = getCAMConfig(args)
    if not os.path.isdir(args.img):
        raise ValueError(f'img Parameter does not specify a directory {args.img}')
    with open(args.ann_file, encoding='utf-8') as f:
        if args.classes and ('None' not in args.classes):
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines() if any(x.startswith(s) for s in args.classes)]
        else:
            samples = [x.strip().rsplit(' ', 1) for x in f.readlines()]

    cams = {}
    totalFiles = len(samples)
    with tqdm(total=totalFiles) as pbar:
        for index, sample in enumerate(samples):
            path = os.path.join(args.img, sample[0])
            targets = args.target_category
            if args.use_ann_labels:
                if len(sample)<2:
                    print(f'No target category in annotation file for {sample[0]} using model result.')
                else:
                    targets = [(int)(sample[1].split("_")[0])]
            cam = getCAM_Multiple(path, cfg.data.test.pipeline, args.method, model, target_layers, use_cuda, reshape_transform, args.eigen_smooth, args.aug_smooth, targets)
            cams[os.path.basename(path)] = cam.squeeze()
            pbar.set_description(f'CAMs generated:{index+1}/{totalFiles}')
            pbar.update(1)
            #print(f'CAMs generated:{index+1}/{totalFiles}')
            #printProgess(iteration=index+1, total=totalFiles, prefix=f'CAMs generated:{index+1}/{totalFiles}')
    return cams

def saveCAMs(args, cams):
    print("Save generated CAMs to " + args.save_path)
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    if os.path.isdir(args.save_path) or not os.path.basename(args.save_path):
        print(f'No filename specified. Generating file "generated_cams.npz" in directory {args.save_path}')
        path = os.path.join(args.save_path,"generated_cams.npz")
    else: 
        if os.path.basename(args.save_path)[-4:] == ".npz":
            path = args.save_path
        else:
            path = os.path.join(os.path.dirname(args.save_path), os.path.basename(args.save_path)+".npz")
    print(f'Output path to CAM file:{path}')
    
    np.savez(path,**cams)

    

def main(args):
    args = parse_args(args)
    if args.preview_model:
        cfg = Config.fromfile(args.config)
        model = init_model(cfg, args.checkpoint, device=args.device)
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    if args.ann_file:
        cams = generateDefined(args) 
    elif os.path.isdir(args.img):
        cams = generateDir(args) 
    else:
        print("Generating CAM for file:" + args.img)
        cam = getCAM(args)
        cams = {os.path.basename(args.img): cam.squeeze()}

    if args.save_path:
        saveCAMs(args, cams)
    return cams

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])