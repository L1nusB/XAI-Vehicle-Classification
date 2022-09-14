import argparse
import os
import warnings
from mmcv import Config, DictAction
from mmcls.apis import init_model
import numpy as np
import mmcv
from pytorch_grad_cam.utils.image import show_cam_on_image

from .vis_cam_custom import getCAM_without_build, get_default_traget_layers, get_layer, build_reshape_transform
from .utils.CAMGenDataset import ImageDataset, add_blurring_pipeline_step
from .utils.io import  generate_split_file, get_dir_and_file_path, savePIL
from .utils.imageProcessing import convert_numpy_to_PIL
from torch.utils.data import DataLoader

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
    parser.add_argument(
        '--save', '-s',
        type=str,
        nargs='?',
        const='./output/',
        help='Save generated cams.'
    )
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
        ' by the annotation file. If a single image is directly specified use '
        ' --target-category with the respective class.'
    )
    parser.add_argument(
        '--classes',
        default=[],
        type=str,
        nargs='+',
        help='Allows specification of classes for which CAMs are to be generated.'
    )
    parser.add_argument(
        '--results','-r',
        type=bool,
        nargs='?',
        const=True,
        help='Return the results.'
    )
    parser.add_argument(
        '--blurredSegments',
        default=[],
        nargs='+',
        help='Segments to be blurred out. Either by name of segment or index. '
        'Multiple Segments may be specified.'
    )
    parser.add_argument(
        '--segData',
        type=str,
        help='Path to numpy file containing segmentation data. '
        'Required if --blurredSegments is specified'
    )
    parser.add_argument(
        '--segConfig',
        type=str,
        help='Path to config File for segmentation model. '
        'Required if --blurredSegments is specified'
    )
    parser.add_argument(
        '--segCheckpoint',
        type=str,
        help='Path to checkpoint File for segmentation model. '
        'Required if --blurredSegments is specified'
    )
    parser.add_argument(
        '--saveBlurred',
        type=bool,
        default=False,
        nargs='?',
        const=True,
        help='Save the blurred images over which the cams are computed. '
        'Required if --blurredSegments is specified'
    )
    args = parser.parse_args(args)
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


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

def generateCAMs(dataset, args, custom_cfg=None):
    print(f'Generate Results for specified files')
    cfg,model,use_cuda,target_layers,reshape_transform = getCAMConfig(args)
    if custom_cfg is not None:
        cfg = custom_cfg
    imgLoader = DataLoader(dataset)

    cams={}
    prog_bar = mmcv.ProgressBar(len(dataset))

    for item in imgLoader:
        path = args.img if os.path.isfile(args.img) else os.path.join(args.img, item['name'][0])
        target_category = args.target_category
        if args.use_ann_labels:
            if target_category:
                warnings.warn('Both use-ann-labels and target-category are specified. Target-category will be ignored.')
            target_category = item['gt_target']
        cam = getCAM_without_build(path, cfg.data.test.pipeline,
            args.method, model, target_layers, use_cuda, reshape_transform, args.eigen_smooth, args.aug_smooth, target_category)
        cams[item['name'][0]] = cam.squeeze()
        prog_bar.update()
    return cams


def generate_cam_overlay(sourceImg, camHeatmap):
    """Creates an overlay of the cam Heatmap over the sourceImg.

    :param sourceImg: Img Data for the Source Img
    :type sourceImg: np.ndarray
    :param camHeatmap: Data for the CAM Heatmap
    :type camHeatmap: np.ndarray
    :return: Overlay Image Data
    :rtype: np.ndarray
    """
    assert sourceImg.shape[:-1] == camHeatmap.shape, f"Shape of camHeatmap {camHeatmap.shape} and sourceImg (heightxwidth) {sourceImg.shape} do not match"
    return show_cam_on_image(sourceImg, camHeatmap, use_rgb=False)


def main(args):
    args = parse_args(args)
    cfg = Config.fromfile(args.config)
    useBlurredDataset = False
    if len(args.blurredSegments) > 0:
        assert args.segData and os.path.isfile(args.segData), f'segData must be specified when using --blurredSegments and lead to file.'
        assert args.segConfig and os.path.isfile(args.segConfig), f'segConfig must be specified when using --blurredSegments and lead to file.'
        assert args.segCheckpoint and os.path.isfile(args.segCheckpoint), f'segCheckpoint must be specified when using --blurredSegments and lead to file.'
        print('Generating CAMs using blurred images.')
        useBlurredDataset = True
    cams = {}
    if args.preview_model:
        model = init_model(cfg, args.checkpoint, device=args.device)
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    if os.path.isfile(args.img):
        if args.ann_file:
            raise ValueError(f'img Parameter does not specify a directory {args.img}')
        print(f'Generate Results for file: {args.img}')
        if useBlurredDataset:
            print(f'Modifying pipeline cfg so that specified segments {",".join(args.blurredSegments)} will be blurred out.')
            saveDir = os.path.join(args.save, 'blurredImgs') if args.save else 'blurredImgs/'
            cfg = add_blurring_pipeline_step(cfg, args.blurredSegments, args.segData, args.segConfig, args.segCheckpoint,
                                            saveDir=saveDir, saveImgs=args.saveBlurred)

        imgDataset = ImageDataset(os.path.dirname(args.img), imgNames=[os.path.basename(args.img)], get_gt=args.use_ann_labels)
    else:
        assert os.path.isdir(args.img), f'Provided path is no file or directory: {args.img}'
        if useBlurredDataset:
            print(f'Modifying pipeline cfg so that specified segments {",".join(args.blurredSegments)} will be blurred out.')
            saveDir = os.path.join(args.save, 'blurredImgs') if args.save else 'blurredImgs/'
            cfg = add_blurring_pipeline_step(cfg, args.blurredSegments, args.segData, args.segConfig, args.segCheckpoint,
                                            saveDir=saveDir, saveImgs=args.saveBlurred)
        
        imgDataset = ImageDataset(args.img, annfile=args.ann_file, dataClasses=args.classes, get_gt=args.use_ann_labels)

    print(f'Method for CAM generation: {args.method}, eigen-smooth:{args.eigen_smooth}, aug-smooth:{args.aug_smooth}, vit-like:{args.vit_like}')
    if args.use_ann_labels:
        print('Using annotation labels provided by the annfile.')
    
    cams = generateCAMs(imgDataset, args, custom_cfg=cfg)
    print("")

    if args.save:
        work_dir, result_file_prefix = get_dir_and_file_path(args.save, defaultName='resultsCAM', removeFileExtensions=True)
        mmcv.mkdir_or_exist(os.path.abspath(work_dir))

        print(f'Save Split file for Cams')
        generate_split_file(imgDataset.imgPaths, work_dir, fileprefix=result_file_prefix)
        
        path = os.path.join(work_dir, result_file_prefix + ".npz")
        print(f'Save generated CAMs to {path}')
        np.savez(path, **cams)
    if args.results:   
        return cams

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

def saveImgsFromFile(path, saveDir='./CAMImages/'):
    """
    Saves the images of all CAMs that are stored in a file.
    """
    assert os.path.isfile(path), f'No such file {path}'
    file = np.load(path)
    for key in file:
        pil = convert_numpy_to_PIL(file[key])
        savePIL(pil, fileName=key, dir=saveDir, logSave=False)