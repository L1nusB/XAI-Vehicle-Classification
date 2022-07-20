import os.path as osp
import mmcv
from pathlib import Path
import os
import numpy as np

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

def get_samples(annfile=None, imgRoot=None, fc=None, dataClasses=[], **kwargs):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if annfile:
        if len(dataClasses)>0:
            samples = [i for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i for i in mmcv.list_from_file(annfile, file_client_args=dict(backend='disk'))]
    else:
        if dataClasses:
            samples = [i for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)if any(i.startswith(c) for c in dataClasses)]
        else:
            samples = [i for i in fc.list_dir_or_file(dir_path=imgRoot, list_dir=False, recursive=True)]
    return samples

def get_sample_count(args, fc=None, dataClasses=[]):
    return len(get_samples(annfile=args.ann_file, imgRoot=osp.join(args.root, args.imgDir), fc=fc, dataClasses=dataClasses))

def generate_split_files(sample_iterator, batch_count, batch_size, work_dir, dataClasses=[]):
    sample_list = list(sample_iterator)
    if len(dataClasses)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in dataClasses)]
    if batch_size == -1:
        with open(osp.join(work_dir, f'split_{0}.txt'),'w') as f:
            f.write('\n'.join(sample_list))
        return
    for i in range(batch_count):
        with open(osp.join(work_dir, f'split_{i}.txt'),'w') as f:
            f.write('\n'.join(sample_list[i*batch_size:(i+1)*batch_size]))

def saveResults(savePath, defaultName='generated_result.npz', **results):
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

def saveFigure(savePath, figure, defaultName='figure.jpg'):
    print(f'Saving figure in: {savePath}')
    Path(os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
    base = os.path.dirname(savePath)
    if not os.path.isdir(savePath):
        print(f'Output path is not a directory. Using base directory: {os.path.dirname(savePath)}.')
        if os.path.basename(savePath):
            if os.path.basename(savePath)[-4:] == ".jpg" or os.path.basename(savePath)[-4:] == ".png":
                outPath = savePath
            else:
                outPath = savePath + ".jpg"
        else:
            outPath = os.path.join(base, defaultName)
            
    Path(os.path.dirname(outPath)).mkdir(parents=True, exist_ok=True)
    print(f'Saving images to: {outPath}')
    figure.savefig(outPath)


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