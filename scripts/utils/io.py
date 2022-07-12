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

def get_samples(ann_file=None, imgRoot=None, imgDir=None, fc=None, classes=[], **kwargs):
    if fc is None:
        fc = mmcv.FileClient.infer_client(dict(backend='disk'))
    if ann_file:
        if len(classes)>0:
            samples = [i for i in mmcv.list_from_file(ann_file, file_client_args=dict(backend='disk')) if any(i.startswith(c) for c in classes)]
        else:
            samples = [i for i in mmcv.list_from_file(ann_file, file_client_args=dict(backend='disk'))]
    else:
        if classes:
            samples = [i for i in fc.list_dir_or_file(dir_path=osp.join(imgRoot, imgDir), list_dir=False, recursive=True)if any(i.startswith(c) for c in classes)]
        else:
            samples = [i for i in fc.list_dir_or_file(dir_path=osp.join(imgRoot, imgDir), list_dir=False, recursive=True)]
    return samples

def get_sample_count(args, fc=None, classes=[]):
    return len(get_samples(ann_file=args.ann_file, imgRoot=args.root, imgDir=args.imgDir, fc=fc, classes=classes))

def generate_split_files(sample_iterator, batch_count, batch_size, work_dir, classes=[]):
    sample_list = list(sample_iterator)
    if len(classes)>0:
        sample_list = [sample for sample in sample_list if any(sample.startswith(c) for c in classes)]
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