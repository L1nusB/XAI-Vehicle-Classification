import os.path as osp
import mmcv

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