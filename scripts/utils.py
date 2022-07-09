import os
from pathlib import Path
import numpy as np

def getImageList(imgRoot, annfile=None, classes=None, addRoot=True):
    # Convert classes into list if string is passed
    if classes and isinstance(classes, str):
        classes = [classes]
    if annfile:
        with open(annfile, encoding='utf-8') as f:
            if classes:
                print(f'Generating results based on file: {annfile} matching any of class: {",".join(classes)}')
                imageList = [x.strip().rsplit(' ', 1)[0] for x in f.readlines() if any(x.startswith(s) for s in classes)]
            else:
                print(f'Generating results based on file: {annfile}')
                imageList = [x.strip().rsplit(' ', 1)[0] for x in f.readlines()]
    else:
        if classes:
            print(f'Generating results for all files in folder: {imgRoot} matching any of class {",".join(classes)}')
            imageList = [f for f in os.listdir(imgRoot) if any(f.startswith(s) for s in classes) and os.path.isfile(os.path.join(imgRoot,f))]
        else:
            print(f'Generating results for all files in folder: {imgRoot}')
            imageList = [f for f in os.listdir(imgRoot) if os.path.isfile(os.path.join(imgRoot,f))]

    if addRoot:
        imageList = [os.path.join(imgRoot, img) for img in imageList]
    # Set to avoid accidental duplicates
    return list(set(imageList))

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