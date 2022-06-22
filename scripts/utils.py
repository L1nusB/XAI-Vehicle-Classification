import os

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
    # Return Set to avoid accidental duplicates
    return set(imageList)