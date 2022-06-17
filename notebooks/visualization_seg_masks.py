import numpy as np
import matplotlib.pyplot as plt
import math

def generateUnaryMasks(segmentation, width=224,height=224, segmentCount=None):
    """Generate binary segmentation masks for each segment in the given Segmentation.
    Index in the resulting list corresponds to original segment value.

    :param segmentation: (np.ndarray) Segmentation data. Must be shape (h,w) or (h,w,1)
    :type segmentation: np.ndarray
    :param width: Width of the resulting Masks. Must match width of input segmentation, defaults to 224
    :type width: int, optional
    :param height: Height of the resulting Masks. Must match height of input segmentation, defaults to 224
    :type height: int, optional
    :param segmentCount: Amount of binary masks to be generated. Will only produce binary masks up to the 
    specified number. If not specified will create masks amounting to the biggest value in the segmentation , defaults to None  
    :type segmentCount: int, optional
    :return: list of binary segmentation Masks
    :rtype: list(np.ndarray)
    """
    assert len(segmentation.shape) == 2 or (len(segmentation.shape)==3 and segmentation.shape[-1]==1), f'Expected shape (h,w) or (h,w,1) but got {segmentation.shape}'
    assert width == segmentation.shape[1], f'Specified width {width} does not match segmentation width {segmentation.shape[1]}'
    assert height == segmentation.shape[0], f'Specified height {height} does not match segmentation height {segmentation.shape[0]}'
    if not segmentCount:
        segmentCount = (int)(np.max(segmentation))
    masks = [np.zeros((height,width)) for _ in range(segmentCount)] 
    for i in range(segmentCount):
        masks[i][segmentation==i] = 1
    return masks


def show_segmentation_Masks_Overlay(classes, segmentation, imgData , model,segmentImageOverlay =None, scaleValues=False, width=224, height=224, palette=None):
    """Generates plots of the segmentation Masks over the original image.

    :param classes: List of Classes of the segmentation.
    :type classes: tuple or list
    :param segmentation: Segmentation Data. Must match shape (h,w) or (h,w,1)
    :type segmentation: np.ndarray
    :param model: Model to be used.
    :param imgData: Path to the image.
    :param scaleValues: Scale values by a factor of 255, defaults to False
    :type scaleValues: bool, optional
    :param width: width of the binary masks. Must match segmentation width, defaults to 224
    :type width: int, optional
    :param segmentImageOverlay: Optional image overlaying the segmentation over the original image.
    :param height: Heigt of the binary masks. Must match segmentation height, defaults to 224
    :type height: int, optional
    :param heatmap: Use binary heatmaps or overlay segmentation masks over the original image. (default True)
    :type heatmap: bool, optional
    :param palette: Palette to be used..
    """
    assert width == segmentation.shape[1], f'Specified width {width} does not match segmentation width {segmentation.shape[1]}'
    assert height == segmentation.shape[0], f'Specified height {height} does not match segmentation height {segmentation.shape[0]}'
    
    assert model is not None, "Model must be specified when not using Heatmaps."

    if len(segmentation.shape) == 3:
        assert segmentation.shape[-1] == 1, f'Segmentation does not match expected shape: {segmentation.shape}. Expected (h,w) or (h,w,1)'
        segmentation = segmentation.squeeze()
    if scaleValues:
        segmentation = 255 * segmentation
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(20)
    fig.set_figwidth(15)
    ncols = 3
    # Add one for segmentation image
    totalPlots = len(classes)+1
    if segmentImageOverlay is not None:
        totalPlots+=1
    nrows = math.ceil(totalPlots/ncols)
    
    grid = fig.add_gridspec(nrows,ncols)

    # Magic number of what to use for the regions that are not in the binary mask when not using Heatmaps.
    # Depends on the palette used.
    backgroundSegmentKey = 5

    for row in range(nrows):
        for col in range(ncols):
            index = row*ncols+col
            if index < len(classes):
                ax = fig.add_subplot(grid[row,col])
                # Make sure that not all is one color.
                z = np.full(segmentation.shape, backgroundSegmentKey)
                z[segmentation==index]=index
                ax.imshow(model.show_result(imgData, [z], palette=palette))
                ax.set_title(classes[index])
    ax = fig.add_subplot(grid[len(classes)//ncols, len(classes) % ncols])
    ax.imshow(segmentation)
    ax.set_title('Segmentation')
    if segmentImageOverlay is not None:
        ax = fig.add_subplot(grid[totalPlots//ncols, (len(classes)+1) % ncols])
        ax.imshow(segmentImageOverlay)
        ax.set_title('Segmentation Overlay')

def show_segmentation_Masks(classes, segmentation, segmentImageOverlay =None, scaleValues=False, width=224, height=224):
    """Generates binary plots of the segmentation Masks.

    :param classes: List of Classes of the segmentation.
    :type classes: tuple or list
    :param segmentation: Segmentation Data. Must match shape (h,w) or (h,w,1)
    :type segmentation: np.ndarray
    :param scaleValues: Scale values by a factor of 255, defaults to False
    :type scaleValues: bool, optional
    :param width: width of the binary masks. Must match segmentation width, defaults to 224
    :type width: int, optional
    :param segmentImageOverlay: Optional image overlaying the segmentation over the original image.
    :param height: Heigt of the binary masks. Must match segmentation height, defaults to 224
    :type height: int, optional
    """
    assert width == segmentation.shape[1], f'Specified width {width} does not match segmentation width {segmentation.shape[1]}'
    assert height == segmentation.shape[0], f'Specified height {height} does not match segmentation height {segmentation.shape[0]}'

    if len(segmentation.shape) == 3:
        assert segmentation.shape[-1] == 1, f'Segmentation does not match expected shape: {segmentation.shape}. Expected (h,w) or (h,w,1)'
        segmentation = segmentation.squeeze()
    if scaleValues:
        segmentation = 255*segmentation
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(20)
    fig.set_figwidth(15)
    ncols = 3
    # Add one for segmentation image
    totalPlots = len(classes)+1
    if segmentImageOverlay is not None:
        totalPlots+=1
    nrows = math.ceil(totalPlots/ncols)
    
    grid = fig.add_gridspec(nrows,ncols)

    masks = generateUnaryMasks(segmentation=segmentation, width=width, height=height, segmentCount=len(classes))

    for row in range(nrows):
        for col in range(ncols):
            index = row*ncols+col
            if index < len(classes):
                ax = fig.add_subplot(grid[row,col])
                ax.imshow(masks[index])
                ax.set_title(classes[index])
    ax = fig.add_subplot(grid[len(classes)//ncols, len(classes) % ncols])
    ax.imshow(segmentation)
    ax.set_title('Segmentation')
    if segmentImageOverlay is not None:
        ax = fig.add_subplot(grid[totalPlots//ncols, (len(classes)+1) % ncols])
        ax.imshow(segmentImageOverlay)
        ax.set_title('Segmentation Overlay')
    