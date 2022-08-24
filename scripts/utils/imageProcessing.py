from PIL import Image, ImageDraw
from matplotlib import cm
import numpy as np

def add_text(img, text, pos=(0,0), font=None, fill=None):
    """Adds the specified text onto the given img.
    Font and Fill can be specified

    :param img: PIL Image that will be modified
    :type img: PIL Image
    :param text: Text to be added to the image
    :type text: str
    :param pos: Position where the text will be written to, defaults to (0,0)
    :type pos: tuple(int,int), optional
    :param font: Font of the text. if not specified default will be used
    :param fill: Textcolor. if not specified default(white) will be used.
    """
    ImageDraw.Draw(img).text(pos, text, fill, font)

def concatenate_images(img1, img2, direction='horizontal'):
    """Concatenates two images along the specified direction.
    Both images must have matching height if concatenating horizontal or matching width if concatenating verical.

    :param img1: First PIL Image
    :type img1: PIL Image
    :param img2: Second PIL Image
    :type img2: PIL Image
    :param direction: Direction along which to concatenate. Either 'horizontal' or 'vertical', defaults to 'horizontal'
    :type direction: str, optional
    """
    assert direction == 'horizontal' or direction == 'vertical', f'Direction {direction} not valid. Must either be "vertical" or "horizontal"'
    if direction == 'horizontal':
        assert img1.height == img2.height, f'img Heights do not match: {img1.height} != {img2.height}'
        newImg = Image.new('RGB', (img1.width + img2.width, img1.height))
        newImg.paste(img1, (0,0))
        newImg.paste(img2, (img1.width, 0))
    else:
        assert img1.width == img2.width, f'img Heights do not match: {img1.width} != {img2.width}'
        newImg = Image.new('RGB', (img1.width, img1.height + img2.height))
        newImg.paste(img1, (0,0))
        newImg.paste(img2, (0, img1.height))
    return newImg

def convert_numpy_to_PIL(arr, colormap='viridis', channel_order='bgr'):
    """Converts a given numpy array into a PIL image.
    If the arr has a floating dtype it is assumed the values lie in the range [0,1]
    and will be multiplied by 255 and then converted to type uint8
    If the type is already uint8 no rescaling or else will be performed

    :param arr: Array to be transformed
    :type arr: np.ndarray
    :param colormap: Colormap that will be used if img has 2D Shape. Must be in matplotlib.cm
    :type colormap: str
    :param channel_order: (defaults to 'bgr') Order in which channels are arranged in numpy array.
                        Ignored if arr is only two dimensional
    :type channel_order: str
    """
    if len(arr.shape) == 2:
        # Convert to 3-Channel Image so we do not get grayscales.
        cmmap = getattr(cm, colormap)
        arr = cmmap(arr)
        channel_order = 'rgb' # Manually override channel_order since we add the channel here.
    assert len(arr.shape) == 3, f'Shape of array {arr.shape} is not 3 dimensional.'

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.uint8(arr*255)
    
    assert arr.dtype == np.uint8, f'Type of array {arr.dtype} is invalid. Must be np.uint8 or np.float32'

    rawImg = Image.fromarray(arr)
    channels = rawImg.split()
    r = channels[channel_order.index('r')]
    g = channels[channel_order.index('g')]
    b = channels[channel_order.index('b')]

    return Image.merge('RGB', (r,g,b))