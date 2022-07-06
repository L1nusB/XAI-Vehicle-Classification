"""
COPY THIS INTO mmseg.datasets.pipelines.transforms
And add
ResizeCls, RandomCropCls, CenterCropCls into __init__
"""


@PIPELINES.register_module()
class LoadImageFromArray(object):
    """Load an image a np.ndarray
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = 'loadedFile'

        img = results

        results['filename'] = filename
        results['ori_filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
