from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset

"""
NEEDS TO BE ADDED INTO mmseg.datasets 
AND ADDED INTO THE __INIT__ FILE IN THERE.
"""

@DATASETS.register_module()
class GenerationDataset(CustomDataset):
    """
    Testing
    """

    # Ensure that it uses .png and not .jpg
    def __init__(self, img_suffix='.png', **kwargs):
        super(GenerationDataset, self).__init__(img_suffix=img_suffix, **kwargs)
