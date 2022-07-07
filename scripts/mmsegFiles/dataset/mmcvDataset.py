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

    # Here one could set the img_suffix
    def __init__(self, img_suffix='', **kwargs):
        super(GenerationDataset, self).__init__(img_suffix=img_suffix, **kwargs)
