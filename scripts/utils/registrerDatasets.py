from mmseg.datasets.builder import DATASETS

from mmcls.datasets.compcars import CompCars
from mmcls.datasets.compcarsWeb import CompCarsWeb


"""Here all model Classes used in classification must be registered into the DATASET Registry of mmseg"""
@DATASETS.register_module('CompCars')
def CompCarsWrapper(**kwargs):
    return CompCars(**kwargs)

@DATASETS.register_module('CompCarsWeb')
def CompCarsWrapper(**kwargs):
    return CompCarsWeb(**kwargs)
