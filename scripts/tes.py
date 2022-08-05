from mmcv.utils import Registry
from mmseg.datasets.builder import DATASETS
from mmcls.datasets.builder import DATASETS as D2
from mmcls.datasets.compcars import CompCars
from mmseg.datasets.builder import build_dataset, build_dataloader
from mmcls.apis.test import single_gpu_test
from mmcls.models.builder import build_classifier

import mmcv
from mmcv.runner import load_checkpoint

#DATASETS = Registry('dataset')

# @DATASETS.register_module()
# class CompCars(P):
#     def test():
#         print("T")

@DATASETS.register_module('CompCars')
def CompCarsWrapper(**kwargs):
    return CompCars(**kwargs)

def main():
    print(DATASETS)
    print("#"*30)
    cfg = mmcv.Config.fromfile("../CAMModels/resnet/compCars_Original/resnet50_b128x2_compcars-original-split.py")
    print(cfg.data.test)
    cfg.data.test['data_prefix'] = "../data/CompCars_sv_original_split/val" 
    cfg.data.test['ann_file'] = "../data/CompCars_sv_original_split/meta/val.txt"
    print(cfg.data.test)
    dataset = build_dataset(cfg.data.test)
    print(dataset[0]['img'].shape)
    print(cfg.data.workers_per_gpu)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=64,
        workers_per_gpu=cfg.data.workers_per_gpu,
        shuffle=False)
    print(data_loader)
    print(len(dataset))
    model = build_classifier(cfg.model)
    #print(model)
    print("#"*30)
    checkpoint = load_checkpoint(model, "../CAMModels/resnet/compCars_Original/latest.pth", map_location='cpu')
    #print(checkpoint)
    print("#"*30)
    #outputs = single_gpu_test(model, data_loader)
    #print(outputs)
    # for i,d in enumerate(data_loader):
    #     if i % 100 == 0:
    #         print(i)
    # print("#"*30)
    # print(dataset)
    # print("ABC")
    # print(DATASETS)
    # print("-"*20)
    # print(D2)

if __name__ == '__main__':
    main()