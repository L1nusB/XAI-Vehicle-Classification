from .utils.pipeline import add_blurring_pipeline_step

from mmcls.models.builder import build_classifier
from mmcls.apis.test import single_gpu_test
from mmcls.datasets.builder import build_dataloader
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv import Config
from mmseg.datasets.builder import build_dataset

def generate_blurred(classifierCfg, classifierCheckpoint, blurredSegments, segData, imgRoot, annfile, segConfig, segCheckpoint, 
                    singleColor=None, saveDir='./blurredImgs', **kwargs):
    print(f'Creating blurred Images in {saveDir}')
    cfg = Config.fromfile(classifierCfg)
    model = build_classifier(cfg.model)
    load_checkpoint(model, classifierCheckpoint)
    model = MMDataParallel(model, device_ids=[0])

    blur_cfg = add_blurring_pipeline_step(cfg, blurredSegments, segData, saveDir=saveDir, segConfig=segConfig, logInfos=True,
                                        saveImgs=True, singleColor=singleColor, segCheckpoint=segCheckpoint, **kwargs)
    blur_cfg.data.test['data_prefix'] = imgRoot
    blur_cfg.data.test['ann_file'] = annfile
    dataset = build_dataset(blur_cfg.data.test)

    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu, 
            workers_per_gpu=cfg.data.workers_per_gpu,
            shuffle=False)
    
    single_gpu_test(model, data_loader)