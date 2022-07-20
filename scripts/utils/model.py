import torch

from .constants import MODELWRAPPERS



def wrap_model(model, use_threshold=False, threshold=0.7, backgroundIndex=-1):
    modelCfg = model.cfg.model
    if modelCfg.type in MODELWRAPPERS:
        base = MODELWRAPPERS[modelCfg.type]
    else:
       raise KeyError(f'Given model type {modelCfg.type} is not tested/supported.Supported types are ' + ",".join(MODELWRAPPERS.keys()))
    
    # Remove the type entry in model.cfg.model since this does not allow for inits via super
    modelCfg = {key:values for key,values in modelCfg.items() if key != 'type'}


    class ModelWrapper(base):
        def __init__(self, model,  use_threshold=False, threshold=0.7, backgroundIndex=-1, **kwargs):
            super(ModelWrapper, self).__init__(**kwargs)
            self.use_threshold=use_threshold
            self.threshold = threshold
            self.backgroundIndex = backgroundIndex

            """
            Optionally add ALL other params here.
            """
            for key, value in model.__dict__.items():
                setattr(self, key, value)



        def simple_test(self, img, img_meta, rescale=True):
            if self.aug_test:
                seg_logit = self.inference(img, img_meta, rescale)
                seg_maxima = seg_logit.max(dim=1)
                seg_maxima.indices[seg_maxima.values < self.threshold] = self.backgroundIndex
                seg_pred = seg_maxima.indices
                #seg_pred = seg_logit.argmax(dim=1)
                if torch.onnx.is_in_onnx_export():
                    # our inference backend only support 4D output
                    seg_pred = seg_pred.unsqueeze(0)
                    return seg_pred
                seg_pred = seg_pred.cpu().numpy().astype('uint8')
                # unravel batch dim
                seg_pred = list(seg_pred)
                return seg_pred
            else:
                return super().simple_test(self, img, img_meta, rescale)

        def aug_test(self, imgs, img_metas, rescale=True):
            if self.aug_test:
                assert rescale
                # to save memory, we get augmented seg logit inplace
                seg_logit = self.inference(imgs[0], img_metas[0], rescale)
                for i in range(1, len(imgs)):
                    cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
                    seg_logit += cur_seg_logit
                seg_logit /= len(imgs)
                seg_maxima = seg_logit.max(dim=1)
                seg_maxima.indices[seg_maxima.values < self.threshold] = self.backgroundIndex
                seg_pred = seg_maxima.indices
                seg_pred = seg_pred.cpu().numpy().astype('uint8')
                # unravel batch dim
                seg_pred = list(seg_pred)
                return seg_pred
            else:
                return super().simple_test(self, imgs, img_metas, rescale)

    
    return ModelWrapper(model, use_threshold=use_threshold, threshold=threshold, backgroundIndex=backgroundIndex, **modelCfg)