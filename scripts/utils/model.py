import torch
import os.path as osp
import warnings
import mmcv
import tempfile
import numpy as np

from .constants import MODELWRAPPERS

from mmcv.image import tensor2imgs

def single_gpu_test_thresh(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    threshold=0.1,
                    backgroundIndex=-1):
    """Test with single GPU by progressive mode.
    Uses inference method and assigns pixel responses under
    threshold to specified 'background' category.


    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
        threshold (float): Threshold until response is categorised as background.
        backgroundIndex (int): Index of background category.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            _,scattered_data = model.scatter(None, data, model.device_ids)
            result = forward_test_thres(model=model.module, threshold=threshold, background=backgroundIndex, **scattered_data[0])
            #result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results

def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


"""
The Methods below are taken from BaseSegmentor and Encoder_Decoder and
only slightly modified.
These methods apply a cutoff for the given threshold and assign those 
below the threshold to the given background class.
"""

def forward_test_thres(model, img, img_metas, threshold, background, **kwargs):
        """
        Args:
            img (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(img, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return simple_test_thres(model, img[0], img_metas[0], threshold, background, **kwargs)
        else:
            return aug_test_thres(model, img, img_metas, threshold, background, **kwargs)

def simple_test_thres(model, img, img_meta, threshold, background, rescale=True):
    """Simple test with single image."""
    seg_logit = model.inference(img, img_meta, rescale)
    seg_maxima = seg_logit.max(dim=1)
    seg_maxima.indices[seg_maxima.values < threshold] = background
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

def aug_test_thres(model, imgs, img_metas, threshold, background, rescale=True):
    """Test with augmentations.

    Only rescale=True is supported.
    """
    # aug_test rescale all imgs back to ori_shape for now
    assert rescale
    # to save memory, we get augmented seg logit inplace
    seg_logit = model.inference(imgs[0], img_metas[0], rescale)
    for i in range(1, len(imgs)):
        cur_seg_logit = model.inference(imgs[i], img_metas[i], rescale)
        seg_logit += cur_seg_logit
    seg_logit /= len(imgs)
    seg_maxima = seg_logit.max(dim=1)
    seg_maxima.indices[seg_maxima.values < threshold] = background
    seg_pred = seg_maxima.indices
    seg_pred = seg_pred.cpu().numpy().astype('uint8')
    # unravel batch dim
    seg_pred = list(seg_pred)
    return seg_pred


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