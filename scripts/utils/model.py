from torch.utils.data import DataLoader
import numpy as np
import mmcv
import torch


def get_wrongly_classified(model, dataset, gt=None):
    """
    :param gt: Optional path to ground truth file. If None dataset must hold gt.   
    """
    data_loader = DataLoader(dataset, batch_size=64, num_workers=4)
    model.eval()
    wrong_samples = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        batch_size = len(result)
        scores = np.vstack(result)
        pred_label = np.argmax(scores, axis=1)

        if gt:
            raise ValueError('gt not yet supported use Dataset that works.')
        wrong_samples.extend([data['name'][j] for j in batch_size if data['gt_target'][j] == pred_label[j]])

        batch_size = data['img'].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    
    return wrong_samples