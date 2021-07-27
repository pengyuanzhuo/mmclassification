from mmcls.datasets import samplers
from .base_dataset import BaseDataset
from .builder import DATASETS

import numpy as np


@DATASETS.register_module()
class MyDataset(BaseDataset):

    CLASSES = ['0', '1', '2', '3']

    def load_annotations(self):
        assert self.ann_file is not None, "ann_file is None"

        with open(self.ann_file) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

