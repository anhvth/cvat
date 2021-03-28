from .custom import *
import torch

@DATASETS.register_module()
class BinaryDataset(CustomDataset):

    CLASSES = ('background', 'foreground')
    PALETTE = [[0, 0, 0], [0,255,0]]
    
    def __init__(self, target_class_id, *args, **kwargs):
        super(BinaryDataset, self).__init__(*args, **kwargs)
        self.target_class_id = target_class_id


    def __getitem__(self, idx):
        rt = super(BinaryDataset, self).__getitem__(idx)
        if 'gt_semantic_seg' in rt:
            gt_mask = rt['gt_semantic_seg'].data
            bg_ids = torch.logical_and(gt_mask!=255, gt_mask!=self.target_class_id)
            rt['gt_semantic_seg'].data[bg_ids] = 0
        return rt