import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot

from avcv.vision import *

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    args = parser.parse_args()
    return args



def visualize(img, gt, palette, alpha=0.5):
    color_seg = gt_to_color_mask(gt)
    color_seg = color_seg[..., ::-1]
    img = img *alpha + color_seg * (1-alpha)
    img = img.astype(np.uint8)
    return img

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.train)
    print("Dataset length :", len(dataset))
    tensor_img_cfg = {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}
    palette = dataset.PALETTE
    if palette is None and hasattr(cfg, 'PALETTE'):
        palette = cfg.PALETTE

    # for item in dataset:
    for i in np.random.choice(len(dataset), 100):
        item = dataset.__getitem__(i)
        img = mmcv.tensor2imgs(item['img'].data[None], **tensor_img_cfg)[0]
        gt = item['gt_semantic_seg'].data[0] # -> [h,w]
        vis = visualize(img, gt, palette)
        filename = os.path.basename(item['img_metas'].data['filename'])
        img_out_path = f"./cache/{filename}"
        mmcv.imwrite(vis ,img_out_path)
        print(img_out_path)


if __name__ == '__main__':
    main()