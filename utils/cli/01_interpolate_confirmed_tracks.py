import time
import numpy as np
from core.definition import ResourceType
import os.path as osp
from glob import glob
from avcv.utils import get_name
from avcv.utils import memoize
import mmcv
from task_utils import create_task
from yolox.exp import get_exp
import torch
from loguru import logger
import os
import argparse
from avcv.coco import CocoDataset, video_to_coco, AvCOCO, get_overlap_rate, get_bboxes
import logging
import requests
import sys
from http.client import HTTPConnection
from core.core import CLI, CVAT_API_V1
from core.definition import parser
from env import *
from mmcv.ops import bbox_overlaps

log = logging.getLogger(__name__)

SERVER_HOST = CVAT_SERVER_HOST
SERVER_PORT = CVAT_PORT
AUTH = CVAT_AUTH




def read_zip(path):
    import zipfile
    import json
    archive = zipfile.ZipFile(path, 'r')
    data = archive.read('annotations/instances_default.json')
    return json.loads(data)


def update_annotation(data):
    if isinstance(data, dict):
        # Update cats
        for ann in data['annotations']:
            ann['category_id'] += 1
        for cat in data['categories']:
            cat['id'] += 1

        # Remove low score
        data['annotations'] = [
            ann for ann in data['annotations'] if ann['score'] > 0.65]
        return data
    elif isinstance(data, str):
        out_path = data.replace('.json', '_updated.json')
        data = mmcv.load(data)
        data_update = update_annotation(data)
        mmcv.dump(data_update, out_path)
        return out_path


class TrackUpdater:
    def __init__(self, track_anns, coco: AvCOCO):
        self.track_anns = track_anns
        self.coco = coco
        self.imgid2index = {x['id']: int(
            get_name(x['file_name'])) for x in self.coco.imgs.values()}
        self.index2imgid = {v: k for k, v in self.imgid2index.items()}

    def _update_detection_with_high_track_iou(self, interpolated_ann):
        interpolated_box = get_bboxes([interpolated_ann], 'xyxy')

        anns = self.coco.imgToAnns[interpolated_ann['image_id']]
        bboxes = get_bboxes(anns, 'xyxy')
        ious = bbox_overlaps(torch.from_numpy(interpolated_box).cuda(
        ), torch.from_numpy(bboxes).cuda()).cpu().flatten().numpy()
        if len(ious) > 0:
            max_iou_index = np.argmax(ious)

            if ious[max_iou_index] > 0.5:
                ann = anns[max_iou_index]
                logger.info("Update a detection in image {} -> track {}".format(
                    self.imgid2index[ann['image_id']], interpolated_ann['attributes']['track_id']))
                self.coco.anns[ann['id']
                               ]['attributes'] = interpolated_ann['attributes']

    def interpolate_with_confirm_tracks(self):
        for i, ann_from in enumerate(self.track_anns[:-1]):
            ann_to = self.track_anns[i+1]
            index_from = self._ann_2_video_index(ann_from)
            index_to = self._ann_2_video_index(ann_to)
            for index in range(index_from+1, index_to):
                interpolated_ann = ann_from.copy()
                alpha = (index-index_from) / (index_to-index_from)
                interpolated_ann['bbox'] = self._get_interpolated_box(
                    ann_from['bbox'], ann_to['bbox'], alpha)
                interpolated_ann['image_id'] = self.index2imgid[index]
                interpolated_ann['attributes']['keyframe'] = False
                interpolated_ann['id'] = None
                interpolated_ann['area'] = interpolated_ann['bbox'][-1] * \
                    interpolated_ann['bbox'][-2]
                # logger.info("Get interpolated annotation: {}".format(index))
                self._update_detection_with_high_track_iou(interpolated_ann)
        return self.coco

    def __len__(self):
        return self._ann_2_video_index(self.track_anns[-1])-self._ann_2_video_index(self.track_anns[0])

    def _ann_2_video_index(self, ann):
        return self.imgid2index[ann['image_id']]

    def _get_interpolated_box(self, box_a, box_b, alpha):

        box_a, box_b = np.array(box_a), np.array(box_b)
        box = ((1-alpha)*box_a+(alpha)*box_b)
        # logger.info("{}, {}, {} -> {}".format(alpha, box_a, box_b, box))
        return box.tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', help='task id')
    args = parser.parse_args()
    #--- Debug
    # args.task = 54

    with requests.Session() as session:
        session = requests.Session()
        # api = CVAT_API_V1("%s:%s" % (SERVER_HOST, SERVER_PORT), False)
        # cli = CLI(session, api, AUTH)
        api = CVAT_API_V1('%s:%s' % ('localhost', 8080), False)
        cli = CLI(session, api, ('anhvth', 'User@2020'))
        cvat_init = os.path.join(
            '.cache/01/', time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'cvat_init.zip')
        mmcv.mkdir_or_exist(os.path.dirname(cvat_init))
        cli.tasks_dump(54, 'COCO 1.0', cvat_init)
        coco = AvCOCO(read_zip(cvat_init))

        anns = coco.dataset['annotations']
        tracks = dict()
        for ann in anns:
            if 'track' in str(ann):
                track_id = ann['attributes']['track_id']
                if not track_id in tracks:
                    tracks[track_id] = []
                ann['file_name'] = coco.imgs[ann['image_id']]['file_name']
                tracks[track_id].append(ann)
        
        for track_id, track_anns in tracks.items():
            logger.info('Track Update: {}'.format(track_id))
            track_anns = list(sorted(tracks[0], key=lambda x: x['file_name']))
            coco = TrackUpdater(
                track_anns, coco).interpolate_with_confirm_tracks()
        out_path = f'.cache/01_interpolate/{args.task}.json'
        mmcv.mkdir_or_exist(osp.dirname(out_path))
        out_dict = dict(images=list(coco.imgs.values()), annotations=list(
            coco.anns.values()), categories=list(coco.cats.values()))
        mmcv.dump(out_dict, out_path)
        # cli.tasks_upload(54, 'COCO 1.0', out_path)
        # t0_interpolated = t0.interpolate_with_confirm_tracks()
        # tracks = list(sorted([ann for ann in anns if 'track' in str(ann)], key=lambda ann: ann['image_id']))
        # track_ids = set([ann['attributes']['track_id'] for ann in tracks])
