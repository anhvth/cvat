import argparse
import csv
import io
import logging
import os
import os.path as osp
import sys
import time
from glob import glob
from http.client import HTTPConnection

import dicttoxml
import mmcv
import numpy as np
import pandas
import requests
import torch
import xmltodict
from avcv.coco import (AvCOCO, CocoDataset, get_bboxes, get_overlap_rate,
                       video_to_coco)
from avcv.utils import get_name, memoize
from core.core import CLI, CVAT_API_V1
from core.definition import ResourceType, parser
from env import *
from loguru import logger
from mmcv.ops import bbox_overlaps
from task_utils import create_task
from yolox.exp import get_exp

log = logging.getLogger(__name__)

SERVER_HOST = CVAT_SERVER_HOST
SERVER_PORT = CVAT_PORT
AUTH = CVAT_AUTH


def read_zip(path, format='coco'):
    import json
    import zipfile
    archive = zipfile.ZipFile(path, 'r')
    if 'coco' in format:
        data = archive.read('annotations/instances_default.json')
        return json.loads(data)
    elif 'CVAT' in format:
        data = archive.read('annotations.xml')
        return data


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
    def __init__(self, track_anns, coco, track_id):
        self.track_id = track_id
        self.track_anns = track_anns
        self.coco = coco
        self.imgid2index = {x['id']: int(
            get_name(x['file_name'])) for x in self.coco.imgs.values()}
        self.index2imgid = {v: k for k, v in self.imgid2index.items()}

    def _get_bbox_from_xml(self, interpolated_ann, mode='tlbr'):
        assert mode in ['tlbr', 'xyxy', 'xywh']
        keys = ['@xtl', '@ytl', '@xbr', '@ybr']
        x1, y1, x2, y2 = [float(interpolated_ann[key]) for key in keys]
        if mode == 'tlbr':
            return np.array([[x1, y1, x2, y2]])
        elif mode == 'xywh':
            return np.array([[x1, y1, x2-x1, y2-y1]])
        else:
            raise NotImplementedError()

    def _update_with_high_iou_det(self, interpolated_ann):

        interpolated_box = get_bboxes([interpolated_ann], 'xyxy')

        img_id = interpolated_ann['image_id']
        anns = self.coco.imgToAnns[img_id]
        bboxes = get_bboxes(anns, 'xyxy')
        ious = bbox_overlaps(torch.from_numpy(interpolated_box).cuda(
        ), torch.from_numpy(bboxes).cuda()).cpu().flatten().numpy()
        is_replace = False
        if len(ious) > 0:
            max_iou_index = np.argmax(ious)
            
            if ious[max_iou_index] > 0.3:
                ann = anns[max_iou_index]
                interpolated_ann['keyframe'] = True
                interpolated_ann['bbox'] = ann['bbox']
                is_replace = ann['id']
        return interpolated_ann, is_replace

    def interpolate_with_confirm_tracks(self):
        interpolated_boxes = []
        tobe_remove_ann_ids = []
        track_anns = sorted(
            self.track_anns, key=lambda x: self.imgid2index[x['image_id']])
        for i, ann_from in enumerate(track_anns[:-1]):
            ann_to = track_anns[i+1]
            index_from = self.imgid2index[ann_from['image_id']]
            index_to = self.imgid2index[ann_to['image_id']]
            for index in range(index_from, index_to):
                alpha = (index-index_from+1) / (index_to-index_from)
                ann_interpolated = self._get_interpolated_box(
                    ann_from, ann_to, alpha, index)

                ann_interpolated_updated, tobe_remove_ann_id = self._update_with_high_iou_det(
                    ann_interpolated)
                if tobe_remove_ann_id:
                    tobe_remove_ann_ids += [tobe_remove_ann_id]
                    logger.info(
                        'Track Update frame {} -> to ann {}'.format(index, tobe_remove_ann_id))
                    index_from = index
                    ann_from = ann_interpolated_updated

                interpolated_boxes.append(ann_interpolated_updated)
        return interpolated_boxes, tobe_remove_ann_ids

    def __len__(self):
        return self._ann_2_video_index(self.track_anns[-1])-self._ann_2_video_index(self.track_anns[0])

    def _ann_2_video_index(self, ann):
        return self.imgid2index[ann['image_id']]

    def _get_interpolated_box(self, box_a, box_b, alpha, frame_id):
        interpolated_box = box_a.copy()
        interpolated_box['image_id'] = self.index2imgid[frame_id]
        b1 = np.array(box_a['bbox'])
        b2 = np.array(box_b['bbox'])
        interpolated_box['bbox'] = ((1-alpha)*b1 + alpha*b2).tolist()
        interpolated_box['keyframe'] = False
        return interpolated_box


def dump_mot(out_dir, rows):
    MOT = [
        "frame_id",
        "track_id",
        "xtl",
        "ytl",
        "width",
        "height",
        "confidence",
        "class_id",
        "visibility"
    ]

    mot_zip_in = osp.join(out_dir, 'in_MOT_1.1.zip')
    mot_zip_out = osp.join(out_dir, 'out_MOT_1.1')
    mmcv.mkdir_or_exist(mot_zip_out)
    cli.tasks_dump(args.task, 'MOT 1.1', mot_zip_in)
    os.system(f'unzip {mot_zip_in} -d {mot_zip_out}')
    out_gt_txt = osp.join(mot_zip_out, 'gt', 'gt.txt')
    if osp.exists(out_gt_txt):
        os.remove(out_gt_txt)

    with io.TextIOWrapper(open(out_gt_txt, 'wb'), encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MOT)

        for row in reversed(sorted(rows, key=lambda x: x['frame_id'])):
            writer.writerow(row)
    logger.info('generated gt: '+out_gt_txt)
    out_zip_path = osp.abspath(f'{mot_zip_out}.zip')
    mmcv.mkdir_or_exist(osp.dirname(out_zip_path))
    os.system(f'cd {mot_zip_out} && zip -r {out_zip_path} gt')
    return out_zip_path



def download_annotations(cli, task, format='COCO 1.0'):
    _tmp = os.path.join(out_dir, 'before_ann_cvat.zip')
    mmcv.mkdir_or_exist(os.path.dirname(_tmp))
    cli.tasks_dump(task, format, _tmp)
    if format == 'COCO 1.0':
        data = read_zip(_tmp, 'coco')
        coco = AvCOCO(data)
        new_dict = dict(images=[], annotations=[], categories=coco.dataset['categories'])
        for img_id, anns in coco.imgToAnns.items():
            img = coco.imgs[img_id]
            img['id'] = int(get_name(img['file_name']))
            new_dict['images'].append(img)
            for ann in anns:
                ann['image_id'] = img['id']
                new_dict['annotations'].append(ann)
        os.remove(_tmp)
        return AvCOCO(new_dict)
    else:
        return _tmp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', required=True, help='task id')
    parser.add_argument('--undo', '-u', default=False,
                        action='store_true', help='Undo this step')
    args = parser.parse_args()
    #--- Debug


    with requests.Session() as session:
        session = requests.Session()
        api = CVAT_API_V1('%s:%s' % ('localhost', 8080), False)
        cli = CLI(session, api, ('anhvth', 'User@2020'))
        out_dir = osp.join('.cache/01/', time.strftime(
            "%Y_%m_%d_%H_%M_%S", time.localtime()))


        memoized_download = memoize(download_annotations)
        mot_resverd_path = memoized_download(cli, args.task, 'MOT 1.1')
        if args.undo:
            # uploade back
            logger.info("Reserved")
            cli.tasks_upload(args.task, 'MOT 1.1', mot_resverd_path)
                
        else:
            coco = download_annotations(cli, args.task)
            # ann_with_tracks = [ann for ann in coco.anns.values() if 'track' in str(ann)]
            # import ipdb; ipdb.set_trace()
            # -------------------
            tracks = {-1: []}
            for ann in coco.anns.values():
                if 'track' in str(ann):
                    track_id = ann['attributes']['track_id']
                    if not track_id in tracks:
                        tracks[track_id] = []
                    tracks[track_id] += [ann]
                else:
                    tracks[-1] += [ann]

            # Update
            set_tobe_remove_ann_ids = set()
            for track_id, track_anns in tracks.items():
                if track_id != -1:
                    updated_track_anns, tobe_remove_ann_ids = TrackUpdater(
                        track_anns, coco, track_id).interpolate_with_confirm_tracks()

                    set_tobe_remove_ann_ids = set_tobe_remove_ann_ids.union(set(tobe_remove_ann_ids))
                    track_anns = track_anns + updated_track_anns
                    
            mot_rows_as_dict = []
            for track_id, track_anns in tracks.items():
                for ann in track_anns:
                    if not ann['id']  in set_tobe_remove_ann_ids:
                        img = coco.imgs[ann['image_id']]
                        frame_id = int(get_name(img['file_name']))
                        x, y, w, h = ann['bbox']
                        mot_rows_as_dict.append({
                            "frame_id": frame_id,
                            "track_id": 1+track_id if track_id != -1 else track_id,
                            "xtl":    x,
                            "ytl":    y,
                            "width":  w,
                            "height": h,
                            "confidence": 1,
                            "class_id": ann['category_id'],
                            "visibility": float(1)
                        })

            out_zip_path = dump_mot(out_dir, mot_rows_as_dict)

            logger.info('After update: {}'.format(out_zip_path))

            cli.tasks_upload(args.task, 'MOT 1.1', out_zip_path)
