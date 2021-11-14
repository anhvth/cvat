import io, csv
import xmltodict
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
import pandas
import dicttoxml
log = logging.getLogger(__name__)

SERVER_HOST = CVAT_SERVER_HOST
SERVER_PORT = CVAT_PORT
AUTH = CVAT_AUTH


def read_zip(path, format='coco'):
    import zipfile
    import json
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
    def __init__(self, track_anns, coco):
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

        interpolated_box = self._get_bbox_from_xml(interpolated_ann, 'tlbr')
        img_id = self.index2imgid[int(interpolated_ann['@frame'])]
        anns = self.coco.imgToAnns[img_id]
        bboxes = get_bboxes(anns, 'xyxy')
        ious = bbox_overlaps(torch.from_numpy(interpolated_box).cuda(
        ), torch.from_numpy(bboxes).cuda()).cpu().flatten().numpy()
        if len(ious) > 0:
            max_iou_index = np.argmax(ious)

            if ious[max_iou_index] > 0.5:
                ann = anns[max_iou_index]
                x, y, w, h = ann['bbox']
                keys = ['@xtl', '@ytl', '@xbr', '@ybr']
                interpolated_ann['@keyframe'] = '1'
                logger.info('\t\t Update using detection')
                for i, key in enumerate(keys):
                    interpolated_ann[key] = ann['bbox'][i]
        return interpolated_ann
        #         logger.info("Update a detection in image {} -> track {}".format(
        #             self.imgid2index[ann['image_id']], interpolated_ann['attributes']['track_id']))
        #         self.coco.anns[ann['id']
        #                        ]['attributes'] = interpolated_ann['attributes']

    def interpolate_with_confirm_tracks(self):
        interpolated_boxes = []
        for i, ann_from in enumerate(self.track_anns[:-1]):
            ann_to = self.track_anns[i+1]
            index_from = int(ann_from['@frame'])
            index_to = int(ann_to['@frame'])
            for index in range(index_from+1, index_to):
                alpha = (index-index_from) / (index_to-index_from)
                ann_interpolated = self._get_interpolated_box(
                    ann_from, ann_to, alpha, index)
                # interpolated_boxes += [ann_interpolated]
                logger.info(
                    'Track Update frame {} -> to track None'.format(index))
                # interpolated_ann['image_id'] = self.index2imgid[index]
                # interpolated_ann['attributes']['keyframe'] = False
                # interpolated_ann['id'] = None
                # interpolated_ann['area'] = interpolated_ann['bbox'][-1] * \
                #     interpolated_ann['bbox'][-2]
                # logger.info("Get interpolated annotation: {}".format(index))
                ann_interpolated_updated = self._update_with_high_iou_det(
                    ann_interpolated)
                logger.info('interpolated_boxes: {}'.format(
                    len(interpolated_boxes)))
                interpolated_boxes.append(ann_interpolated_updated)
                # if matching_det is not None:
                # logger.info('Found matching iou frame id:{}\n'.format(index))
        return interpolated_boxes

    def __len__(self):
        return self._ann_2_video_index(self.track_anns[-1])-self._ann_2_video_index(self.track_anns[0])

    def _ann_2_video_index(self, ann):
        return self.imgid2index[ann['image_id']]

    def _get_interpolated_box(self, box_a, box_b, alpha, frame_id):
        interpolated_box = box_a.copy()
        keys = ['@xtl', '@ytl', '@xbr', '@ybr']
        for key in keys:
            f_val = ((1-alpha)*float(box_a[key])+(alpha)*float(box_b[key]))
            interpolated_box[key] = str(f_val)
        interpolated_box['@frame'] = str(frame_id)
        interpolated_box['@keyframe'] = '0'
        return interpolated_box


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
        out_dir = osp.join('.cache/01_CVAT_for_video_1.1/', time.strftime(
            "%Y_%m_%d_%H_%M_%S", time.localtime()))
        ann_cvat_before = os.path.join(out_dir, 'before_ann_cvat.zip')
        mmcv.mkdir_or_exist(os.path.dirname(ann_cvat_before))
        cli.tasks_dump(args.task, 'CVAT for video 1.1', ann_cvat_before)
        logger.info('ann_cvat_before: {}'.format(ann_cvat_before))
        xml_data_str = read_zip(ann_cvat_before, 'CVAT')
        xml_data_dict = xmltodict.parse(xml_data_str)

        cli.tasks_dump(args.task, 'COCO 1.0', ann_cvat_before)
        coco = AvCOCO(read_zip(ann_cvat_before, 'coco'))
        os.remove(ann_cvat_before)




        # tracks = dict()
        for track in xml_data_dict['annotations']['track']:
            if len(track['box']):
                # tracks[track['@id']] =
                updated_track_anns = TrackUpdater(
                    track['box'], coco).interpolate_with_confirm_tracks()

                # boxes = track['box']+updated_track_anns
                # boxes = list(sorted(boxes, key=lambda x: x['@frame']))
                # track['box'] = boxes

        # with open(xml_data_dict, 'w') as f:
        #     for track in xml_data_dict['annotations']['track']:
        #         if len(track['box']):
        #             f.write('')
        # file_object = open('.cache/out.txt', 'wb')
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

        name2clsid = {cat['name']:id for id, cat in coco.cats.items()}
        with io.TextIOWrapper(open('.cache/out.txt', 'wb'), encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=MOT)
            for i, track in enumerate(reversed(xml_data_dict['annotations']['track'])):
                for shape in track['box'][:-1]:
                    # MOT doesn't support polygons or 'outside' property
                    # if shape.type != 'rectangle':
                        # continue
                    writer.writerow({
                        "frame_id": int(shape['@frame']),
                        "track_id": i+1 if len(track['box'])>2 else -1,
                        "xtl":    float(shape['@xtl']),
                        "ytl":    float(shape['@ytl']),
                        "width":  float(shape['@xbr'])- float(shape['@xtl']),
                        "height": float(shape['@ybr'])- float(shape['@ytl']),
                        "confidence": 1,
                        "class_id": name2clsid[track['@label']],
                        "visibility": float(1 - int(shape['@occluded']))
                    })

        # updated_xml_data_str = dicttoxml.dicttoxml(xml_data_dict)
        # out_file = osp.join(out_dir, 'cvat_updated_xml.xml')
        # with open(out_file, "w+") as f:
            # f.write(str(xml_data_str))
        # logger.info('After update: {}'.format(out_file))

        # cli.tasks_upload(args.task, 'CVAT for video 1.1', update_annotation(out_file))
        # updated_tracks = dict()
        # for track_id, track_anns in tracks.items():

        # import ipdb
        # ipdb.set_trace()
        # updated_tracks[track_id] = updated_track_anns
