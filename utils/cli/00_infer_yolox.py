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
from avcv.coco import CocoDataset, video_to_coco
import logging
import requests
import sys
from http.client import HTTPConnection
from core.core import CLI, CVAT_API_V1
from core.definition import parser
from env import *


log = logging.getLogger(__name__)


# load_dotenv()
SERVER_HOST = CVAT_SERVER_HOST
SERVER_PORT = CVAT_PORT
AUTH = CVAT_AUTH


# TEST_JSON = os.environ.get('TEST_JSON')
assert TEST_JSON is not None, "check .env"
# -----

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
        data['annotations'] = [ann for ann in data['annotations'] if ann['score'] > YOLO_CONF_THR]
        return data
    elif isinstance(data, str):
        out_path = data.replace('.json', '_updated.json')
        data = mmcv.load(data)
        data_update = update_annotation(data)
        mmcv.dump(data_update, out_path)
        return out_path
        



class VideoPrediction:
    def __init__(self, model_name='/l.py', ckpt_path='/ckpts/l.pth'):
        self.exp = get_exp(None, model_name)
        self.model = self.exp.get_model().cuda()
        self.model.load_state_dict(torch.load(ckpt_path)['model'])

    @memoize
    def get_prediction(self, input_video):
        json_path, images_path = video_to_coco(input_video, TEST_JSON)
        logger.info('1. Runing detection model')
        self.exp.data_dir = json_path.split('annotations')[0]
        self.exp.val_ann = os.path.basename(json_path)
        self.exp.val_img_dir = os.path.basename(images_path)
        evaluator = self.exp.get_evaluator(4, False, True)
        # video_path = '/temp/f0.mp4'
        # evaluator, json_path, images_path = get_evaluator(video_path, exp)
        results = evaluator.get_prediction(
            self.model, False, False, None, None, self.exp.test_size, out_file=None,
        )

        ds = CocoDataset(json_path, images_path)
        ds.pred = ds.gt.loadRes(results[0])
        return ds.pred, json_path, images_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=None, help='Input video')
    parser.add_argument('--task', '-t',default=None, help='task id')
    args = parser.parse_args()
    #--- Debug
    video_predictor = VideoPrediction()

    coco_pred, json_path, images = video_predictor.get_prediction(args.input)

    pred_json_path = json_path.replace('.json', '_pred.json')
    mmcv.dump(coco_pred.dataset, pred_json_path)

    task_name = get_name(args.input)
    img_paths = glob(osp.join(images, '*.jpg'))

    with requests.Session() as session:
        session = requests.Session()
        api = CVAT_API_V1('%s:%s' % ('localhost', 8080), False)
        cli = CLI(session, api, ('anhvth', 'User@2020'))
        if args.task is None: # Create 
            assert args.input is not None, "Task and input can not be both None"
            cli.tasks_create(
                task_name,labels=[], overlap=0, segment_size=0,bug='',
                project_id=37,
                resource_type=ResourceType.LOCAL,
                resources=img_paths,
                annotation_format="COCO 1.0",
                annotation_path=update_annotation(pred_json_path),
            )
        else: # update with task id
            cli.tasks_upload(args.task, 'COCO 1.0', update_annotation(pred_json_path))
