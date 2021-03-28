
import base64
import io
import math
import os
# import numpy as np
# import sys
import os.path as osp
import time

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
# import panopticapi
# import requests
import torch
import torchvision.transforms as T
from avcv.vision import gt_to_color_mask
from panopticapi.utils import id2rgb, rgb2id
from PIL import Image
# from skimage.measure import approximate_polygon
from torch import nn
# from torchvision.models import resnet50
from avcv.utils import read_json
torch.set_grad_enabled(False)


def find_contours(thresh):
    """
        Get contour of a binary image
            Arguments:
                thresh: binary image
            Returns:
                Contours: a list of contour
                Hierarchy:

    """
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours



meta = read_json("coco_panoptic_meta.json")

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Detectron2 uses a different numbering scheme, we build a conversion table
coco2d2 = {}
count = 0
for i, c in enumerate(CLASSES):
    if c != "N/A":
        coco2d2[i] = count
        count += 1

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load(
    'facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
model.eval()


def inference_segmentor(model, image):
    im = Image.fromarray(image)
    img = transform(im).unsqueeze(0)
    out = model(img)
    result = postprocessor(out, torch.as_tensor(
        img.shape[-2:]).unsqueeze(0))[0]
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
    panoptic_seg_id = rgb2id(panoptic_seg)
    return panoptic_seg_id


def pred(data, debug=False):
    result = []
    if not isinstance(data, np.ndarray):
        buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
        image = np.array(Image.open(buf))
    else:
        image = data
    h, w = image.shape[:2]

    with torch.no_grad():
        print("Infer image:", image.shape)
        panoptic_seg_id = inference_segmentor(model, image)


    

    instance_ids = np.unique(panoptic_seg_id)
    segments_info = result["segments_info"]

    for instance_id, info in zip(instance_ids, segments_info):
        if not info['isthing']:
            cate = meta['stuff_dataset_id_to_contiguous_id'][str(info['category_id'])]
            name = meta['stuff_classes'][cate]
        else:
            cate = meta['thing_dataset_id_to_contiguous_id'][str(info['category_id'])]
            name = meta['thing_classes'][cate]
        info['class_name'] = name
        instance_mask = panoptic_seg_id == instance_id
        contours = find_contours(instance_mask)

        for contour in contours:
            contour = np.flip(contour, axis=1)
            # epsilon = 0.005 * cv2.arcLength(contour, True)
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            # cv2.drawContours(
            #     vis, [contour], -1, (int(color[0]), int(color[1]), int(color[2])), -1)
            # Approximate the contour and reduce the number of points
            # if len(contour) < 6:
            #     continue
            # label = self.labels[class_id]

            result.append({
                "confidence": str(0.9),
                "label": info['class_name'],
                "points": contour.ravel().tolist(),
                "type": "polygon",
            })

    print(result)
    return dict(status=True, result=result)


if __name__ == "__main__":
    img_path = "./sample.png"
    image = cv2.imread(img_path)[..., ::-1].copy()
    pred(image)

    # while True:
    #     if osp.exists('./data.pth'):
    #         data = torch.load("./data.pth")
    #         result = pred(data)
    #         torch.save(result, './result.pth')
    #         os.rename('./data.pth', './data.old.pth')
    #     time.sleep(0.2)
