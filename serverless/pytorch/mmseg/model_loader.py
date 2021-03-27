# # Copyright (C) 2020-2021 Intel Corporation
# #
# # SPDX-License-Identifier: MIT
import time
import os
# import numpy as np
# import sys
import os.path as osp
from skimage.measure import find_contours, approximate_polygon
# import tensorflow as tf
# MASK_RCNN_DIR = os.path.abspath(os.environ.get('MASK_RCNN_DIR'))
# if MASK_RCNN_DIR:
#     sys.path.append(MASK_RCNN_DIR)  # To find local version of the library
# from mrcnn import model as modellib
# from mrcnn.config import Config


# class ModelLoader:
#     def __init__(self, labels):
#         COCO_MODEL_PATH = os.path.join(MASK_RCNN_DIR, "mask_rcnn_coco.h5")
#         if COCO_MODEL_PATH is None:
#             raise OSError('Model path env not found in the system.')

#         class InferenceConfig(Config):
#             NAME = "coco"
#             NUM_CLASSES = 1 + 80  # COCO has 80 classes
#             GPU_COUNT = 1
#             IMAGES_PER_GPU = 1

#         # Limit gpu memory to 30% to allow for other nuclio gpu functions. Increase fraction as you like
#         import keras.backend.tensorflow_backend as ktf
#         def get_session(gpu_fraction=0.333):
#             gpu_options = tf.GPUOptions(
#             per_process_gpu_memory_fraction=gpu_fraction,
#             allow_growth=True)
#             return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

#         ktf.set_session(get_session())
#         # Print config details
#         self.config = InferenceConfig()
#         self.config.display()

#         self.model = modellib.MaskRCNN(mode="inference",
#             config=self.config, model_dir=MASK_RCNN_DIR)
#         self.model.load_weights(COCO_MODEL_PATH, by_name=True)
#         self.labels = labels

#     def infer(self, image, threshold):
#         output = self.model.detect([image], verbose=1)[0]

#         result = []
#         MASK_THRESHOLD = 0.5
#         for i in range(len(output["rois"])):
#             score = output["scores"][i]
#             class_id = output["class_ids"][i]
#             mask = output["masks"][:, :, i]
#             if score >= threshold:
#                 mask = mask.astype(np.uint8)
#                 contours = find_contours(mask, MASK_THRESHOLD)
#                 # only one contour exist in our case
#                 contour = contours[0]
#                 contour = np.flip(contour, axis=1)
#                 # Approximate the contour and reduce the number of points
#                 contour = approximate_polygon(contour, tolerance=2.5)
#                 if len(contour) < 6:
#                     continue
#                 label = self.labels[class_id]

#                 result.append({
#                     "confidence": str(score),
#                     "label": label,
#                     "points": contour.ravel().tolist(),
#                     "type": "polygon",
#                 })

#         return result

# CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
#             'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
#             'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
#             'bicycle')

# PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
#             [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
#             [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
#             [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
#             [0, 80, 100], [0, 0, 230], [119, 11, 32]]


from mmseg.apis import *
from mmseg.datasets import CityscapesDataset
import numpy as np
import cv2
import torch
from avcv.vision import gt_to_color_mask
import base64
from PIL import Image
import io



cfg_path = "./mmsegmentation/configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py"
# ckpt = "https://download.openmmlab.com/mmsegmentation/v0.5/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth"
ckpt= "/ckpt.pth"
model = init_segmentor(cfg_path, ckpt)

CLASSES = CityscapesDataset.CLASSES
PALETTE = CityscapesDataset.PALETTE

def pred(data):
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    image = Image.open(buf)
    result = []
    image = np.array(image)
    with torch.no_grad():
        pred = inference_segmentor(model, image)[0]
    for class_id, name in enumerate(CLASSES):
        mask = ((pred)==class_id).astype('uint8')*255
        contours = find_contours(mask, 0.5)
        # only one contour exist in our case
        if len(contours):
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            # Approximate the contour and reduce the number of points
            contour = approximate_polygon(contour, tolerance=2.5)
            if len(contour) < 6:
                continue
            # label = self.labels[class_id]

            result.append({
                "confidence": str(0.9),
                "label": name,
                "points": contour.ravel().tolist(),
                "type": "polygon",
            })
    return dict(status=True, result=result)
    
if __name__ == "__main__":
    while True:
        if osp.exists('./data.pth'):
            data = torch.load("./data.pth")
            result = pred(data)
            torch.save(result, './result.pth')
            os.rename('./data.pth', './data.old.pth')
        time.sleep(0.2)