import os
import sys
sys.path.insert(0, '/home/anhvth/gitprojects/traffic-sign-models-dev/2d-dets/yolox')
TEST_JSON='/data/tiny-tsd/annotations/mini_json.json'
# CVAT_USER="anhvth"
# CVAT_PASSWORD="User@2020"

CVAT_USER="anhvth"
CVAT_PASSWORD="User@2020"
CVAT_AUTH=f"{CVAT_USER}:{CVAT_PASSWORD}"
CVAT_PORT=8080
CVAT_SERVER_HOST = "localhost"
YOLO_CONF_THR=0.35