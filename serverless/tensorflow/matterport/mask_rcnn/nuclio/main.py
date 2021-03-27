import json
import base64
from PIL import Image
import io
from model_loader import ModelLoader
import numpy as np
import yaml
import pickle

def init_context(context):
    context.logger.info("Init context...  0%")

    functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    model_handler = ModelLoader(labels)
    setattr(context.user_data, 'model_handler', model_handler)

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run tf.matterport.mask_rcnn model")
    data = event.body
    try:
        pickle.dumps(data, open('data.pkl', 'wb'))
    except:
        pass
    buf = io.BytesIO(base64.b64decode(data["image"].encode('utf-8')))
    threshold = float(data.get("threshold", 0.2))
    image = Image.open(buf)

    results = context.user_data.model_handler.infer(np.array(image), threshold)
    try:
        pickle.dump(results, open('results.pkl', 'wb'))
    except:
        pass
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)