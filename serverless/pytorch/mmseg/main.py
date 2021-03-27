import os
import json
# import base64
# from PIL import Image
# import io
# from model_loader import ModelLoader
# import numpy as np
# import yaml
# import pickle
import torch
import time
import os.path as osp
def init_context(context):
    context.logger.info("Init context...  0%")

    # functionconfig = yaml.safe_load(open("/opt/nuclio/function.yaml"))
    # labels_spec = functionconfig['metadata']['annotations']['spec']
    # labels = {item['id']: item['name'] for item in json.loads(labels_spec)}

    # model_handler = ModelLoader(labels)
    # setattr(context.user_data, 'model_handler', model_handler)

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run mmseg model")
    data = event.body
    results = dict()
    torch.save(data,'./data.pth')
    start = time.time()
    while True:
        if osp.exists('./result.pth'):
            try:
                results = torch.load('./result.pth')
                results['status'] = True
                os.rename('./result.pth', './result.old.pth')
                break
            except:
                # context.logger.
                if time.time() - start > 30:
                    break

        # finally:
        time.sleep(0.2)
    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)