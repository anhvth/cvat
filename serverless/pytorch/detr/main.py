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
from model_loader import pred
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
    # output['status'] = True
    output = pred(data)
    result = output['result']
    context.logger.info("result"+str(result))
    return context.Response(body=json.dumps(result), headers={},
        content_type='application/json', status_code=200)

# def handler(context, event):
#     context.logger.info("Run mmseg model")
#     data = event.body
#     result = dict()
#     torch.save(data,'./data.pth')
    # start = time.time()
    # while True:
    #     abs_result_path = osp.abspath("./result.pth")
    #     context.logger.info(f"Looking for {abs_result_path}")
    #     if osp.exists('./result.pth'):
    #         try:
    #             output = torch.load('./result.pth')
    #             output['status'] = True
    #             result = output['result']
    #             os.rename('./result.pth', './result.old.pth')
    #             context.logger.info("Found result -> Return")
    #             break
    #         except:
    #             # context.logger
    #             if time.time() - start > 30:
    #                 context.logger.info("Timout :( stop waiting")
    #                 break

    #     # finally:
    #     time.sleep(0.2)
    # return context.Response(body=json.dumps(result), headers={},
    #     content_type='application/json', status_code=200)