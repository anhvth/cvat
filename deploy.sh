# sudo wget https://github.com/nuclio/nuclio/releases/download/1.5.8/nuctl-1.5.8-linux-amd64 -O /usr/local/bin/nuctl
# sudo chmod +x /usr/local/bin/nuctl

# docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml down | docker ps | grep nuclio | awk '{print $1}' | xargs -r docker kill
# docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d



# nuctl deploy --project-name cvat \
#    --path ./serverless/tensorflow/matterport/mask_rcnn/nuclio \
#    --platform local --base-image tensorflow/tensorflow:1.15.5-gpu-py3 \
#    --desc "GPU based implementation of Mask RCNN on Python 3, Keras, and TensorFlow." \
#    --image cvat/tf.matterport.mask_rcnn_gpu \
#    --triggers '{"myHttpTrigger": {"maxWorkers": 1}}' \
#    --resource-limit nvidia.com/gpu=1

# nuctl deploy --project-name cvat \
#   --path ./serverless/tensorflow/matterport/mask_rcnn/nuclio \
#   --resource-limit nvidia.com/gpu=1 --platform local



  # nuctl deploy --project-name cvat \
  #   --path serverless/pytorch/saic-vul/fbrs/nuclio \
  #   --resource-limit nvidia.com/gpu=1 --platform local


#############- Pytorch segmentor model -debuging
str='''
nuctl deploy --project-name cvat
  --path '$1'
  --resource-limit nvidia.com/gpu=1
'''
eval $str
echo $str

# nuctl deploy --project-name cvat \
#   --path serverless/pytorch/detector \
#   --resource-limit nvidia.com/gpu=1 --platform local --volume `pwd`/serverless/pytorch/detector:/opt/nuclio/common

