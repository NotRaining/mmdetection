from mmdet.apis import init_detector, inference_detector
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
device = '0'
model = init_detector(config_file, device)
inference_detector(model, 'demo/demo.jpg')
print('end testing')
