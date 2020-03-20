import json

import mrcnn.model as modellib
from coco.coco import CocoConfig, COCO_MODEL_PATH
from mrcnn.utils import *

ROOT_DIR = os.path.abspath("/")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_CLASSES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


class CocoInferenceConfig(CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def detect_coco(image):
    inference_config = CocoInferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    return model.detect([image], verbose=1)


def save_coco_result(filename, coco_result):
    mask = coco_result['masks']
    box = extract_bboxes(mask)

    coco_result_json = {
        'masks': minimize_mask(box, coco_result['masks'], CocoInferenceConfig.MINI_MASK_SHAPE).tolist(),
        'rois': coco_result['rois'].tolist(),
        'scores': coco_result['scores'].tolist(),
        'class_ids': coco_result['class_ids'].tolist(),
        'bbox': box.tolist()
    }

    with open(filename, "w") as write_file:
        json.dump(coco_result_json, write_file)


def read_coco_result(filename, image_shape):
    with open(filename, "r") as coco_file:
        coco_result_json = json.load(coco_file)
        bbox = np.array(coco_result_json['bbox'])

        coco_result = {
            'masks': expand_mask(bbox, np.array(coco_result_json['masks']), image_shape),
            'rois': np.array(coco_result_json['rois']),
            'scores': np.array(coco_result_json['scores']),
            'class_ids': np.array(coco_result_json['class_ids'])
        }
        return coco_result
