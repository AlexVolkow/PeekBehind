import json
import os

import numpy as np
import skimage.draw
import skimage.io

import mrcnn.model as modellib
from coco.coco import CocoDataset
from coco_annotator import COCO_CLASS_IDS
from coco_utils import COCO_CLASSES, read_coco_result
from mrcnn.config import Config

DATASET_PATH = "/sign_dataset"

ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

ROAD_SIGN_CLASS = 91


class RoadSignsConfig(Config):
    NAME = "coco"

    NUM_CLASSES = 1 + 1 + len(COCO_CLASS_IDS)  # Background + road sign

    STEPS_PER_EPOCH = 100

    DETECTION_MIN_CONFIDENCE = 0.8


class RoadSignsDataset(CocoDataset):

    def load_signs(self, dataset_dir, subset):
        assert subset in ["train", "val"]

        # load coco classes
        for class_id in COCO_CLASS_IDS[:-1]:
            self.add_class("coco", class_id, COCO_CLASSES[class_id])

        self.add_class("coco", 13, "stop sign")

        # load my classes
        self.add_class("coco", ROAD_SIGN_CLASS, "road_sign")

        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())

        annotations = [a for a in annotations if a['regions']]

        have_classes = set()
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])

            width = a['file_attributes']['width']
            height = a['file_attributes']['height']

            coco_result_file = os.path.join(dataset_dir, a['filename'] + "_coco.json")

            if os.path.exists(coco_result_file):
                coco_result = read_coco_result(coco_result_file, (height, width))
            else:
                print("No coco annotation")
                continue

            have_classes.update(coco_result['class_ids'])

            self.add_image(
                "coco",
                image_id=a['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                coco_mask=coco_result['masks'],
                coco_class_ids=coco_result['class_ids'])

        print("Have classes {}: {}".format(len(have_classes), str(have_classes)))
        print(self.class_info)

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            if p['name'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
            elif p['name'] == 'circle':
                rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'])
            elif p['name'] == 'rect':
                rr, cc = skimage.draw.rectangle((p['y'], p['x']), (p['y'] + p['height'], p['x'] + p['width']))
            else:
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            rr = (np.vectorize(lambda x: min(x, info["height"] - 1))(rr)).astype(np.int32)
            cc = (np.vectorize(lambda x: min(x, info["width"] - 1))(cc)).astype(np.int32)
            mask[rr, cc, i] = 1

        class_id = np.array([self.class_names.index("road_sign")] * mask.shape[-1])
        mask = mask.astype(np.bool)

        renamed_coco_classes = self.normalize_class_names(info)

        return np.concatenate([mask, info["coco_mask"]], axis=2).astype(np.bool), \
               np.concatenate([class_id.astype(np.int32), renamed_coco_classes]).astype(np.int32)

    def normalize_class_names(self, info):
        if len(info['coco_class_ids']) > 0:
            renamed_coco_classes = np.vectorize(
                lambda x:
                self.class_names.index("stop sign")
                if x == COCO_CLASSES.index("stop sign") + 1
                else self.class_names.index(COCO_CLASSES[x]))(info['coco_class_ids'])
        else:
            renamed_coco_classes = []
        return renamed_coco_classes

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]


# todo improve post detection
def detect_post(detection):
    detection["rois"] = np.append(detection["rois"], [[3000, 1850, 3700, 1850 + 20]], axis=0)
    detection["class_ids"] = np.append(detection["class_ids"], 9)
    detection["scores"] = np.append(detection["scores"], 0.9)
    shape = detection["masks"].shape
    mask = np.zeros((shape[0], shape[1], 1), dtype=np.uint8)
    rr, cc = skimage.draw.rectangle((3000, 1835), (3000 + 700, 1835 + 35))
    mask[rr, cc, 0] = 1
    detection["masks"] = np.append(detection["masks"], mask, axis=2)
    return detection


class InferenceConfig(RoadSignsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def detect(image):
    COCO_CLASSES[9] = "road_sign"
    model_path = "mask_rcnn_signs.h5"
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=MODEL_DIR)
    model.load_weights(model_path, by_name=True)
    detection_sign = model.detect([image], verbose=1)[0]
    return detect_post(detection_sign)
