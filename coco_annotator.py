import json

from coco_utils import save_coco_result, detect_coco
from mrcnn.utils import *

ROOT_DIR = os.path.abspath("/")
DATASET_PATH = "./sign_dataset"
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Classes that we save from coco
# 1 - Person
# 2 - Bicycle
# 3 - Car
# 4 - Motorcycle
# 5 - Bus
# 8 - Truck
# 10 - Traffic light
# 13 - Stop sign
COCO_CLASS_IDS = [1, 2, 3, 4, 6, 8, 10, 13]


def take_classes(result, class_ids):
    new_r = {'rois': [],
             'masks': [],
             'scores': [],
             'class_ids': []}
    deleted_rows = []
    for i in range(len(result['class_ids'])):
        class_id = result['class_ids'][i]
        if class_id in class_ids:
            new_r['rois'].append(result['rois'][i])
            new_r['class_ids'].append(class_id)
            new_r['scores'].append(result['scores'][i])
        else:
            deleted_rows.append(i)
    new_r['masks'] = np.delete(result['masks'], deleted_rows, axis=2)
    new_r['rois'] = np.array(new_r['rois'])
    new_r['scores'] = np.array(new_r['scores'])
    new_r['class_ids'] = np.array(new_r['class_ids'])
    return new_r


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Annotate your dataset coco classes.')
    parser.add_argument("subset",
                        metavar="<subset>")

    args = parser.parse_args()
    subset = args.subset

    assert subset in ["train", "val"]

    dataset_dir = os.path.join(DATASET_PATH, subset)
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())

    annotations = [a for a in annotations if a['regions']]

    totalCount = 0
    for a in annotations:
        totalCount += 1
        coco_result_file = os.path.join(dataset_dir, a['filename'] + "_coco.json")
        image_path = os.path.join(dataset_dir, a['filename'])

        if not os.path.exists(coco_result_file):
            image = skimage.io.imread(image_path)
            coco_detection = detect_coco(image)
            coco_result = take_classes(coco_detection, COCO_CLASS_IDS)
            save_coco_result(coco_result_file, coco_result)
            print("Save coco detection " + coco_result_file)
    print("Coco annotated " + str(totalCount))
