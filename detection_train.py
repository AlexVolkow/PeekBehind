from imgaug import augmenters as iaa

from coco_utils import COCO_MODEL_PATH
from detection import *

if __name__ == '__main__':
    config = RoadSignsConfig()
    config.display()

    dataset_train = RoadSignsDataset()
    dataset_train.load_signs(DATASET_PATH, "train")
    dataset_train.prepare()

    dataset_val = RoadSignsDataset()
    dataset_val.load_signs(DATASET_PATH, "val")
    dataset_val.prepare()

    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    model.metrics_tensors = []

    print("Training network heads")
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=15,
                augmentation=augmentation,
                layers='heads')

    model_path = os.path.join(ROOT_DIR, "mask_rcnn_signs.h5")
    model.keras_model.save_weights(model_path)
    print("OK")
