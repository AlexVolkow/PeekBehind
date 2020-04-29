import math
import os.path

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from numpy import asarray
from numpy import cov
from numpy import iscomplexobj
from numpy import trace
from scipy.linalg import sqrtm
from skimage.io import imread
from skimage.measure import compare_ssim
from skimage.transform import resize
import matplotlib.pyplot as plt


# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


def get_segment_crop(img, mask):
    rect = cv2.boundingRect(mask)
    x, y, w, h = rect
    return img[y:y + h, x:x + w].copy()


# calculate frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(300, 300, 3))


def calculate_metric_fid(images1, images2):
    # convert integer to floating point values
    transformer = lambda img: imread(img).astype('float32')

    images1 = np.array([transformer(img) for img in images1.values()])
    images2 = np.array([transformer(img["origin"]) for img in images2.values()])
    # resize images
    images1 = scale_images(images1, (300, 300, 3))
    images2 = scale_images(images2, (300, 300, 3))
    print('Scaled', images1.shape, images2.shape)
    # pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)
    # calculate fid
    return calculate_fid(model, images1, images2)

from itertools import groupby

def calculate_metric_ssmi(images1, images2):
    ssim = 0
    counter = 0
    for key, value in images1.items():
        img1_path = images1[key]
        img2_path = images2[key]["origin"]
        score_mask = images2[key]["score_mask"]
        pixels, c_ssim = calculate_ssim(img1_path, img2_path, score_mask)
        ssim += c_ssim * pixels
        counter += pixels

    return ssim / counter


def calculate_metric_psnr(images1, images2):
    psnr = 0
    counter = 0
    for key, value in images1.items():
        img1_path = images1[key]
        img2_path = images2[key]["origin"]
        score_mask = images2[key]["score_mask"]

        result_image = cv2.imread(img1_path)
        truth_image = cv2.imread(img2_path)
        score_mask = cv2.imread(score_mask, 0)

        grayA = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(truth_image, cv2.COLOR_BGR2GRAY)

        grayA = cv2.resize(grayA, (300, 300))
        grayB = cv2.resize(grayB, (300, 300))
        grayMask = cv2.resize(score_mask, (300, 300))

        grayA = get_segment_crop(grayA, mask=grayMask)
        grayB = get_segment_crop(grayB, mask=grayMask)

        size = grayB.shape[0] * grayB.shape[1]
        psnr += cv2.PSNR(grayB, grayA) * size
        counter += size
    return psnr / counter


def calculate_metric_l1(images1, images2):
    l1 = 0
    counter = 0
    all = 0
    result = 0
    for key, value in images1.items():
        img1_path = images1[key]
        img2_path = images2[key]["origin"]
        score_mask = images2[key]["score_mask"]

        result_image = cv2.imread(img1_path)
        truth_image = cv2.imread(img2_path)
        score_mask = cv2.imread(score_mask, 0)

        grayA = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(truth_image, cv2.COLOR_BGR2GRAY)

        grayA = cv2.resize(grayA, (300, 300)).astype(np.int32)
        grayB = cv2.resize(grayB, (300, 300)).astype(np.int32)
        grayMask = cv2.resize(score_mask, (300, 300))

        grayA = get_segment_crop(grayA, mask=grayMask)
        grayB = get_segment_crop(grayB, mask=grayMask)

        size = grayB.shape[0] * grayB.shape[1]

        b = grayB - grayA
        l1 += np.linalg.norm(b, ord=1)
        all += np.linalg.norm(grayB, ord=1)
        result += (np.linalg.norm(b, ord=1) / np.linalg.norm(grayB, ord=1)) * size
        counter += size
    return (result / counter) * 100


def calculate_ssim(img1, img2, score_mask):
    result_image_cv2 = cv2.imread(img1)
    truth_image = cv2.imread(img2)
    score_mask = cv2.imread(score_mask, 0)

    grayA = cv2.cvtColor(result_image_cv2, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(truth_image, cv2.COLOR_BGR2GRAY)

    grayA = cv2.resize(grayA, (300, 300))
    grayB = cv2.resize(grayB, (300, 300))
    grayMask = cv2.resize(score_mask, (300, 300))

    grayA = get_segment_crop(grayA, mask=grayMask)
    grayB = get_segment_crop(grayB, mask=grayMask)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    print("Pixels: {}, Score: {}".format(grayB.shape[0] * grayB.shape[1], score))
    return grayB.shape[0] * grayB.shape[1], score


def read_log(log_name):
    with open(log_name) as f:
        content = f.readlines()
    return [x.strip() for x in content]


def already_scores(log, module, result_img, case):
    log_entry = "Module: {}, Image: {}, Case: {}".format(module, result_img, case)
    for l in log:
        if l.startswith(log_entry):
            return True
    return False


if __name__ == '__main__':
    modules = {
     "ours": "target_result.jpg",
        "criminisi": "output.jpg",
        "deep": "output.jpg",
        "generative": "output.jpg",
    }

    modules_files = {
        "ours": {},
        "criminisi": {},
        "deep": {},
        "generative": {},
    }

    truth = {}

    log_writer = open("scores.txt", "a+")

    dataset_size = 33

    base_score_path = "/Users/am.volkov/Documents/scores"
    score_type = "cars"

    for i in range(1, dataset_size):
        case = str(i)

        case_path = os.path.join(base_score_path, score_type, case)
        if not os.path.exists(case_path):
            continue

        origin_image_path = os.path.join(case_path, "source_truth.jpg")
        score_mask_path = os.path.join(case_path, "score_mask.png")

        if not os.path.exists(origin_image_path):
            continue

        if not os.path.exists(score_mask_path):
            print("No mask " + str(i))
            continue

        truth[case] = {"origin": origin_image_path, "score_mask": score_mask_path}

        for module in modules:
            result_image = None
            img = modules[module]
            result_image_path = os.path.join(case_path, module, img)
            if os.path.exists(result_image_path):
                result_image = result_image_path
            if result_image is None:
                continue
            modules_files[module][case] = result_image

    for module in modules_files:
        images = modules_files[module]

        # fid = calculate_metric_fid(images, truth)
        # log_message = "Type: {}, Module: {}, FID: {}".format(score_type, module, fid)
        # print(log_message)
        # log_writer.write(log_message + "\n")

        ssmi = calculate_metric_ssmi(images, truth)
        log_message = "Type: {}, Module: {}, SSMI: {}".format(score_type, module, ssmi)
        print(log_message)
        log_writer.write(log_message + "\n")

        psnr = calculate_metric_psnr(images, truth)
        log_message = "Type: {}, Module: {}, PSNR: {}".format(score_type, module, psnr)
        print(log_message)
        log_writer.write(log_message + "\n")

        l1 = calculate_metric_l1(images, truth)
        log_message = "Type: {}, Module: {}, L1: {} %".format(score_type, module, l1)
        print(log_message)
        log_writer.write(log_message + "\n")

        log_writer.flush()
