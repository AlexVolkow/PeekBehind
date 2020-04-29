import os.path

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from skimage.measure import compare_ssim
from skimage.transform import resize

# scale an array of images to a new size
from peek_behind import peek_behind


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
    plt.imsave("verbose/grayA.jpg", grayA)
    plt.imsave("verbose/grayB.jpg", grayB)

    (score, diff) = compare_ssim(grayA, grayB, full=True, multichannel=True)
    return score

if __name__ == '__main__':
    params = {
        "kps": ["sift", "surf", "orb", "brisk"],
        "matching": ["nndr7", "nndr8", "nn"],
        "homography": ["ransac", "lms", "prosac"],
        "ransac_thresh": [i for i in range(1, 11)],
        "blend": ["poisson"]
    }

    dataset_size = 33

    log_file = open("hyper.txt", "+a")
    with open("hyper.txt") as f:
        full_log = f.readlines()

    base_score_path = "/Users/am.volkov/Documents/scores"
    score_type = "signs"

    for i in range(1, dataset_size):
        if i < 9 or i == 11 or i == 18:
            continue
        case = str(i)

        case_path = os.path.join(base_score_path, score_type, case)
        result_path = os.path.join(case_path, "hyper")
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        origin_image_path = os.path.join(case_path, "source_truth.jpg")
        source_image_path = os.path.join(case_path, "source.jpg")
        mask_image_path = os.path.join(case_path, "mask.png")
        if not os.path.exists(mask_image_path):
            mask_image_path = os.path.join(case_path, "mask.jpg")
        front_image_path = os.path.join(case_path, "dest.jpg")

        for kps_method in params["kps"]:
            for matching_method in params["matching"]:
                for homography in params["homography"]:
                    for blend in params["blend"]:
                        for ransac_thresh in params["ransac_thresh"]:
                            filename = "{}_{}_{}_{}_{}.jpg".format(kps_method, matching_method, homography, ransac_thresh, blend)
                            hpath = os.path.join(result_path, filename)

                            print("Process " + filename)
                            if os.path.exists(hpath) or (homography == "lms" and ransac_thresh != "1"):
                                print("Skip")
                                continue

                            calculated = False
                            for log_line in full_log:
                                s = "{}/{}".format(case, filename)
                                if log_line.startswith(s):
                                    calculated = True
                                    break
                            if calculated:
                                print("Already calculated")
                                continue

                            query_image = imageio.imread(source_image_path)
                            side_image = imageio.imread(front_image_path)
                            mask_image = cv2.imread(mask_image_path, 0)

                            features = {
                                "feature_extractor": kps_method,
                                "feature_matching":  matching_method,
                                "homography": homography,
                                "blending_method": blend,
                                "ransac_thresh": ransac_thresh
                            }

                            peek_behind_result = peek_behind(query_image, mask_image, side_image, features, verbose=True)
                            filename = "{}_{}_{}_{}_{}.jpg".format(kps_method, matching_method, homography, ransac_thresh, blend)

                            if peek_behind_result is None:
                                print("Can't compute " + filename)
                                continue

                            plt.imsave(hpath, peek_behind_result.astype(np.uint8))
                            if blend != "gp-gan":
                                ssim = calculate_ssim(hpath, origin_image_path, mask_image_path)

                                print("SSIM: " + str(ssim))
                                log_file.write("{}/{} {}\n".format(case, filename, ssim))
                                log_file.flush()

                                os.remove(hpath)
    log_file.close()
