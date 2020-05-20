import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from peek_behind import detect_features, match_keypoints_nndr
from poisson import poisson_edit


def get_depth_parts(image, depth, k):
    train = depth.reshape((-1, 1))
    train = np.float32(train)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    attempts = 100
    ret, label, center = cv2.kmeans(train, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((depth.shape))

    masks = []
    for i in range(0, k):
        c = center[i]
        part = np.zeros(image.shape, np.uint8)
        for x in range(0, result_image.shape[0]):
            for y in range(0, result_image.shape[1]):
                if result_image[x][y] == c:
                    part[x][y] = image[x][y]
        masks.append(part.copy())
    return masks


def get_depth_specific_keypoints(imageA, kpsA, kpsB, matches):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    ptsA_l = []
    ptsB_l = []
    if len(matches) > 4:
        for i in range(0, len(matches)):
            m = matches[i]
            ptsA = kpsA[m.queryIdx].astype(np.float32)
            ptsB = kpsB[m.trainIdx].astype(np.float32)

            xA = int(ptsA[1])
            yA = int(ptsA[0])

            if np.sum(imageA[xA][yA]) > 0:
                ptsA_l.append(ptsA)
                ptsB_l.append(ptsB)
    return np.array(ptsA_l), np.array(ptsB_l)


def get_optimal_clusters_count(depth_map):
    max_k = 1
    max_score = -1
    for k in range(2, 10):
        train = depth_map.reshape(len(depth_map), -1)
        kmeans_model = KMeans(n_clusters=k).fit(train)
        labels = kmeans_model.labels_
        score = metrics.cluster.calinski_harabaz_score(train, labels)
        if score > max_score:
            max_score = score
            max_k = k
        print("K: {}, Score: {}".format(k, score))
    return max_k


if __name__ == '__main__':
    source = imageio.imread("samples/parallax/source.jpg")
    dest = imageio.imread("samples/parallax/dest.jpg")
    dest_depth = cv2.imread("samples/parallax/depth.png", 0)

    k = get_optimal_clusters_count(dest_depth)
    print("K: {}".format(k))

    dest_parts = get_depth_parts(dest, dest_depth, k)

    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    dest_gray = cv2.cvtColor(dest, cv2.COLOR_RGB2GRAY)

    mask = cv2.imread("samples/parallax/mask.png", 0)

    feature_extractor = "sift"
    kps_a, features_a = detect_features(dest_gray, method=feature_extractor)
    kps_b, features_b = detect_features(source_gray, method=feature_extractor)
    matches = match_keypoints_nndr(features_a, features_b, ratio=0.8, method=feature_extractor)

    layer_specific_keypoints_source = []
    layer_specific_keypoints_dest = []
    for i in range(0, k):
        d, s = get_depth_specific_keypoints(dest_parts[i], kps_a, kps_b, matches)
        layer_specific_keypoints_source.append(s)
        layer_specific_keypoints_dest.append(d)

    homography = []
    for i in range(0, k):
        if len(layer_specific_keypoints_dest[i]) > 4:
            (H, status) = cv2.findHomography(layer_specific_keypoints_dest[i], layer_specific_keypoints_source[i],
                                             cv2.RANSAC, 10)
            homography.append(H)
        else:
            homography.append(np.zeros((3, 3)))

    sigma = 2
    for i in range(0, k):
        if np.sum(homography[i]) == 0:
            H = np.identity(3)
            for j in range(0, k):
                if np.sum(homography[j]) > 0:
                    w = np.exp(-((j - i) ** 2) / sigma)
                    H_i = ((j + 1) / (i + 1)) * (homography[j] - np.identity(3)) + np.identity(3)
                    H += w * (H_i - np.identity(3))
            homography[i] = H

    mask = cv2.imread("samples/parallax/mask.png", 0)
    w, h, _ = source.shape
    mask = cv2.resize(mask, (h, w))

    result = source.copy()
    for i in range(0, k):
        H = homography[i]
        side_image = dest_parts[i]
        height, width, _ = side_image.shape
        side_image_transformed = cv2.warpPerspective(side_image, H, (width, height))
        plt.imshow(side_image_transformed)
        plt.show()

        for x in range(0, height):
            for y in range(0, width):
                if np.sum(side_image_transformed[x][y]) > 100:
                    result[x][y] = side_image_transformed[x][y]

    alpha = 0.7
    beta = (1.0 - alpha)
    outImage = cv2.addWeighted(result, alpha, source, beta, 0.0)
    plt.imshow(outImage)
    plt.show()

    blending_result = poisson_edit(result, source, mask, (0, 0))
    plt.imsave("bl_result.jpg", blending_result.astype(np.uint8))
    plt.imshow(blending_result)
    plt.show()
