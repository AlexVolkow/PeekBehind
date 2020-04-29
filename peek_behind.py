import cv2
import numpy as np
import skimage.draw
from skimage.measure import find_contours
import matplotlib.pyplot as plt

from poisson import poisson_edit
from pyramid_blending import pyramid_blending

cv2.ocl.setUseOpenCL(False)


def detect_features(image, method=None, mask=None):
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    else:
        raise Exception("Unknown method: " + str(method))

    (kps, features) = descriptor.detectAndCompute(image, mask)

    return kps, features


def create_matcher(method, crossCheck):
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def match_keypoints_bf(featuresA, featuresB, method):
    bf = create_matcher(method, crossCheck=True)

    best_matches = bf.match(featuresA, featuresB)

    rawMatches = sorted(best_matches, key=lambda x: x.distance)[:100]
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def match_keypoints_nndr(featuresA, featuresB, ratio, method):
    bf = create_matcher(method, crossCheck=False)

    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (nddr):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def match_keypoints_nn(featuresA, featuresB, method):
    bf = create_matcher(method, crossCheck=False)

    rawMatches = bf.knnMatch(featuresA, featuresB, 1)
    print("Raw matches (nn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        matches.append(m[0])
    return matches


def compute_homography(kpsA, kpsB, matches, reprojThresh, method=None):
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        if method == "prosac":
            cv2Method = cv2.RHO
        elif method == "lms":
            cv2Method = cv2.LMEDS
        else:
            cv2Method = cv2.RANSAC

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2Method, reprojThresh)

        return matches, H, status
    else:
        return None


def expand_mask(mask, pixels):
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        verts = np.fliplr(verts) - 1
        rr, cc = skimage.draw.polygon(verts[:, 1], verts[:, 0])
        for j in range(len(rr)):
            x = rr[j]
            y = cc[j]
            mask[x][y] = 1
            for k in range(1, pixels):
                if x - k > 0:
                    mask[x - k][y] = 1
                if x + k < mask.shape[0] and y + k < mask.shape[1]:
                    mask[x + k][y + k] = 1
                if x + k < mask.shape[0]:
                    mask[x + k][y] = 1
                if y + k < mask.shape[1]:
                    mask[x][y + k] = 1
                if x - k > 0 and y - k > 0:
                    mask[x - k][y - k] = 1
                if y - k > 0:
                    mask[x][y - k] = 1
    return mask


def peek_behind(main_image, mask, side_image, features, verbose=False):
    feature_extractor = features['feature_extractor']
    feature_matching = features['feature_matching']
    homo_method = features.get('homography', "ransac")
    ransac_thresh = int(features.get('ransac_thresh', "4"))
    blending_method = features.get("blending_method", "poisson")

    main_image_gray = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)
    side_image_gray = cv2.cvtColor(side_image, cv2.COLOR_RGB2GRAY)

    kps_a, features_a = detect_features(side_image_gray, method=feature_extractor)
    kps_b, features_b = detect_features(main_image_gray, method=feature_extractor)

    if verbose:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
        ax1.imshow(cv2.drawKeypoints(side_image_gray, kps_a, None, color=(0, 255, 0)))
        ax1.set_xlabel("(a)", fontsize=14)
        ax2.imshow(cv2.drawKeypoints(main_image_gray, kps_b, None, color=(0, 255, 0)))
        ax2.set_xlabel("(b)", fontsize=14)
        plt.savefig("verbose/kps.jpg")

    if feature_matching == 'threshold':
        matches = match_keypoints_bf(features_a, features_b, method=feature_extractor)
    elif feature_matching == 'nndr8' or feature_matching == "knn":
        matches = match_keypoints_nndr(features_a, features_b, ratio=0.8, method=feature_extractor)
    elif feature_matching == 'nndr7':
        matches = match_keypoints_nndr(features_a, features_b, ratio=0.7, method=feature_extractor)
    elif feature_matching == 'nn':
        matches = match_keypoints_nn(features_a, features_b, method=feature_extractor)

    if verbose:
        sk = cv2.drawKeypoints(side_image_gray, np.random.choice(kps_a, 30000), None, color=(255, 255, 0))
        qk = cv2.drawKeypoints(main_image_gray, np.random.choice(kps_b, 30000), None, color=(255, 255, 0))

        matched_keypoints = cv2.drawMatches(sk, kps_a, qk, kps_b, matches,
                                            outImg=None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(matched_keypoints)
        plt.imsave("verbose/matched.jpg", matched_keypoints.astype(np.uint8))

    M = compute_homography(kps_a, kps_b, matches, reprojThresh=ransac_thresh, method=homo_method)
    if M is None:
        print("Can't compute homogprahy")
        return None
    (matches, H, status) = M

    height, width, _ = side_image.shape
    if H is None:
        print("Can't compute homogprahy")
        return None
    side_image_transformed = cv2.warpPerspective(side_image, H, (width, height))
    if verbose:
        plt.imsave("verbose/transformed.jpg", side_image_transformed.astype(np.uint8))

    if blending_method == "poisson":
        blending_result = poisson_edit(side_image_transformed, main_image, mask, (0, 0))
    elif blending_method == "pyramid":
        blending_result = pyramid_blending(main_image, side_image_transformed, mask)
    else:
        print("Unsupported blending_method " + blending_method)
        blending_result = side_image_transformed
    return blending_result
