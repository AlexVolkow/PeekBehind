import cv2
import numpy as np


def gaussian_pyramid(img, num_levels):
    lower = img.copy()
    gaussian_pyr = [lower]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gaussian_pyr.append(np.float32(lower))
    return gaussian_pyr


def laplacian_pyramid(gaussian_pyr):
    laplacian_top = gaussian_pyr[-1]
    num_levels = len(gaussian_pyr) - 1

    laplacian_pyr = [laplacian_top]
    for i in range(num_levels, 0, -1):
        size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
        gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
        laplacian = np.subtract(gaussian_pyr[i - 1], gaussian_expanded)
        laplacian_pyr.append(laplacian)
    return laplacian_pyr


def blend(laplacian_A, laplacian_B, mask_pyr):
    LS = []
    for la, lb, mask in zip(laplacian_A, laplacian_B, mask_pyr):
        ls = lb * mask + la * (1.0 - mask)
        LS.append(ls)
    return LS


def reconstruct(laplacian_pyr):
    laplacian_top = laplacian_pyr[0]
    laplacian_lst = [laplacian_top]
    num_levels = len(laplacian_pyr) - 1
    for i in range(num_levels):
        size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
        laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
        laplacian_top = cv2.add(laplacian_pyr[i + 1], laplacian_expanded)
        laplacian_lst.append(laplacian_top)
    return laplacian_lst


def pyramid_blending(image1, image2, mask, num_levels=4):
    mask1 = np.zeros(image1.shape, dtype='float32')
    mask1[:, :, 0] = np.vectorize(lambda x: 1 if x > 0 else 0)(mask)
    mask1[:, :, 1] = np.vectorize(lambda x: 1 if x > 0 else 0)(mask)
    mask1[:, :, 2] = np.vectorize(lambda x: 1 if x > 0 else 0)(mask)

    gaussian_pyr_1 = gaussian_pyramid(image1, num_levels)
    laplacian_pyr_1 = laplacian_pyramid(gaussian_pyr_1)
    gaussian_pyr_2 = gaussian_pyramid(image2, num_levels)
    laplacian_pyr_2 = laplacian_pyramid(gaussian_pyr_2)
    mask_pyr_final = gaussian_pyramid(mask1, num_levels)
    mask_pyr_final.reverse()
    add_laplace = blend(laplacian_pyr_1, laplacian_pyr_2, mask_pyr_final)
    final = reconstruct(add_laplace)
    return final[num_levels]
