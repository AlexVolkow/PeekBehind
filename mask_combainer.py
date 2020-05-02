import numpy as np

from poisson import poisson_edit


def copy_pasty_combiner(src, dst, masks, offsets):
    height, width, _ = src.shape
    result_image = np.zeros((height, width, 3), np.int8)
    for x in range(height):
        for y in range(width):
            if np.sum(src[x][y]) > 0:
                replace = False
                for i in range(len(masks)):
                    if masks[i][x][y] > 0:
                        x_offset, y_offset = offsets[i]
                        if np.sum(src[x + x_offset][y + y_offset]) > 0:
                            result_image[x][y] = src[x + x_offset][y + y_offset]
                            replace = True
                            break
                if not replace:
                    result_image[x][y] = dst[x][y]
            else:
                result_image[x][y] = dst[x][y]
    return result_image


def copy_with_offset(img, offset):
    x_offset, y_offset = offset
    if x_offset == 0 and y_offset == 0:
        return img.copy()

    height, width, _ = img.shape
    result_image = np.zeros((height, width, 3), np.uint8)
    for x in range(height):
        for y in range(width):
            if x + x_offset < height and y + y_offset < width:
                result_image[x][y] = img[x + x_offset][y + y_offset]
    return result_image


def poisson_combiner(src, dst, masks, offsets):
    height, width, _ = src.shape
    result_image = np.zeros((height, width, 3), np.int8)
    for i in range(len(masks)):
        mask = masks[i]
        img = copy_with_offset(src, offsets[i])
        blending_result = poisson_edit(img, dst.copy(), mask, (0, 0))
        for x in range(height):
            for y in range(width):
                if masks[i][x][y] > 0:
                    result_image[x][y] = blending_result[x][y]
    for x in range(height):
        for y in range(width):
            if np.sum(result_image[x][y]) == 0:
                result_image[x][y] = dst[x][y]
    return result_image
