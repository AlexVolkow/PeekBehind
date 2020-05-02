import cv2
import numpy as np


def adjust_shift(src, dst, mask, interval=(-5, 5)):
    min_shift = (0, 0)
    min_error = contour_error(src, dst, mask, min_shift)
    for x in range(interval[0], interval[1]):
        for y in range(interval[0], interval[1]):
            error = contour_error(src, dst, mask, (x, y))
            print("Shift: ({}, {}), Error: {}".format(x, y, error))
            if min_error > error:
                min_error = error
                min_shift = (x, y)
    print("Best shift: ({}, {}), Error: {}".format(min_shift[0], min_shift[1], min_error))
    return [mask], [min_shift]


def contour_error(src, dst, mask, offset, width=4):
    x_offset, y_offset = offset

    contour = get_mask_contour(mask)
    bx, by, bw, bh = cv2.boundingRect(contour)
    bx -= width
    by -= width
    bw += 2 * width
    bh += 2 * width

    points = []
    radius = width // 2
    for x in range(by, by + bh, radius):
        points.append((x, bx + radius))
        points.append((x, bx + bw - radius))
    for y in range(bx, bx + bw, radius):
        points.append((by + radius, y))
        points.append((by + bh - radius, y))

    error = 0
    count = 0
    for x, y in points:
        error += dot_error(src, dst, (x, y), (x + x_offset, y + y_offset), radius=width)
        count += 1
    return error / count


def get_mask_contour(mask):
    border = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    _, contours, hierarchy = cv2.findContours(border, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    return contours[0]


def dot_error(src, dst, src_dot, dst_dot, radius):
    src_area = safe_crop(src, src_dot, radius)
    dst_area = safe_crop(dst, dst_dot, radius)

    diff = dst_area - src_area
    error = 0
    for h in range(0, 3):
        error += np.linalg.norm(diff[:, :, h], ord=2)
    return error / 3


def safe_crop(img, dot, radius):
    return img[max(0, dot[0] - radius):min(dot[0] + radius, img.shape[0]),
           max(0, dot[1] - radius):min(dot[1] + radius, img.shape[1]), :]


