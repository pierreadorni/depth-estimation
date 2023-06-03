import numpy as np


def merge_disps(left: np.ndarray, right: np.ndarray):
    # for each pixel on the left image, go to the right image and back to the left image and check if we are still on
    # the same pixel (approximately) if not, we are on an occluded pixel
    result = left.copy()
    for j in range(left.shape[0]):
        for i in range(left.shape[1]):
            # get the disparity of the pixel
            disp = int(left[j, i])
            # get the pixel on the right image
            try:
                pixel_right = right[j, i - disp]
                # if the pixel on the right image is not approx. the same as the pixel on the left image, we are on
                # an occluded pixel
                if disp < pixel_right - 10 or disp > pixel_right + 10:
                    result[j, i] = 0
            except IndexError as e:
                result[j, i] = 0

    return result


def mode_filter_occl(disp_map: np.ndarray, filter_size=5):
    """
    Apply a mode filter to the disparity map, except for the occluded pixels
    :param disp_map: disparity map
    :param filter_size: size of the filter, defaults to 5
    :return: filtered disparity map
    """
    disp_map_filtered = np.zeros_like(disp_map)
    for j in range(disp_map.shape[0] - filter_size):
        for i in range(disp_map.shape[1] - filter_size):
            # get the values in the filter
            values = disp_map[j:j + filter_size, i:i + filter_size]
            # compute the mode except 0
            mode = np.bincount(values.flatten().astype(int)).argmax()
            if mode == 0:
                try:
                    mode = np.bincount(values.flatten().astype(int))[1:].argmax() + 1
                except ValueError:
                    mode = 0
            # assign the mode to the center of the filter
            disp_map_filtered[j + filter_size // 2, i + filter_size // 2] = mode
    return disp_map_filtered


def mode_filter(disp_map: np.ndarray, filter_size=5):
    """
    Apply a mode filter to the disparity map
    :param disp_map: disparity map
    :param filter_size: size of the filter, defaults to 5
    :return: filtered disparity map
    """
    disp_map_filtered = np.zeros_like(disp_map)
    for j in range(disp_map.shape[0] - filter_size):
        for i in range(disp_map.shape[1] - filter_size):
            # get the values in the filter
            values = disp_map[j:j + filter_size, i:i + filter_size]
            # compute the mode
            mode = np.bincount(values.flatten().astype(int)).argmax()
            # assign the mode to the center of the filter
            disp_map_filtered[j + filter_size // 2, i + filter_size // 2] = mode
    return disp_map_filtered
