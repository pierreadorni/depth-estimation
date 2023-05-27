import numpy as np


# mode filter to smooth the disparity map
def mode_filter(disp_map: np.ndarray, filter_size=5):
    """
    Apply a mode filter to the disparity map
    :param disp_map: disparity map
    :param filter_size: size of the filter, defaults to 5
    :return: filtered disparity map
    """
    disp_map_filtered = np.zeros_like(disp_map)
    for j in range(disp_map.shape[0]-filter_size):
        for i in range(disp_map.shape[1]-filter_size):
            # get the values in the filter
            values = disp_map[j:j+filter_size, i:i+filter_size]
            # compute the mode
            mode = np.bincount(values.flatten().astype(int)).argmax()
            # assign the mode to the center of the filter
            disp_map_filtered[j+filter_size//2, i+filter_size//2] = mode
    return disp_map_filtered