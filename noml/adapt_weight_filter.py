import numpy as np
from tqdm import tqdm


def adapt_weight(disp_map: np.ndarray, lab_image: np.ndarray, filter_size=11):
    """
    Apply an adaptative weight filter to the disparity map
    :param disp_map: disparity map
    :param lab_image: lab image
    :param filter_size: size of the filter, defaults to 5
    :return: filtered disparity map
    """
    disp_map_filtered = np.zeros_like(disp_map)
    for j in tqdm(range(disp_map.shape[0]-filter_size)):
        for i in range(disp_map.shape[1]-filter_size):
            # get the values in the filter
            values = disp_map[j:j+filter_size, i:i+filter_size]
            # get the color differences in the filter (compute the Euclidean distance in the lab space)
            color_diff = np.sqrt(np.sum((lab_image[j:j+filter_size, i:i+filter_size] - lab_image[j+filter_size//2, i+filter_size//2])**2, axis=2))
            # get the position differences in the filter (compute the Euclidean distance in the image space)
            pos_diff = np.sqrt(np.sum((np.indices((filter_size, filter_size)).transpose(1, 2, 0) - np.array([filter_size//2, filter_size//2]))**2, axis=2))
            # compute the weights
            weights = np.exp(-color_diff/np.max(color_diff)) * np.exp(-pos_diff/np.max(pos_diff))
            # count the number of occurrences of each value, multiply by the weights and take the mode
            mode = np.bincount(values.flatten().astype(int), weights=weights.flatten()).argmax()
            # assign the mode to the center of the filter
            disp_map_filtered[j+filter_size//2, i+filter_size//2] = mode
            # compute the weighted average
            # weighted_average = np.sum(values * weights) / np.sum(weights)
            # find the window value with the closest disparity to the weighted average
            # closest_value = values[np.unravel_index(np.argmin(np.abs(values - weighted_average)), values.shape)]
            # assign the closest value to the center of the filter
            # disp_map_filtered[j+filter_size//2, i+filter_size//2] = closest_value
    return disp_map_filtered