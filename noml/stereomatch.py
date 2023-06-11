import numpy as np
from numba import njit
from skimage import io
from skimage.color import rgb2gray
import sys


@njit
def SSD(block1, block2):
    return np.sum((block1 - block2) ** 2)


@njit
def SAD(block1, block2):
    return np.sum(np.abs(block1 - block2))


@njit
def SAD_all(ref_block, comp_blocks):
    return np.array([SAD(ref_block, comp_block) for comp_block in comp_blocks])


@njit
def SSD_all(ref_block, comp_blocks):
    return np.array([SSD(ref_block, comp_block) for comp_block in comp_blocks])


@njit
def SSD(block1, block2):
    return np.sum((block1 - block2) ** 2)


@njit
def SSD_all(ref_block, comp_blocks):
    return np.array([SSD(ref_block, comp_block) for comp_block in comp_blocks])


@njit
def ZSSD(block1, block2):
    return np.sum(((block2 - block2.mean()) - (block1 - block1.mean())) ** 2)


@njit
def ZSSD_all(ref_block: np.ndarray, comp_blocks: np.ndarray):
    return [ZSSD(ref_block, comp_block) for comp_block in comp_blocks]


# implementation of block matching disparity computing
def disp(img1: np.ndarray, img2: np.ndarray, maxdisp: int, block_size: int = 5) -> np.ndarray:
    """
    Compute the disparity map between two images
    :param img1: left image in grayscale
    :param img2: right image in grayscale
    :param maxdisp: maximum disparity
    :param block_size: size of the block, defaults to 5
    :return: disparity map
    """
    disp_map = np.zeros_like(img1)
    for j in range(img1.shape[0]):  # for each line
        for i in range(img1.shape[1]):  # for each block in the line
            ref_block = img1[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2:i + block_size // 2 + 1]
            if ref_block.shape != (block_size, block_size):
                continue
            comp_blocks = [
                img2[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2 - k:i + block_size // 2 + 1 - k]
                for k in range(maxdisp)]
            # remove bad shape
            comp_blocks = np.array([comp_block for comp_block in comp_blocks if comp_block.shape == ref_block.shape])
            # compute the SAD between the reference block and the blocks in the right image
            ssd: np.ndarray = ZSSD_all(ref_block, comp_blocks)
            min_ssd = np.argmin(ssd)
            # compute the disparity
            disp_map[j:j + block_size, i:i + block_size] = min_ssd
    return disp_map


def disp_inv(img1: np.ndarray, img2: np.ndarray, maxdisp: int, block_size: int = 5) -> np.ndarray:
    """
    Compute the disparity map between two images, from right to left
    :param img1: left image in grayscale
    :param img2: right image in grayscale
    :param maxdisp: maximum disparity
    :param block_size: size of the block, defaults to 5
    :return: disparity map
    """
    disp_map = np.zeros_like(img2)
    for j in range(img2.shape[0]):  # for each line
        for i in range(img2.shape[1]):  # for each block in the line
            ref_block = img2[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2:i + block_size // 2 + 1]
            if ref_block.shape != (block_size, block_size):
                continue
            comp_blocks = [
                img1[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2 + k:i + block_size // 2 + 1 + k]
                for k in range(maxdisp)]
            # remove bad shape
            comp_blocks = np.array([comp_block for comp_block in comp_blocks if comp_block.shape == ref_block.shape])
            # compute the SAD between the reference block and the blocks in the right image
            ssd = ZSSD_all(ref_block, comp_blocks)
            # find the index of the minimum SSD
            min_ssd = np.argmin(ssd)
            # compute the disparity
            disp_map[j:j + block_size, i:i + block_size] = min_ssd
    return disp_map


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


def main(left, right, output, maxdisp=64, block_size=5, mode_filter_size=7):
    left_img: np.ndarray = rgb2gray(io.imread(left))
    right_img: np.ndarray = rgb2gray(io.imread(right))

    # compute the disparity map
    disp_map = disp(left_img, right_img, int(maxdisp), int(block_size))

    # compute the inverse disparity map
    disp_map_inv = disp_inv(left_img, right_img, int(maxdisp), int(block_size))

    # merge the two disparity maps
    disp_map_merged = merge_disps(disp_map, disp_map_inv)

    # apply a mode filter
    disp_map_merged_filtered = mode_filter(disp_map_merged, mode_filter_size)

    # save the disparity map
    io.imsave(output, (disp_map_merged_filtered * 4).astype(np.uint8))


if __name__ == '__main__':
    left = sys.argv[1]
    right = sys.argv[2]
    output = sys.argv[3]
    main(left, right, output)

