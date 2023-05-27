from numba import njit
import numpy as np


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
def ZSSD_all(ref_block, comp_blocks) -> np.ndarray:
    return np.array([ZSSD(ref_block, comp_block) for comp_block in comp_blocks])


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
            ssd = ZSSD_all(ref_block, comp_blocks)
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
        for i in range(img2.shape[1] - maxdisp):  # for each block in the line
            ref_block = img2[j:j + block_size, i:i + block_size]
            comp_blocks = [img1[j:j + block_size, i + k:i + k + block_size] for k in range(maxdisp)]
            # remove bad shape
            comp_blocks = np.array([comp_block for comp_block in comp_blocks if comp_block.shape == ref_block.shape])
            # compute the SAD between the reference block and the blocks in the right image
            ssd = SAD_all(ref_block, comp_blocks)
            # find the index of the minimum SSD
            min_ssd = np.argmin(ssd)
            # compute the disparity
            disp_map[j:j + block_size, i:i + block_size] = min_ssd
    return disp_map
