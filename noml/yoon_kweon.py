from skimage import color
from typing import Tuple
from numba import njit
import numpy as np
from tqdm import tqdm


@njit
def delta_c(p: np.ndarray, q: np.ndarray) -> float:
    """
    Computes the color distance between two pixels as the Euclidean distance between the two colors in the L*a*b* color space
    :param p: first pixel in LAB
    :param q: second pixel in LAB
    :return: color distance
    """
    # compute the Euclidean distance
    return np.sqrt(np.sum((p - q) ** 2))


@njit
def delta_g(p: Tuple[int, int], q: Tuple[int, int]) -> float:
    """
    Computes the Euclidean distance between two pixels positions
    :param p: first pixel position
    :param q: second pixel position
    :return: distance
    """
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


@njit
def w(i: np.ndarray, p: Tuple[int, int], q: Tuple[int, int], gamma_c: float = 5, gamma_p: float = 17.5) -> float:
    """
    Computes the weight between two pixels as defined by Yoon and Kweon (2006) eq.6
    :param i: image in LAB
    :param p: first pixel position
    :param q: second pixel position
    :param gamma_c: color variance, defaults to 10
    :param gamma_p: position variance, defaults to 2
    :return: weight
    """
    return np.exp(-delta_c(i[p], i[q]) ** 2 / gamma_c ** 2 - delta_g(p, q) ** 2 / gamma_p ** 2)


@njit
def e(p: np.ndarray, q: np.ndarray, t: float = 40.0) -> float:
    """
    Computes pixel-based raw matching cost by using the colors of p and q, as defined by Yoon and Kweon (2006) eq.8 (truncated SAD)
    :param p: first pixel
    :param q: second pixel
    :param t: truncation parameter
    :return: matching cost
    """
    return min(np.sum(np.abs(p - q)), t)


@njit
def dissimilarity(i_ref: np.ndarray, i_tgt: np.ndarray, i_ref_lab: np.ndarray, i_tgt_lab: np.ndarray,
                  p: Tuple[int, int], d: int, win_size: int = 35) -> float:
    """
    Computes the dissimilarity between two pixels as defined by Yoon and Kweon (2006) eq.7
    :param i_ref: reference image
    :param i_tgt: target image
    :param i_ref_lab: reference image in LAB
    :param i_tgt_lab: target image in LAB
    :param p: pixel position
    :param d: first estimation of disparity
    :param t: truncation parameter
    :param win_size: size of the window, defaults to 35
    :return: dissimilarity
    """
    total_num = 0
    total_deno = 0
    for dx in range(win_size):
        dx = dx - win_size // 2
        for dy in range(win_size):
            dy = dy - win_size // 2
            if p[0] + dy >= i_ref.shape[0] or p[1] + dx >= i_ref.shape[1] or p[0] + dy < 0 or p[1] + dx < 0:
                continue
            w_ref = w(i_ref_lab, p, (p[0] + dy, p[1] + dx))
            w_tgt = w(i_tgt_lab, (p[0], p[1] - d), (p[0] + dy, p[1] + dx - d))
            e_q = e(i_ref[p[0] + dy, p[1] + dx], i_tgt[p[0] + dy, p[1] + dx - d])
            total_num += w_ref * w_tgt * e_q
            total_deno += w_ref * w_tgt
    return total_num / total_deno


@njit
def dissimilarity_all(i_ref: np.ndarray, i_tgt: np.ndarray, i_ref_lab: np.ndarray, i_tgt_lab: np.ndarray,
                  p: Tuple[int, int], max_d: int, win_size: int = 35) -> np.ndarray:
    """
    Computes the dissimilarity between two pixels as defined by Yoon and Kweon (2006) eq.7 for each possible disparity value
    :param i_ref: reference image
    :param i_tgt: target image
    :param i_ref_lab: reference image in LAB
    :param i_tgt_lab: target image in LAB
    :param p: pixel position
    :param max_d: maximum disparity
    :param win_size: size of the window, defaults to 35
    :return: dissimilarities array
    """
    diss = np.zeros(max_d)
    for d in range(max_d):
        diss[d] = dissimilarity(i_ref, i_tgt, i_ref_lab, i_tgt_lab, p, d, win_size)
    return diss


def yoon_kweon_filter(i_ref: np.ndarray, i_tgt: np.ndarray, i_ref_lab: np.ndarray, i_tgt_lab: np.ndarray,
                      init_dis: np.ndarray) -> np.ndarray:
    """
    Computes the disparity map by using the Yoon and Kweon (2006) algorithm
    :param i_ref: reference image
    :param i_tgt: target image
    :param i_ref_lab: reference image in LAB
    :param i_tgt_lab: target image in LAB
    :param init_dis: initial disparity map
    :return: disparity map
    """
    # get the shape of the image
    h, w = i_ref.shape[:2]

    # compute dissimilarity (E) map
    diss = np.zeros_like(init_dis)
    for x in tqdm(range(w)):
        for y in range(h):
            diss[y, x] = dissimilarity(i_ref, i_tgt, i_ref_lab, i_tgt_lab, (y, x), init_dis[y, x])

    # initialize the disparity map
    dis = np.zeros_like(init_dis)
    window_size = 5
    # for each pixel
    for x in tqdm(range(h)):
        for y in range(w):
            min_diss = np.inf
            for i in range(5):
                for j in range(5):
                    di = i - 2
                    dj = j - 2
                    if x + di < 0 or x + di >= h or y + dj < 0 or y + dj >= w:
                        continue
                    # fetch diss (E) from map
                    d = diss[x + di, y + dj]
                    if d < min_diss and init_dis[x + di, y + dj] != 0:
                        min_diss = d
                        dis[x, y] = init_dis[x + di, y + dj]
    return dis


def yoon_kweon_diss(i_ref: np.ndarray, i_tgt: np.ndarray, i_ref_lab: np.ndarray, i_tgt_lab: np.ndarray, max_disp=63,
                    block_size=35):
    """
    Compute the disparity map between two images using yoon kweon aggregation step.
    :param i_ref: left image
    :param i_tgt: right image
    :param i_ref_lab: left image in LAB
    :param i_tgt_lab: right image in LAB
    :param max_disp: maximum disparity, defaults to 63
    :param block_size: size of the block, defaults to 35
    :return: disparity map
    """
    disp_map = np.zeros_like(i_ref)
    for j in tqdm(range(i_ref_lab.shape[0])):  # for each line
        for i in range(i_ref_lab.shape[1]):  # for each block in the line

            ref_block = i_ref_lab[j - block_size // 2:j + block_size // 2 + 1,
                        i - block_size // 2:i + block_size // 2 + 1]

            if ref_block.shape[:2] != (block_size, block_size):
                continue

            diffs = dissimilarity_all(i_ref, i_tgt, i_ref_lab, i_tgt_lab, (j, i), max_disp, block_size)

            min_diff = np.argmin(diffs)
            # compute the disparity
            disp_map[j:j + block_size, i:i + block_size] = min_diff
    return disp_map

# @njit
def our_diss(i_ref: np.ndarray, i_tgt: np.ndarray, i_ref_lab: np.ndarray, i_tgt_lab: np.ndarray, max_disp=63, block_size=35) -> np.ndarray:
    """
    Compute the disparity map between two images using our aggregation step, similar to yoon kweon but hopefully less ressource intensive.
    :param i_ref: left image
    :param i_tgt: right image
    :param i_ref_lab: left image in LAB
    :param i_tgt_lab: right image in LAB
    :param max_disp: maximum disparity, defaults to 63
    :param block_size: size of the block, defaults to 35
    :return: disparity map
    """
    disp_map = np.zeros((i_ref_lab.shape[0], i_ref_lab.shape[1]))
    for j in tqdm(range(i_ref_lab.shape[0])):  # for each line
        for i in range(i_ref_lab.shape[1]):  # for each block in the line
            ref_block = i_ref[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2:i + block_size // 2 + 1]
            ref_block_lab = i_ref_lab[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2:i + block_size // 2 + 1]
            if ref_block.shape[:2] != (block_size, block_size):
                continue
            disps = np.zeros(max_disp)

            # compute weights for each pixel in the ref block
            weights = np.zeros((block_size, block_size))
            for dj in range(block_size):
                for di in range(block_size):
                    # weight is the color differece in LAB between the pixel and the center of the block,
                    # and the euclidean distance of the coordinates to the center of the block
                    weights[dj, di] = np.exp(-np.linalg.norm(ref_block_lab[dj, di] - ref_block_lab[block_size // 2, block_size // 2]) ** 2)
                    weights[dj, di] *= np.exp(-np.linalg.norm(np.array([dj*1.0, di]) - np.array([block_size // 2, block_size // 2])) ** 2)
            weights /= np.sum(weights)

            for d in range(max_disp):
                tgt_block = i_tgt[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2 - d:i + block_size // 2 + 1 - d]
                tgt_block_lab = i_tgt_lab[j - block_size // 2:j + block_size // 2 + 1, i - block_size // 2 - d:i + block_size // 2 + 1 - d]
                if tgt_block.shape[:2] != (block_size, block_size):
                    continue

                # compute the dissimilarity between the ref block and the tgt block taking into account the weights
                disps[d] = np.sum(weights * np.linalg.norm(ref_block_lab - tgt_block_lab, axis=2))
            disp_map[j:j + block_size, i:i + block_size] = np.argmin(disps)
    return disp_map