import click
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.color import rgb2gray

from block_matching import disp, disp_inv
from utils import mode_filter_occl, merge_disps, mode_filter


@click.command()
@click.option('--left', help='Left image', required=True)
@click.option('--right', help='Right image', required=True)
@click.option('--maxdisp', help='Maximum disparity', default=64)
@click.option('--block_size', help='Block size', default=5)
def main(left, right, maxdisp=64, block_size=5):
    left_img: np.ndarray = rgb2gray(io.imread(left))
    right_img: np.ndarray = rgb2gray(io.imread(right))

    # compute the disparity map
    disp_map = disp(left_img, right_img, int(maxdisp), int(block_size))

    # compute the inverse disparity map
    disp_map_inv = disp_inv(left_img, right_img, int(maxdisp), int(block_size))

    # merge the two disparity maps
    disp_map_merged = merge_disps(disp_map, disp_map_inv)

    # apply a mode filter but without the occluded pixels
    disp_map_merged_filtered_no_occl_11 = mode_filter_occl(disp_map_merged, 11)
    disp_map_merged_filtered_no_occl_9 = mode_filter_occl(disp_map_merged, 9)
    disp_map_merged_filtered_no_occl_7 = mode_filter_occl(disp_map_merged, 7)
    disp_map_merged_filtered_no_occl_5 = mode_filter_occl(disp_map_merged, 5)
    disp_map_merged_filtered_no_occl_3 = mode_filter_occl(disp_map_merged, 3)

    # apply a mode filter
    disp_map_merged_filtered_11 = mode_filter(disp_map_merged, 11)
    disp_map_merged_filtered_9 = mode_filter(disp_map_merged, 9)
    disp_map_merged_filtered_7 = mode_filter(disp_map_merged, 7)
    disp_map_merged_filtered_5 = mode_filter(disp_map_merged, 5)
    disp_map_merged_filtered_3 = mode_filter(disp_map_merged, 3)

    # save the disparity maps
    io.imsave('tests/disp_map_no_occl_11.png', disp_map_merged_filtered_no_occl_11.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_no_occl_9.png', disp_map_merged_filtered_no_occl_9.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_no_occl_7.png', disp_map_merged_filtered_no_occl_7.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_no_occl_5.png', disp_map_merged_filtered_no_occl_5.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_no_occl_3.png', disp_map_merged_filtered_no_occl_3.astype(np.uint8) * 4)

    io.imsave('tests/disp_map_11.png', disp_map_merged_filtered_11.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_9.png', disp_map_merged_filtered_9.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_7.png', disp_map_merged_filtered_7.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_5.png', disp_map_merged_filtered_5.astype(np.uint8) * 4)
    io.imsave('tests/disp_map_3.png', disp_map_merged_filtered_3.astype(np.uint8) * 4)

    io.imsave('tests/disp_map.png', disp_map_merged.astype(np.uint8) * 4)


if __name__ == '__main__':
    main()

