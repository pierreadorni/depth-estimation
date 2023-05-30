import click
import numpy as np
from skimage import io
from skimage.color import rgb2gray

from block_matching import disp, disp_inv
from utils import mode_filter_occl, merge_disps


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
    disp_map_inv = disp_inv(right_img, left_img, int(maxdisp), int(block_size))

    # merge the two disparity maps
    disp_map_merged = merge_disps(disp_map, disp_map_inv)

    # apply a mode filter to the merged disparity map
    disp_map_merged_filtered = mode_filter_occl(disp_map_merged, int(block_size))

    # save the disparity map
    io.imsave('disp_map.png', disp_map_merged_filtered)


if __name__ == '__main__':
    main()

