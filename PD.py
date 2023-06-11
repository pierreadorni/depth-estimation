#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:09:24 2023

@author: fengzhixuan
"""

import numpy as np
import argparse
from skimage import io
import cv2
import matplotlib.pyplot as plt
from numba import njit


@njit
def calculate_disparity_map(left_image, right_image, window_size, max_disparity):
    height, width = left_image.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    cost_matrix = np.zeros((height, width, max_disparity), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            for d in range(max_disparity):
                left_patch = get_patch(left_image, x, y, window_size)
                right_patch = get_patch(right_image, x - d, y, window_size)
                cost = calculate_cost(left_patch, right_patch)
                cost_matrix[y, x, d] = cost
    
    for y in range(height):
        for x in range(width):
            disparity_map[y, x] = np.argmin(cost_matrix[y, x])
    
    return disparity_map

@njit
def get_patch(image, x, y, window_size):
    height, width = image.shape
    half_size = window_size // 2
    patch = np.zeros((window_size, window_size), dtype=image.dtype)
    
    for i in range(window_size):
        for j in range(window_size):
            patch[i, j] = image[max(0, min(y - half_size + i, height - 1)), max(0, min(x - half_size + j, width - 1))]
    
    return patch

@njit
def calculate_cost(left_patch, right_patch):
    return np.sum(np.abs(left_patch - right_patch))

# zh : 读取左右视图图像
# fr : lecture les images gauche et droite
left_image = io.imread("im2.png", 0)
right_image = io.imread("im6.png", 0)

# zh : 转换图像数据类型为灰度图像
# fr : convertit les images en niveaux de gris
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
window_size = 1
max_disparity = 100

disparity_map = calculate_disparity_map(left_gray, right_gray, window_size, max_disparity)

plt.imshow(disparity_map, cmap='gray')
plt.colorbar()
plt.show()