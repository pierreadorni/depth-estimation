#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:14:54 2023

@author: fengzhixuan
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

# zh : 读取左右视图图像
# fr : lecture les images gauche et droite
left_image = cv2.imread("im2.png", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread("im6.png", cv2.IMREAD_GRAYSCALE)

# 特征匹配
# fr : extraction des points d'intérêt
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(left_image, None)
kp2, des2 = orb.detectAndCompute(right_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:100]  # 选择前100个最佳匹配点 / fr: sélection des 100 meilleurs points

# 提取关键点坐标
# fr : extraction des coordonnées des points d'intérêt
points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC估计基础矩阵
# fr : estimation de la matrice fondamentale par RANSAC
F, mask = cv2.findFundamentalMat(points1, points2, cv2.RANSAC, 3.0, 0.99)

# 应用基础矩阵剔除错误匹配点
# fr : application de la matrice fondamentale pour éliminer les points erronés
points1 = points1[mask.ravel() == 1]
points2 = points2[mask.ravel() == 1]

# 定义SGBM参数
# fr : définition des paramètres SGBM
window_size = 8
min_disp = 0
num_disp = 16 * 5

# 预处理图像
# fr : prétraitement des images
left_image = cv2.GaussianBlur(left_image, (5, 5), 0)
right_image = cv2.GaussianBlur(right_image, (5, 5), 0)

# 代价计算
# fr : calcul du coût
stereo = np.zeros((left_image.shape[0], left_image.shape[1], num_disp), dtype=np.float32)
for d in range(num_disp):
    stereo[:, :, d] = cv2.absdiff(left_image, np.roll(right_image, d))

# 代价聚合
# fr : agrégation du coût
disparity = np.zeros_like(left_image, dtype=np.float32)
for y in range(left_image.shape[0]):
    for x in range(left_image.shape[1]):
        min_cost = float('inf')
        best_disparity = 0
        for d in range(num_disp):
            cost = np.sum(stereo[max(0, y - window_size // 2):min(y + window_size // 2 + 1, left_image.shape[0]),
                                max(0, x - window_size // 2):min(x + window_size // 2 + 1, left_image.shape[1]), d])
            if cost < min_cost:
                min_cost = cost
                best_disparity = d
        disparity[y, x] = best_disparity
        
    
# avec ZNSSD        
    
# 代价计算
# stereo = np.zeros((left_image.shape[0], left_image.shape[1], num_disp), dtype=np.float32)
# for d in range(num_disp):
#     diff = (left_image - np.roll(right_image, d)) ** 2
#     mean_diff = np.mean(diff)
#     std_diff = np.std(diff)
#     normalized_diff = (diff - mean_diff) / std_diff
#     stereo[:, :, d] = normalized_diff

# 代价聚合
# disparity = np.zeros_like(left_image, dtype=np.float32)
# for y in range(left_image.shape[0]):
#     for x in range(left_image.shape[1]):
#         min_cost = float('inf')
#         best_disparity = 0
#         for d in range(num_disp):
#             cost = np.sum(stereo[max(0, y - window_size // 2):min(y + window_size // 2 + 1, left_image.shape[0]),
#                                 max(0, x - window_size // 2):min(x + window_size // 2 + 1, left_image.shape[1]), d])
#             if cost < min_cost:
#                 min_cost = cost
#                 best_disparity = d
#         disparity[y, x] = best_disparity


# 调整视差范围和缩放因子
disparity = (disparity - min_disp) / num_disp

# 显示视差图
plt.imshow(disparity, cmap='gray')
plt.colorbar()
plt.show()










