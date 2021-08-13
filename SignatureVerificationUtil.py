
# This should be roughly the content of the first code cell
import numpy as np
import random
np.random.seed(1337)
random.seed(1337)

# Plotting support
from matplotlib import pyplot as plt
# from plotnine import
# Standard libraries
import pandas as pd
import sklearn as sk
import tensorflow as tf
import time
import os
from skimage import io
import cv2
from scipy import stats
from skimage.util import compare_images
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity, hausdorff_distance, normalized_root_mse, peak_signal_noise_ratio
plt.rc('image', cmap='gray')

# class SignatureVerificationUtil:

def resize_image(image):
    resized_img = cv2.resize(image, (512, 256))
    return np.array(resized_img)

def resize_images(array):
    temp = []
    for image in array:
        resized = resize_image(image)
        temp.append(resized)
    return np.array(temp)

def binarize_images(array):
    temp= np.copy(array)

    super_threshold_indices = temp > 0.8
    temp[ temp != super_threshold_indices] = 1
    temp[super_threshold_indices] = 0
    return temp

def dilate_images(array):
    temp = []
    kernel = np.ones((5,5),np.uint8)

    for image in array:
        dilation = cv2.dilate(image, kernel, iterations=2)
        temp.append(dilation)
    return np.array(temp)

def get_comparison_metrics(image1, image2):
    mse = mean_squared_error(image1, image2)
    ssim = structural_similarity(image1, image2, data_range=1)
    log_and = np.sum(np.logical_and(image1, image2))
    simple_diff = np.sum(np.abs(image1 - image2))

    checkerboard = compare_images(image1, image2, method='checkerboard')
    blend = compare_images(image1, image2, method='blend')
    checkerboard_mean = np.mean(checkerboard)
    blend_mean = np.mean(blend)

    hd = hausdorff_distance(image1, image2)
    nrmse = normalized_root_mse(image1, image2)

    psnr = peak_signal_noise_ratio(image1, image2)

    return [mse, ssim, log_and, simple_diff, checkerboard_mean, blend_mean, hd, nrmse, psnr]


def create_feature_table(curr_genuine, curr_forged):
    # [mse, ssim, log_and, simple_diff, checkerboard_mean, blend_mean, hd, nrmse, psnr]
    feature_table = []

    # curr_genuine = genuine_processed[genuine_labels == 1]
    # curr_forged = forged_processed[forged_labels == 1]

    for i in range(len(curr_genuine)):
        for j in range(i + 1, len(curr_genuine)):
            row = ['label1_gen' + str(i) + '_gen' + str(j)]
            row.extend(get_comparison_metrics(curr_genuine[i], curr_genuine[j]))
            row.append(True)
            feature_table.append(np.array(row))

        for j in range(len(curr_forged)):
            row = ['label1_gen' + str(i) + '_for' + str(j)]
            row.extend(get_comparison_metrics(curr_genuine[i], curr_forged[j]))
            row.append(False)
            feature_table.append(np.array(row))

    feature_table = np.array(feature_table)
    return feature_table

def test_func():
    print('testttttt')