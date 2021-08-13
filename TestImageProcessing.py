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

from SignatureVerificationUtil import *

curr_dir = '013'


def import_reference(dir):
    path = 'ICDAR/Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)/'+dir
    count = 0
    reference_genuine_images = []
    for filename in os.listdir(path):
        if filename.endswith('.PNG'):
            count += 1
            writer_id, image_id = (filename.rsplit('.', 1)[0].split('_'))
            writer_id = int(writer_id)
            image_id = int(image_id)
            image = np.array(io.imread(path + '/' + filename, plugin='pil', as_gray=True))
            reference_genuine_images.append([writer_id, image_id, np.array(image)])
    print('# of images read in:', count)
    return np.array(reference_genuine_images)


def main():
    ref_gen_images = import_reference(curr_dir)


if __name__ == '__main__':
    main()