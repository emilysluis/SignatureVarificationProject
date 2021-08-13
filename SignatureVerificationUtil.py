
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

class SignatureVarificationUtil:
    