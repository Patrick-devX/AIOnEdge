import shutil
import sys
import wget
import os
from pathlib import Path
import subprocess
import json
import PIL
import urllib.request
import cv2
import matplotlib.pyplot as plt


from openvino.runtime import Core
import tensorflow as tf
import numpy as np
import time

from IPython.display import Markdown, display
import nibabel

saved_model_dir = "./model/2d_unet_decathlon"
IR_batch_size = 1
pecision = "FP32"

open_vino_model_dir = os.path.join("output", pecision)
open_vino_model_name = "2d_unet_decathlon"

data_path = "./Task01_BrainTumor"
crop_dim = 128
# The following are not used for inference but are needed becacause we are using the
#same "training" function for getting inference images.
batch_size = 12
seed=16
train_test_split = 0.85
