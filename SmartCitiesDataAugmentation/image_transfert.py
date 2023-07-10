# This tutorial will go step-by-step through the necessary steps to demonstrate style transfer on images and video.
# Style transfer, where the style of one image is transferred to another as if recreated using the same style,
# is performed using a pre-trained network and running it using the Intel® Distribution of OpenVINO™ toolkit Inference Engine.
# Inference will be executed using the same CPU(s) running this Jupyter* Notebook


#The fast-neural-style-mosaic-onnx model is one of the style transfer models designed to mix the content of an image with the style of another image.
# The model uses the method described in Perceptual Losses for Real-Time Style Transfer and Super-Resolution along with Instance Normalization.
# Original ONNX models are provide.


import os
import cv2
import tensorflow as tf
from pathlib import Path
from openvino.runtime import Core
import matplotlib.pyplot as plt
import time
import subprocess
import numpy as np


classification_model = "fast-neural-style-mosaic-onnx"

# The paths of the source and converted models.
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# download The Models
subprocess.run(["omz_downloader",  "--name", "fast-neural-style-mosaic-onnx", "--output_dir", "./model/model_raw", "--precision", "FP32"])

# convert te model
subprocess.run(["omz_converter",  "--name", "fast-neural-style-mosaic-onnx", "--download_dir",
                "./model/model_raw", "--output_dir", "./model/model_IR", "--precision", "FP32"])

# IR Representation
model_xml = "./model/model_IR/public/fast-neural-style-mosaic-onnx/FP32/fast-neural-style-mosaic-onnx.xml"
model_bin = "./model/model_IR/public/fast-neural-style-mosaic-onnx/FP32/fast-neural-style-mosaic-onnx.bin"

# input image file
input_path = "tubingen.jpg"

# device to use
device = "CPU"

# RGB mean values to add to results
mean_val_r = 0
mean_val_g = 0
mean_val_b = 0

# Create Inference Engine Object
ie = Core()

net = ie.read_model(model=model_xml, weights=model_bin)
net_compiled_model = ie.compile_model(model=net, device_name=device)

#Now that the model is loaded, fetch information about the input and output layers (shape).
net_input_layer = net_compiled_model.input(0)
net_output_layer = net_compiled_model.output(0)

# N, C, H, W = batch_size, number of channels, height, width
N, C, H, W = net_input_layer.shape
print(f"the batch_size is {N}, the number of channels is {C}, the height of the input image is {H} with the width {W}")

input_image = './test_data/tubingen.jpg'

# Load input image
def load_input_image(image_path):
    # global variables to store input height and width
    global input_h, input_w

    # use cv2 to load input image
    cap = cv2.VideoCapture(image_path)

    # store input height and width
    input_w = cap.get(3) #cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_h = cap.get(4) #cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # load the input image
    ret, image = cap.read()
    #del cap
    return image, cap

def resize_input_image(image):
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))

    # Transpose imgage(narray): (227, 227, 3) -> (3, 227, 227)
    resized_image_tr = resized_image.transpose(2, 0, 1)

    # Reshape image to Network Input shape (3, 227, 227) -> (1, 3, 227, 227)
    input_image = np.expand_dims(resized_image_tr, axis=0)

    return  input_image, resized_image


# load image
image, cap = load_input_image(input_image)

# resize the input image
in_frame, just_resized_image = resize_input_image(image)

# display input image
print("Input image:")
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# save start time
inf_start = time.time()

# run inference
# Do Inference
result_infer = net_compiled_model([in_frame])[net_output_layer]

# calculate time from start until now
inf_time = time.time() - inf_start
print("Inference complete, run time: {:.3f} ms".format(inf_time * 1000))


# create function to process inference results
def processResults(res):
    # get output: (1, 3, 224, 224) --> (3, 224, 224)
    result = res[0]

    # Change layout from CxHxW to HxWxC
    result = np.swapaxes(result, 0, 2)  #CxHxW -->HxWxC
    result = np.swapaxes(result, 0, 1)

    # add RGB mean values to
    result = result[::] + (mean_val_r, mean_val_g, mean_val_b)

    # Clip RGB values to [0, 255] range
    result[result < 0] = 0
    result[result > 255] = 255

    # Matplotlib expects normalized image with pixel RGB values in range [0,1].
    result = result / 255
    return result

result = processResults(result_infer)


# create function to process and display inference results
def processAndDisplayResults(res, orig_input_image, verbose=True):
    # display original input image
    plt.figure()
    plt.axis("off")
    im_to_show = cv2.cvtColor(orig_input_image, cv2.COLOR_BGR2RGB)
    plt.imshow(im_to_show)
    plt.show()

    # get output
    result = processResults(res)

    # Show styled image
    #if verbose: print("Results for input image: {}".format(orig_input_path))

    out_height, out_width, _ = orig_input_image.shape
    result = cv2.resize(result, (out_width, out_height))
    plt.figure()
    plt.axis("off")
    plt.imshow(result)
    plt.show()


processAndDisplayResults(result_infer, image)


fig = plt.figure()
# input_path may be set to local file or URL
input_path = "cars_trim.mp4"
print("Loading video [",input_path,"]")
print ("Please wait while the video is converted: about 55 seconds")

from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=15)

def video_inference():
    cap = cv2.VideoCapture(input_path)
    scale = 0.5
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    output_video_name = 'output.mp4'

    with writer.saving(fig, output_video_name, 100):
        while cap.isOpened():
            # read video frame
            ret, image = cap.read()

            # break if no more video frames
            if not ret:
                break

            result_infer = net_compiled_model([in_frame])[net_output_layer]
            result = processResults(result_infer)
            result_resize = cv2.resize(result, (out_width, out_height))
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(result_resize)
            writer.grab_frame()

    print(output_video_name, ": inference completed")
