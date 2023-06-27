import unittest
import os
import cv2
import tensorflow as tf
from pathlib import Path
from openvino.runtime import Core
import matplotlib.pyplot as plt
import time
import subprocess
import numpy as np

############ ndustrial Worker Safety - Object Detection - Task 1: Instruction ###################

classification_model = "mobilenet-ssd"

# The paths of the source and converted models.
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# download The Models
subprocess.run(["omz_downloader",  "--name", "mobilenet-ssd", "--output_dir", "./model/model_raw", "--precision", "FP32"])

# convert te model
subprocess.run(["omz_converter",  "--name", "mobilenet-ssd", "--download_dir",
                "./model/model_raw", "--output_dir", "./model/model_IR", "--precision", "FP32"])

# IR Representation
model_xml = "./model/model_IR/public/mobilenet-ssd/FP32/mobilenet-ssd.xml"
model_bin = "./model/model_IR/public/mobilenet-ssd/FP32/mobilenet-ssd.bin"

# numbering labels
with open('./model/labels/labels.txt') as infile, open('./model/labels/enumerate_labels.txt','w') as outfile:
    for idx, line in enumerate(infile):
        outfile.write(f'{idx} {line}')

# device
device = "CPU"

# labels
labels = './model/labels/enumerate_labels.txt'

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

# Load input image
def load_input_image(image_path):
    # global variables to store input height and width
    global input_h, input_w

    # use cv2 to load input image
    cap = cv2.VideoCapture(image_path)

    # store input height and width
    input_w = cap.get(3)
    input_h = cap.get(4)

    # load the input image
    ret, image = cap.read()
    del cap
    return image

input_image_ = load_input_image("./model/test_image/004545.jpg")
video_path = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4"

def load_input_video(video_path):
    # input_path may be set to local file or URL
    input_path = video_path

    print("Loading video [", input_path, "]")

    # use OpenCV to load the input image
    cap = cv2.VideoCapture(input_path)
    scale = 0.5
    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    global input_h, input_w

    input_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = 'output.webm'
    frame = 20.0
    vw = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'vp80'), frame, (out_width, out_height), True)

    while cap.isOpened():
        # read video frame
        ret, image = cap.read()

        # break if no more video frames
        if not ret:
            break

        result_infer = net_compiled_model([input_image])[net_output_layer]
        image = cv2.resize(image, (out_width, out_height))
        vw.write(image)

    cap.release()
    vw.release()
    print("Done.")
    return video_name

#video_name = load_input_video(video_path)
#from IPython.display import Video
#Video(video_name)

def resize_input_image(image):
    # Resize the image to meet network expected input sizes.
    resized_image = cv2.resize(image, (W, H))

    # Transpose imgage(narray): (227, 227, 3) -> (3, 227, 227)
    resized_image_tr = resized_image.transpose(2, 0, 1)

    # Reshape image to Network Input shape (3, 227, 227) -> (1, 3, 227, 227)
    input_image = np.expand_dims(resized_image_tr, axis=0)

    return  input_image, resized_image

input_image, resized_image = resize_input_image(input_image_)

# Display Image
print('Input image resized')
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

# Do Inference
result_infer = net_compiled_model([input_image])[net_output_layer]
result_index = np.argmax(result_infer)

# minimum probability threshold to detect an object
prob_threshold = 0.5

labels_map = None
# if labels points to a label mapping file, then load the file into labels_map
print(labels)
if os.path.isfile(labels):
    with open(labels, 'r') as f:
        labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    print("Loaded label mapping file [",labels,"]")
else:
    print("No label mapping file has been loaded, only numbers will be used",
          " for detected object labels")

# create function to process inference results

    #obj[1] = Class ID (type of object detected)
    #obj[2] = Probability of detected object
    #obj[3] = Lower x coordinate of detected object
    #obj[4] = Lower y coordinate of detected object
    #obj[5] = Upper x coordinate of detected object
    #obj[6] = Upper y coordinate of detected object

def processResults(result):
    # get output results
    res = result
    colors = list()

    # loop through all possible results
    for obj in res[0][0]:
        # If probability is more than specified threshold, draw and label box
        if obj[2] > prob_threshold:
            # get coordinates of box containing detected object
            xmin = int(obj[3] * input_w)
            ymin = int(obj[4] * input_h)
            xmax = int(obj[5] * input_w)
            ymax = int(obj[6] * input_h)

            # get type of object detected
            class_id = int(obj[1])

            # Draw box and label for detected object
            color = (min(class_id * 12.5, 255), 255, 255)
            colors.append(color)
            cv2.rectangle(input_image_, (xmin, ymin), (xmax, ymax), color, 4)
            det_label = labels_map[class_id] if labels_map else str(class_id)
            cv2.putText(input_image_, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
            print(det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %')

    return colors

colors = processResults(result_infer)
print(colors)
# Display Image
print('Input image resized')
plt.axis("off")
plt.imshow(cv2.cvtColor(input_image_, cv2.COLOR_BGR2RGB))
plt.show()