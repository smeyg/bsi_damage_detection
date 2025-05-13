######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras (Original)
# Modified by Ismail Uzun for optimization on CPU
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# This version includes optimizations for CPU-based inference.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

import datetime

print('Start Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'viewtech_borescope_1.PNG'
#IMAGE_NAME = 'rvi_ltd_1.PNG'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 3

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory - simplified approach
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        # Configure session for better CPU performance
        config = tf.ConfigProto()
        
        # Adjust thread counts based on your CPU (increase if you have more cores)
        config.intra_op_parallelism_threads = 4  # Use multiple CPU cores
        config.inter_op_parallelism_threads = 4
        
        # Enable CPU optimizations 
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        # Additional optimizations - using proper TF 1.13.2 syntax
        from tensorflow.core.protobuf import rewriter_config_pb2
        config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.constant_folding = rewriter_config_pb2.RewriterConfig.ON
        config.graph_options.rewrite_options.arithmetic_optimization = rewriter_config_pb2.RewriterConfig.ON
        
        sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV
original_image = cv2.imread(PATH_TO_IMAGE)

# Resize image to reduce processing time (adjust dimensions based on your needs)
# Reducing the image size significantly improves inference speed
TARGET_SIZE = 600  # Maximum dimension
height, width = original_image.shape[:2]
scaling_factor = min(TARGET_SIZE / width, TARGET_SIZE / height)
new_width = int(width * scaling_factor)
new_height = int(height * scaling_factor)
image = cv2.resize(original_image, (new_width, new_height))

# Convert to RGB (TensorFlow models expect RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Expand image dimensions to have shape: [1, None, None, 3]
image_expanded = np.expand_dims(image_rgb, axis=0)

# Warm-up run to initialize and optimize computations
print('Performing warm-up inference...')
_ = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Perform the actual detection by running the model with the image as input
print('Inference Start Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})
print('Inference End Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))

# Draw the results of the detection on the original image
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
vis_util.visualize_boxes_and_labels_on_image_array(
    original_image_rgb,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.80)

# Convert back to BGR for OpenCV display
original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', original_image_bgr)

print('End Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()))
# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
sess.close()