######## Optimized Multi-Image Object Detection Using Tensorflow-trained Classifier #########
#
# Modified by Ismail Uzun from the original by Evan Juras
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It processes all images in a directory with various optimizations for speed.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import glob
from datetime import datetime
import multiprocessing
import gc

# Enable TensorFlow optimizations
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging output

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

print('Start Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.now()))

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_DIR = 'test_images'  # Directory containing images to process

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image directory
PATH_TO_IMAGE_DIR = os.path.join(CWD_PATH, IMAGE_DIR)

# Create output directory if it doesn't exist
OUTPUT_DIR = os.path.join(CWD_PATH, 'detection_results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Number of classes the object detector can identify
NUM_CLASSES = 3

# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Maximum dimension for resizing images - using power of 2 for efficiency
TARGET_SIZE = 512

# Load the Tensorflow model into memory with optimizations
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        # Configure session for optimal CPU performance
        config = tf.ConfigProto()
        # Set parallelism threads based on available CPU cores
        cpu_count = multiprocessing.cpu_count()
        config.intra_op_parallelism_threads = cpu_count
        config.inter_op_parallelism_threads = max(1, cpu_count // 2)
        
        # Enable advanced CPU optimizations
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        
        # Allow TensorFlow to manage memory efficiently
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        
        sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Warm-up run with multiple passes to initialize and optimize computations
print('Performing warm-up inference...')
dummy_image = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
dummy_expanded = np.expand_dims(dummy_image, axis=0)

# Multiple warm-up passes help with optimization
for _ in range(3):
    _ = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: dummy_expanded})

print('Warm-up complete')

# Force garbage collection to free memory
gc.collect()

# Get all image files from the directory
image_paths = []
for extension in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
    image_paths.extend(glob.glob(os.path.join(PATH_TO_IMAGE_DIR, f'*.{extension}')))

if not image_paths:
    print(f"No images found in {PATH_TO_IMAGE_DIR}")
    sys.exit(1)

print(f"Found {len(image_paths)} images to process")

# Pre-process all images to avoid disk I/O during detection
print("Pre-processing images...")
processed_images = []
for image_path in image_paths:
    # Load and prepare each image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image: {image_path}, skipping...")
        continue
    
    # Resize image with efficient interpolation method
    height, width = original_image.shape[:2]
    scaling_factor = min(TARGET_SIZE / width, TARGET_SIZE / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    image = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to RGB and expand dimensions
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)
    
    processed_images.append({
        'path': image_path,
        'original': original_image,
        'processed': image_expanded
    })

print(f"Successfully pre-processed {len(processed_images)} images")

# Run detection on all pre-processed images
detection_times = []
for i, img_data in enumerate(processed_images):
    image_path = img_data['path']
    image_name = os.path.basename(image_path)
    original_image = img_data['original']
    image_expanded = img_data['processed']
    
    print(f"Processing image {i+1}/{len(processed_images)}: {image_name}")
    
    # Perform detection and measure time
    start_time = time.time()
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    end_time = time.time()
    
    detection_time = end_time - start_time
    detection_times.append(detection_time)
    print(f"Detection time: {detection_time:.3f} seconds")
    
    # Visualization and saving can be done in a separate loop to not affect timing measurements
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
    
    # Convert back to BGR for OpenCV
    result_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
    
    # Save the result image
    output_path = os.path.join(OUTPUT_DIR, f"detected_{image_name}")
    cv2.imwrite(output_path, result_image)
    
    # Free memory after each image processing
    if i % 10 == 0:
        gc.collect()

# Calculate and display average detection time
if detection_times:
    avg_time = sum(detection_times) / len(detection_times)
    print(f"\nResults:")
    print(f"Processed {len(detection_times)} images")
    print(f"Average detection time: {avg_time:.1f} seconds")
    
    # Additional statistics
    min_time = min(detection_times)
    max_time = max(detection_times)
    print(f"Fastest detection: {min_time:.1f} seconds")
    print(f"Slowest detection: {max_time:.1f} seconds")
else:
    print("No images were successfully processed")

print('End Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.now()))

# Clean up
sess.close()