######## GPU-Optimized Object Detection Using Tensorflow-trained Classifier #########
#
# Modified by Ismail Uzun for NVIDIA RTX A10000 Laptop GPU with CUDA 10
# For TensorFlow 1.13.2 GPU compatibility
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection
# optimized for GPU acceleration on NVIDIA hardware.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
import glob
from datetime import datetime
import gc

# Configure CUDA and TensorFlow GPU settings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Dynamic memory allocation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logging output

# Check if CUDA is available and print GPU information
print("Checking for GPU availability...")
if tf.test.is_gpu_available(cuda_only=True):
    print("CUDA is available")
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    print(f"Tensorflow version: {tf.__version__}")
else:
    print("WARNING: CUDA is not available. The script will run on CPU.")

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

# Fixed dimensions for all images (for consistent batch processing)
# TensorFlow 1.13.2 requires consistent dimensions for batched inference
INPUT_WIDTH = 640
INPUT_HEIGHT = 480

# GPU batch size - adjust based on GPU memory
GPU_BATCH_SIZE = 4  # RTX A10000 has ample memory for batching

# Load the Tensorflow model into memory with GPU optimizations
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
        # Configure session with GPU optimizations for TensorFlow 1.13.2
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=0.8,  # Use 80% of GPU memory
            allow_growth=True,                    # Dynamic memory allocation
        )
        
        config = tf.ConfigProto(
            gpu_options=gpu_options,
            log_device_placement=False,           # Don't log device placement info
            allow_soft_placement=True,            # Fall back to CPU if operation not available on GPU
        )
        
        sess = tf.Session(graph=detection_graph, config=config)

# Define input and output tensors (i.e. data) for the object detection classifier
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Warm-up run with fixed dimensions
print('Performing warm-up inference on GPU...')
dummy_batch = np.zeros((GPU_BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)

# Multiple warm-up passes help with GPU optimization
for _ in range(3):  # Fewer warm-up iterations for TF 1.13.2
    _ = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: dummy_batch})

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

# Pre-process all images with consistent dimensions
print("Pre-processing images...")
processed_images = []
for image_path in image_paths:
    # Load and prepare each image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image: {image_path}, skipping...")
        continue
    
    # Store original dimensions for later use
    original_height, original_width = original_image.shape[:2]
    
    # Resize all images to the same fixed dimensions for batching
    try:
        # Standard OpenCV resize (cuda version not compatible with TF 1.13.2)
        image_resized = cv2.resize(original_image, (INPUT_WIDTH, INPUT_HEIGHT), 
                         interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"Error resizing image {image_path}: {str(e)}")
        continue
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    processed_images.append({
        'path': image_path,
        'original': original_image,
        'processed': image_rgb,
        'original_width': original_width,
        'original_height': original_height
    })

print(f"Successfully pre-processed {len(processed_images)} images")

# Create batches for GPU processing
def create_batches(items, batch_size):
    """Create batches of the specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

image_batches = create_batches(processed_images, GPU_BATCH_SIZE)
print(f"Created {len(image_batches)} batches of size up to {GPU_BATCH_SIZE}")

# Process images in batches for better GPU utilization
detection_times = []
overall_start = time.time()

for batch_idx, batch in enumerate(image_batches):
    print(f"Processing batch {batch_idx+1}/{len(image_batches)}")
    
    # Create input batch for the model - ensure all images have the same dimensions
    # In TensorFlow 1.13.2, all images in a batch must have the same dimensions
    input_batch = np.stack([img['processed'] for img in batch])
    
    # Perform detection on the batch and measure time
    start_time = time.time()
    (boxes, scores, classes, nums) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: input_batch})
    end_time = time.time()
    
    batch_time = end_time - start_time
    per_image_time = batch_time / len(batch)
    detection_times.extend([per_image_time] * len(batch))
    
    print(f"Batch detection time: {batch_time:.3f} seconds ({per_image_time:.3f} seconds per image)")
    
    # Process detection results for each image in the batch
    for i, img_data in enumerate(batch):
        image_path = img_data['path']
        image_name = os.path.basename(image_path)
        original_image = img_data['original']
        
        # Convert normalized boxes back to original image coordinates
        original_height, original_width = original_image.shape[:2]
        
        # Draw the results of the detection on the original image
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Scale boxes to original image size
        scaled_boxes = boxes[i].copy()
        
        vis_util.visualize_boxes_and_labels_on_image_array(
            original_image_rgb,
            scaled_boxes,
            classes[i].astype(np.int32),
            scores[i],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.80)
        
        # Convert back to BGR for OpenCV
        result_image = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
        
        # Save the result image
        output_path = os.path.join(OUTPUT_DIR, f"detected_{image_name}")
        cv2.imwrite(output_path, result_image)
    
    # Free memory after each batch
    gc.collect()

overall_time = time.time() - overall_start

# Calculate and display average detection time
if detection_times:
    avg_time = sum(detection_times) / len(detection_times)
    print(f"\nResults:")
    print(f"Processed {len(detection_times)} images in {overall_time:.1f} seconds")
    print(f"Average detection time per image: {avg_time:.3f} seconds")
    print(f"Overall throughput: {len(detection_times) / overall_time:.2f} images per second")
    
    # Additional statistics
    min_time = min(detection_times)
    max_time = max(detection_times)
    print(f"Fastest detection: {min_time:.3f} seconds")
    print(f"Slowest detection: {max_time:.3f} seconds")
else:
    print("No images were successfully processed")

print('End Timestamp: {:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.now()))

# Clean up
sess.close()