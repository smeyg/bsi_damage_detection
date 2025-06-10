# Borescope Damage Detection Using Faster R-CNN (TensorFlow 1)

This repository provides selected files of a damage detection system for aircraft engine borescope inspection images using the Faster R-CNN with Inception V2 architecture. It is based on TensorFlow 1.13.2 and the TensorFlow Object Detection API v1.

## Repository Structure

```
bsi_damage_detection/
├── LICENSE
├── README.md
├── config/
│   └── training/
│       ├── faster_rcnn_inception.config
│       ├── graph.pbtxt
│       └── labelmap.pbtxt
├── scripts/
│   ├── generate_tfrecord.py
│   ├── Object_detection_image.py
│   ├── Object_detection_image_optimized.py
│   ├── Object_detection_multiple_images_gpu.py
│   ├── Object_detection_multiple_images_optimized.py
│   ├── Object_detection_video.py
│   ├── resizer.py
│   ├── train.py
│   └── xml_to_csv.py
└── slurm/
    └── ts_bsi_fasterrcnn_inception.slurm
```

## File Descriptions

### 📁 `config/training/`

- **`faster_rcnn_inception.config`**: Pipeline configuration file for training the Faster R-CNN Inception V2 model.
- **`graph.pbtxt`**: Graph definition used for visualization in TensorBoard or for editing.
- **`labelmap.pbtxt`**: Label map file that assigns integer labels to class names.

### 📁 `scripts/`

- **`generate_tfrecord.py`**: Converts image datasets and XML annotations into TFRecord format.
- **`Object_detection_image.py`**: Runs object detection on a single image. Implemented by Evan Juras.
- **`Object_detection_image_optimized.py`**: Optimized version with faster inference or streamlined output on a CPU.
- **`Object_detection_multiple_images_gpu.py`**: Batch inference on multiple images using GPU.
- **`Object_detection_multiple_images_optimized.py`**: Further optimized batch inference.
- **`Object_detection_video.py`**: Applies detection to video frames.
- **`resizer.py`**: Utility to resize images.
- **`train.py`**: Launches model training.
- **`xml_to_csv.py`**: Converts XML annotations to CSV.

### 📁 `slurm/`

- **`ts_bsi_fasterrcnn_inception.slurm`**: SLURM script for HPC training.

## Usage Instructions

### 1. Environment Setup

Install dependencies:
```bash
pip install tensorflow==1.13.2 protobuf
in case of GPU availability
pip install tensorflow-gpu==1.13.2 protobuf

pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python
```


Ensure `PYTHONPATH` includes the models directory:
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`/models/research:`pwd`/models/research/slim
```

### 2. Data Preparation

Use [LabelImg](https://github.com/tzutalin/labelImg) to annotate. Then run:
```bash
python scripts/xml_to_csv.py
python scripts/generate_tfrecord.py --csv_input=annotations.csv --output_path=train.record
```
