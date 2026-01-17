# Advanced Computer Vision and Video Analytics
A comprehensive implementation of advanced computer vision techniques and video analytics solutions using Python, OpenCV, and deep learning frameworks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Modules & Implementations](#modules--implementations)
- [Notebook: Image Processing & Digits Classification](#notebook-image-processing--digits-classification)
- [Notebook: Shape and Image Transformations](#notebook-shape-and-image-transformations)
- [Notebook: Edge Detection and Image Segmentation](#notebook-edge-detection-and-image-segmentation)

## Overview
This repository contains implementations of advanced computer vision algorithms and video analytics techniques including:
- Image transformations and geometric operations
- Object detection and tracking
- Face recognition and analysis
- Pose estimation
- Video processing and frame analysis
- Motion detection and activity recognition
- Semantic and instance segmentation
- 3D reconstruction

## Features
✨ **Image Processing**
- Geometric transformations (rotation, scaling, translation, shearing, reflection)
- Image filtering and enhancement
- Morphological operations
- Edge detection and contour analysis

✨ **Video Analytics**
- Real-time video processing
- Frame-level analysis
- Motion tracking
- Temporal analysis

✨ **Deep Learning Integration**
- Pre-trained model support (YOLO, ResNet, etc.)
- Custom model implementations
- Transfer learning examples

✨ **Visualization Tools**
- Matplotlib-based visualization
- Real-time plotting
- Annotated frame display

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup
```bash
git clone https://github.com/anshika1279/Computer-Vision-Implementation.git
cd Computer-Vision-Implementation
pip install -r requirements.txt
```

## Project Structure
- README.md: Project overview and instructions.
- requirements.txt: Python dependencies for all notebooks.
- image_processing_and_digits_classification.ipynb: Image resizing/blur demo plus digits classification with multiple models.
- ShapeAndImageTransformations.ipynb: Shape and image transformation examples.
- edge_detection_and_image_segmentation.ipynb: Edge detection operators and image segmentation techniques.

## Modules & Implementations
- Image resizing with multiple interpolation methods and blurring with box, Gaussian, and bilateral filters.
- Digits classification using sklearn digits dataset with Gaussian Naive Bayes, RBF SVM, and Random Forest, including cross-validation and ROC visualization.

## Notebook: Image Processing & Digits Classification
- File: image_processing_and_digits_classification.ipynb
- Part 1: Resize (linear, nearest, cubic) and blur (box, Gaussian, bilateral) a local image; expects image.png in the repo root and displays a comparison grid.
- Part 2: Train/evaluate classifiers (Gaussian Naive Bayes, RBF SVM, Random Forest) on sklearn digits with 5-fold CV; prints metrics and shows ROC curves.

## Notebook: Shape and Image Transformations
- File: ShapeAndImageTransformations.ipynb
- Part 1: 2D rectangle transformations (translate, scale, rotate, reflect, shear, composite) visualized with Matplotlib.
- Part 2: Image transformations on `input.jpg` using OpenCV (translate, reflect, rotate, scale, crop, shear on x/y) with side-by-side plots.
- Part 3: Additional 2D shape transformations with reusable helpers for translate/scale/rotate/reflect/shear and composite examples.

## Notebook: Edge Detection and Image Segmentation
- File: edge_detection_and_image_segmentation.ipynb
- **Edge Detection**: Implements multiple edge detection operators including:
  - Sobel operator (combined X and Y gradients)
  - Prewitt edge detection
  - Roberts cross operator
  - Canny edge detector
- **Image Segmentation**: Demonstrates various segmentation techniques:
  - Global thresholding (fixed threshold binarization)
  - Adaptive thresholding (local neighborhood-based)
  - Watershed segmentation with morphological operations
- **Preprocessing**: Includes color space conversions (BGR→RGB→Grayscale→Binary) and image metrics calculation
- **Visualization**: Displays all results in a comprehensive grid layout with labeled subplots
- **Outputs**: Saves processed images (edge maps, segmented regions) for further analysis
